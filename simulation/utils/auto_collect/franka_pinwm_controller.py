import sys
import numpy as np
import os
import torch
import trimesh
import json
import h5py
from tqdm import tqdm
import time

sys.path.insert(0, os.getcwd())
from simulation.utils.auto_collect.ik_solver import FrankaSolver
from simulation.utils.auto_collect.utils import rotate_quaternion_around_world_z_axis
from simulation.utils.constants import DEFAULT_JOINT_ANGLES, JOINT_NAMES
from diff_simulation.simulator import Simulator
from diff_simulation.physical_material import Physical_Materials
from diff_simulation.constraints.base import Joint_Type
from utils.quaternion_utils import wxyz2xyzw, rotmat_to_quaternion, dq_to_omega_tensor, xyzw2wxyz
from utils.traj_utils import smooth_velocity_interpolation

try:
    import rospy
    from std_msgs.msg import Float64MultiArray, Bool
    ROSPY_ENABLED = True
except:
    ROSPY_ENABLED = False
    print("rospy not loaded. Teleop mode will be disabled.")

class franka_pinwm_controller:

    
    def __init__(self, simulator, builder, ee_id, obj_id, 
                 default_gripper_state=False, default_joint_angles=DEFAULT_JOINT_ANGLES):

        self.simulator = simulator
        self.builder = builder
        self.device = simulator.device
        self.teleop = True
        
        # IK solvers
        self.franka_solver = FrankaSolver(ik_type="motion_gen", ik_sim=True, simulator="pinwm", no_solver=self.teleop)
        
        
        # Control state
        self.default_joint_angles = default_joint_angles
        self.current_joint_control = np.array(self.default_joint_angles[:7])
        default_ee_pos, default_ee_quat = self.franka_solver.compute_fk(self.default_joint_angles[:7])
        self.default_ee_pos = torch.tensor(default_ee_pos, device=self.device, dtype=torch.float32)
        self.default_ee_quat = torch.tensor(default_ee_quat, device=self.device, dtype=torch.float32)
        #self.default_ee_quat = xyzw2wxyz(self.default_ee_quat)  # Convert to wxyz
        self.current_ee_control = np.concatenate([default_ee_pos, default_ee_quat])
        print(f"Default EE position: {self.default_ee_pos}, Default EE quaternion: {self.default_ee_quat}")
        self.simulator.change_body_pose(ee_id, self.default_ee_pos, self.default_ee_quat)
        self.current_gripper_control = np.array([0.0, 0.0])
        self.ee_id = ee_id
        self.obj_id = obj_id
        # Recording
        self.record_started = False
        self.timestamp = 0
        self.record = []
        self.traj_cnt = 0
        
        # Initialize ROS teleoperation if enabled
        if self.teleop:
            if ROSPY_ENABLED:
                self.init_teleop()
            else:
                raise RuntimeError("rospy is not enabled! Install ROS first if in teleop mode.")
    
    def init_teleop(self):
        """
        Initialize ROS teleoperation interface.
        """
        rospy.init_node('genesis_sim', anonymous=True)

        # Publishers for sending robot state
        self.pub_joint = rospy.Publisher('/genesis/joint_states', Float64MultiArray, queue_size=1)
        self.pub_ee = rospy.Publisher('/genesis/ee_states', Float64MultiArray, queue_size=1)

        # Subscribers for receiving control commands
        # self.sub_ee_control = rospy.Subscriber(
        #     "/genesis/ee_control",
        #     Float64MultiArray,
        #     self._callback_ee_control,
        #     queue_size=1,
        # )
        # Keep joint_control for backward compatibility
        self.sub_joint_control = rospy.Subscriber(
            "/genesis/joint_control",
            Float64MultiArray,
            self._callback_joint_control,
            queue_size=1,
        )
        self.sub_gripper_control = rospy.Subscriber(
            "/genesis/gripper_control",
            Bool,
            self._callback_gripper_control,
            queue_size=1,
        )
    
    def _callback_ee_control(self, msg):
        """
        ROS callback for direct end-effector pose control commands.
        This is the primary control method for PIN-WM.
        
        Args:
            msg (Float64MultiArray): End-effector pose [x, y, z, qw, qx, qy, qz]
        """
        self.current_ee_control = np.array(msg.data)
        ee_pos = torch.tensor(msg.data[:3], device=self.device, dtype=torch.float32)
        ee_quat = torch.tensor(msg.data[3:7], device=self.device, dtype=torch.float32)  # wxyz

        current_ee_pos, current_ee_quat = self.simulator.get_body_pose(self.ee_id)
        ee_pos_velocity = (ee_pos - current_ee_pos) / 0.1
        ee_quat_velocity = dq_to_omega_tensor(current_ee_quat, ee_quat, 0.1)
        self.simulator.change_body_vel(self.ee_id, ee_pos_velocity, ee_quat_velocity)


    def _callback_joint_control(self, msg):
        self.current_joint_control = np.array(msg.data)
        #print(f"Current joint control: {self.current_joint_control}")
        current_ee_pos_control, current_ee_quat_control = self.franka_solver.compute_fk(self.current_joint_control[:7])
        self.current_ee_control = np.concatenate([current_ee_pos_control, current_ee_quat_control])
        
        current_ee_pos_control = torch.tensor(current_ee_pos_control, device=self.device, dtype=torch.float32)
        current_ee_quat_control = torch.tensor(current_ee_quat_control, device=self.device, dtype=torch.float32)

        current_ee_pos, current_ee_quat = self.simulator.get_body_pose(self.ee_id)
        ee_pos_velocity = (current_ee_pos_control - current_ee_pos) / 0.07
        ee_quat_velocity = dq_to_omega_tensor(current_ee_quat, current_ee_quat_control, 0.07)
        self.simulator.change_body_vel(self.ee_id, ee_pos_velocity, ee_quat_velocity)

    def _callback_gripper_control(self, msg):
        """
        ROS callback for gripper control commands.
        """
        self.current_gripper_control = np.array(msg.data)

    def publish_states(self):
        """
        Publish current robot states to ROS topics.
        """
        if not self.teleop or not ROSPY_ENABLED:
            return
        
        # Publish joint states (for compatibility)
        joint_pos_list = self.current_joint_control.flatten().tolist()
        joint_pos_msg = Float64MultiArray(data=joint_pos_list)
        self.pub_joint.publish(joint_pos_msg)
        
        # Publish end-effector states (primary for PIN-WM)
        ee_pos, ee_quat = self.get_ee_pose()
        # Format: [x, y, z, qw, qx, qy, qz]
        current_ee_pose = np.concatenate([ee_pos.cpu().detach().numpy(), ee_quat.cpu().detach().numpy()])
        current_ee_pose_list = current_ee_pose.flatten().tolist()
        current_ee_msg = Float64MultiArray(data=current_ee_pose_list)
        self.pub_ee.publish(current_ee_msg)
        
    
    def get_ee_pose(self):
        """
        Returns:
            tuple: (position, orientation) as torch tensors
        """
        return self.simulator.get_body_pose(self.ee_id)
    
    def get_object_pose(self, obj_id):
        """
        Get object pose.
        
        Args:
            obj_id (int): Object body ID
            
        Returns:
            tuple: (position, orientation) as torch tensors
        """
        return self.simulator.get_body_pose(obj_id)
        
    def step(self):
        """
        Step the simulation forward.
        """
        # Publish states for ROS teleoperation
        if self.teleop:
            self.publish_states()
            
        self.simulator.step()
        
        if self.record_started:
            self.record_step()
    
    def record_step(self):
        raise NotImplementedError("record_step() must be implemented in subclass.")
    
    def start_record(self):
        """
        Start recording trajectory.
        """
        self.record_started = True
        self.record = []
        self.timestamp = 0
    
    def end_record(self):
        """
        End recording trajectory.
        """
        self.record_started = False
    
    def save_traj(self, output_dir):
        print(f"Saving trajectory to {output_dir} ...")
        """
        Save recorded trajectory to files.
        
        Args:
            output_dir (str): Output directory path
        """
        
        def save_dict_to_hdf5(dic, filename):
            with h5py.File(filename, 'w') as h5file:
                _save_dict_to_hdf5(dic, h5file)

        def _save_dict_to_hdf5(dic, h5grp):
            for key, item in dic.items():
                if isinstance(item, dict):
                    subgroup = h5grp.create_group(key)
                    _save_dict_to_hdf5(item, subgroup)
                elif isinstance(item, np.ndarray) or isinstance(item, list):
                    h5grp.create_dataset(key, data=np.array(item), compression="gzip")
                else:
                    print(f"Unknown item {key}, {type(item)}")
        
        self.record_started = False
        current_traj_dir = os.path.join(output_dir, "{:05d}".format(self.traj_cnt))
        os.makedirs(current_traj_dir, exist_ok=True)
        
        for idx, state in enumerate(self.record):
            save_dict_to_hdf5(state, os.path.join(current_traj_dir, f"{idx}.h5"))
        
        self.clean_trajectory()
        self.traj_cnt += 1
    
    def clean_trajectory(self):
        """
        Clean recorded trajectory data.
        """
        self.record = []


class push_pinwm_controller(franka_pinwm_controller):
    def __init__(self, simulator, builder, ee_id, obj_id, static_obj_id=None,
                 default_gripper_state=False, default_joint_angles=DEFAULT_JOINT_ANGLES):

        super().__init__(simulator, builder, ee_id, obj_id, default_gripper_state, default_joint_angles)
        if static_obj_id is not None:
            self.static_obj_id = static_obj_id
            self.static_obj_init_pos, self.static_obj_init_quat = self.simulator.get_body_pose(static_obj_id)
        else:
            self.static_obj_id = None

    def reset_scene(self):
        if self.static_obj_id is not None:
            self.simulator.change_body_pose(self.static_obj_id, self.static_obj_init_pos, self.static_obj_init_quat)
        self.simulator.change_body_pose(self.ee_id, self.default_ee_pos, self.default_ee_quat)

    
    def record_step(self):
        """
        Record current state for trajectory logging.
        """
        obj_pos, obj_quat = self.get_object_pose(self.obj_id)
        static_obj_pos, static_obj_quat = self.get_object_pose(self.static_obj_id) if self.static_obj_id is not None else (None, None)
        # Get end-effector state
        ee_pos, ee_quat = self.get_ee_pose()
        current_ee_state = np.concatenate([ee_pos.cpu().numpy(), ee_quat.cpu().numpy()])

        self.record.append({
            "timestamp": np.array([self.timestamp]),
            "joint_states": np.concatenate([self.current_joint_control, [0.0, 0.0]]),
            "ee_states": current_ee_state,
            "joint_control": self.current_joint_control,
            "ee_control": self.current_ee_control,
            "gripper_control": np.array([0.0, 0.0]),
            "object_states": {
                "active": np.concatenate([obj_pos.cpu().numpy(), obj_quat.cpu().numpy()]),
                "passive": np.concatenate([static_obj_pos.cpu().numpy() if static_obj_pos is not None else np.zeros(3), static_obj_quat.cpu().numpy() if static_obj_quat is not None else np.zeros(4)]),
            },
        })
        self.timestamp += 1

class flip_pinwm_controller(franka_pinwm_controller):
    def __init__(self, simulator, builder, ee_id, obj_id,
                 default_gripper_state=False, default_joint_angles=DEFAULT_JOINT_ANGLES):

        super().__init__(simulator, builder, ee_id, obj_id, default_gripper_state, default_joint_angles)
    def reset_scene(self):
        
        self.simulator.change_body_pose(self.ee_id, self.default_ee_pos, self.default_ee_quat)

    
    def record_step(self):
        """
        Record current state for trajectory logging.
        """
        obj_pos, obj_quat = self.get_object_pose(self.obj_id)
        static_obj_pos, static_obj_quat = self.get_object_pose(self.static_obj_id) if self.static_obj_id is not None else (None, None)
        # Get end-effector state
        ee_pos, ee_quat = self.get_ee_pose()
        current_ee_state = np.concatenate([ee_pos.cpu().numpy(), ee_quat.cpu().numpy()])

        self.record.append({
            "timestamp": np.array([self.timestamp]),
            "joint_states": np.concatenate([self.current_joint_control, [0.0, 0.0]]),
            "ee_states": current_ee_state,
            "joint_control": self.current_joint_control,
            "ee_control": self.current_ee_control,
            "gripper_control": np.array([0.0, 0.0]),
            "object_states": {
                "active": np.concatenate([obj_pos.cpu().numpy(), obj_quat.cpu().numpy()]),
            },
        })
        self.timestamp += 1
