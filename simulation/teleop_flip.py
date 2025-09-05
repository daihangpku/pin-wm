import os
import argparse
import torch
import numpy as np
import trimesh
import json
import time
import sys
import yaml
from datetime import datetime
from pathlib import Path
from termcolor import cprint
from tqdm import tqdm
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pynput import keyboard
# Set CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

cur_work_path = os.getcwd()
sys.path.append(cur_work_path)

from builder import Builder
from utils.sys_utils import prepare_output_and_logger
from utils.data_utils import load_twinaligner_dataset
from utils.quaternion_utils import wxyz2xyzw, rotmat_to_quaternion, dq_to_omega_tensor, quaternion_to_rotmat
from utils.traj_utils import smooth_velocity_interpolation
from diff_rendering.gaussian_splatting_2d.utils.camera_utils import cameraList_from_camInfos
from diff_rendering.gaussian_splatting_2d.utils.render_utils import save_img_u8
from diff_rendering.gaussian_splatting_2d.scene.gaussian_model import GaussianModel

from diff_simulation.simulator import Simulator
from diff_simulation.physical_material import Physical_Materials
from diff_simulation.constraints.base import Joint_Type
from simulation.utils.auto_collect.franka_pinwm_controller import flip_pinwm_controller


def create_pinwm_scene(builder, simulator: Simulator, obj_trans, obj_quat, ee_trans, ee_quat):
    """Create the PIN-WM simulation scene"""
    
    # Mesh and URDF paths - check if files exist
    plane_mesh_path = "./envs/asset/plane/plane_collision.obj"
    obj_mesh_path = "diff_rendering/gaussian_splatting_2d/output/small-cup/train/ours_30000/fuse_post_abs.obj"
    ee_mesh_path = "./envs/asset/ee/ee.obj"
    plane_urdf_path = "./envs/asset/plane/plane.urdf"
    obj_urdf_path = "diff_rendering/gaussian_splatting_2d/output/small-cup/train/ours_30000/object.urdf"
    ee_urdf_path = "./envs/asset/franka_ee/ee.urdf"

    plane_mesh = trimesh.load(plane_mesh_path)
    plane_mesh.apply_scale([30, 30, 10])
    plane_physical_material = Physical_Materials(requires_grad=False, device=simulator.device)
    plane_physical_material.no_optimize("mass")
    plane_physical_material.no_optimize('inertia')
    plane_id = simulator.create_mesh_body(
        plane_mesh, plane_physical_material, requires_grad=False,
        urdf=plane_urdf_path if os.path.exists(plane_urdf_path) else None,
        world_position=torch.tensor([0.0, 0.0, -5.008], device=simulator.device),
        world_rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=simulator.device)
    )
    simulator.create_joint(plane_id, Joint_Type.FIX_CONSTRAINT)

    
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_physical_material = Physical_Materials(requires_grad=False, device=simulator.device)
    obj_physical_material.set_material("inertia", [
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01]
    ])
    obj_id = simulator.create_mesh_body(
        obj_mesh, obj_physical_material, requires_grad=False,
        urdf=obj_urdf_path if os.path.exists(obj_urdf_path) else None,
        world_position=obj_trans,
        world_rotation=obj_quat
    )
    

    ee_mesh = trimesh.load(ee_mesh_path)
    ee_physical_material = Physical_Materials(requires_grad=False, device=simulator.device)
    ee_physical_material.set_material("mass", 1000.0)
    ee_physical_material.no_optimize("inertia")
    ee_physical_material.no_optimize("mass")
    ee_id = simulator.create_mesh_body(
        ee_mesh, ee_physical_material, requires_grad=False,
        urdf=ee_urdf_path if os.path.exists(ee_urdf_path) else None,
        world_position=ee_trans,
        world_rotation=ee_quat,
        with_gravity=False
    )
    # simulator.create_joint(ee_id, Joint_Type.NO_TRANS_Z_CONSTRATNT)

    cprint("Scene created successfully", "green")
    return obj_id, ee_id, plane_id


def main(args):

    builder = Builder(args.config)

    
    # Create output directory with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    process_output_dir = os.path.join(args.output_dir, "pinwm_teleop", timestamp)
    os.makedirs(process_output_dir, exist_ok=True)
    
    cprint("*" * 40, "green")
    cprint("  Initializing PIN-WM Teleop", "green")
    cprint(f"  Output: {process_output_dir}", "green")
    cprint("*" * 40, "green")
    
    # Initialize simulator
    sim_device = "cuda" if torch.cuda.is_available() else "cpu"
    cprint(f"Using device: {sim_device}", "cyan")
    
    simulator = Simulator(builder.sim_args.dtime, device=sim_device, vis=True)
    
    # Initial poses (you may want to load these from config or dataset)
    obj_trans = torch.tensor([0.5, 0.0, 0.1], device=simulator.device, dtype=torch.float32)
    obj_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=simulator.device, dtype=torch.float32)
    ee_trans = torch.tensor([0.3, 0.0, 0.2], device=simulator.device, dtype=torch.float32)
    ee_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=simulator.device, dtype=torch.float32)
    

    obj_id, ee_id, plane_id = create_pinwm_scene(builder, simulator, obj_trans, obj_quat, ee_trans, ee_quat)

    
    # Initialize controller
    controller = flip_pinwm_controller(simulator, builder, ee_id, obj_id)
    
    cprint("*" * 40, "green")
    cprint("  PIN-WM Teleop Ready", "green")
    cprint("  Controls:", "green")
    cprint("    Type '1': Reset scene", "green")
    cprint("    Type '2': Start recording", "green")
    cprint("    Type '3': Stop recording", "green")
    cprint("    Type '4': Save trajectory", "green")
    cprint("*" * 40, "green")
    def on_press(key):
        pass
        # print(f"Key pressed: {key}")
    def on_release(key):
        nonlocal process_output_dir
        nonlocal controller
        if key.char == "1":
            print('recollect !!!')
            controller.clean_traj()
            controller.reset_scene()
        elif key.char == "2":
            print('start recording !!!')
            controller.start_record()
        elif key.char == "3":
            print("end recording !!!")
            controller.end_record()
        elif key.char == "4":
            print("save record ...")
            controller.save_traj(process_output_dir)
            controller.reset_scene()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # Main loop
    pbar = tqdm(total=None, bar_format='PIN-WM Teleop: {rate_fmt}', unit='frames')
    
    try:
        while True:
            if not listener.is_alive():
                listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                listener.start()
            # Step simulation and controller
            controller.step()
            pbar.update()
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        cprint('Interrupted by user', 'red')
    finally:
        # Cleanup
        simulator.close()
        pbar.close()
        cprint('PIN-WM Teleop session ended', 'green')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIN-WM Teleop Interface')
    parser.add_argument('--config', type=str, default='configs/milk.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='output/teleop',
                        help='Output directory for recordings')
    
    args = parser.parse_args()
    
    # Load full config
    if args.config.endswith('.yaml'):
        with open(args.config, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        args.config = config_dict
    
    main(args)
