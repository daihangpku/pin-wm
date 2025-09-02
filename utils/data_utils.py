import torch
import numpy as np
import imageio
from tqdm import tqdm
import os,json,sys
import cv2
from PIL import Image
from typing import NamedTuple


from diff_rendering.gaussian_splatting_2d.scene.dataset_readers import SceneInfo,BasicPointCloud,SH2RGB,storePly,fetchPly,CameraInfo,fov2focal,focal2fov,getNerfppNorm
from diff_rendering.gaussian_splatting_2d.utils.camera_utils import cameraList_from_camInfos, loadCam

from diff_simulation.force.constant_force import Constant_Force
from scipy.spatial.transform import Rotation as RR

def readDataFromJson(path, json_file, white_background,extension):
    with open(os.path.join(path, json_file)) as json_file:
        contents = json.load(json_file)

    ee_init_position = np.array(contents["ee_init_position"],dtype=np.float32)
    ee_goal_position = np.array(contents["ee_goal_position"],dtype=np.float32)

    cam_infos = []
    image_datas = contents["image_datas"]
    for entry in image_datas:
        image_path = entry["file_path"]
        cam_id, frame_id = [int(i) for i in image_path.split("/")[-1].rstrip(extension).lstrip("r_").split("_")]
        if frame_id < 0:
            continue
        
        c2w = entry["c2w"]
        c2w.append([0.0,0.0,0.0,1.0])
        c2w = np.array(c2w)
        # temp = np.copy(c2w[1, :])
        # c2w[1, :]=c2w[2, :]
        # c2w[2, :]=temp
        
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]      
        

        # image_path = os.path.join(path, image_path.replace('r', 'm'))
        image_path = os.path.join(path, 'm'.join(image_path.rsplit('r', 1)))
        image_name = os.path.basename(image_path).split(".")[0]
        
        image = Image.open(image_path)
        
        im_data = np.array(image.convert("RGBA"))
        
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        arr = np.concatenate((arr, norm_data[:,:,3][..., np.newaxis]), axis=-1)
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

        intrinsic = entry["intrinsic"]
        fovx = np.arctan(0.5*image.size[0]/intrinsic[0][0])*2
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx            
        
        cam_infos.append(CameraInfo(uid=cam_id,frameid=frame_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
        
    if "pose_datas" in contents:
        pose_datas = np.array(contents["pose_datas"])
    else:
        pose_datas = None

    return cam_infos,ee_init_position,ee_goal_position,pose_datas


def load_pybullet_dynamic_dataset(data_args):
    dynamic_datasets = []
    for index in range(data_args.train_num+data_args.test_num):
        if index < data_args.train_num:
            json_name = "dynamic_train{}.json".format(index)
        else:
            json_name = "dynamic_test{}.json".format(index-data_args.train_num)
        dynamic_cam_infos,ee_init_position,ee_goal_position,pose_datas = readDataFromJson(data_args.data_path, json_name,
                                                                  data_args.white_background, extension=".png")
        dynamic_datasets.append(
            {
                "cam_infos":dynamic_cam_infos,
                'ee_init_position':ee_init_position,
                "ee_goal_position":ee_goal_position,
                "pose_datas":pose_datas
            }
        )
    return dynamic_datasets[:data_args.train_num],dynamic_datasets[data_args.train_num:data_args.test_num]


def get_all_frames_train_camera(cam_id,all_train_cam_infos,dataset):
    all_frames_train_camera_infos = []
    for cam in all_train_cam_infos:
        if cam.uid == cam_id:
            all_frames_train_camera_infos.append(cam)
    all_frames_train_camera_infos = sorted(all_frames_train_camera_infos,key=lambda x: x.frameid)
    all_frames_train_camera = cameraList_from_camInfos(all_frames_train_camera_infos, 1.0,dataset)
    return all_frames_train_camera

def save_images(images,image_paths):
    for image,image_path in zip(images,image_paths):
        cv2.imwrite(image_path, image)


def load_twinaligner_dataset(data_args):
    """
    Load TwinAligner dataset from dataset/banana directory
    """
    CAM_EXTRINSICS_PATH = "envs/asset/cam_extr_init.txt"
    cam_extrinsics = np.loadtxt(CAM_EXTRINSICS_PATH)
    datasets = []
    dataset_path = data_args.data_path
    
    # Get all trajectory directories
    traj_dirs = [d for d in os.listdir(dataset_path) if d.startswith('traj_') and os.path.isdir(os.path.join(dataset_path, d))]
    traj_dirs.sort()
    
    for traj_dir in traj_dirs:
        traj_path = os.path.join(dataset_path, traj_dir)
        
        # Load camera intrinsics
        cam_K_path = os.path.join(traj_path, 'cam_K.txt')
        cam_K = np.loadtxt(cam_K_path)
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]
        
        # Load frame data
        frame_json_path = os.path.join(traj_path, 'frame.json')
        with open(frame_json_path, 'r') as f:
            frame_data = json.load(f)
            
        # Load init data
        init_json_path = os.path.join(traj_path, 'init.json')
        with open(init_json_path, 'r') as f:
            init_data = json.load(f)
            
        # Load control data
        control_json_path = os.path.join(traj_path, 'control.json')
        with open(control_json_path, 'r') as f:
            control_data = json.load(f)
            
        # Load poses
        pose_path = os.path.join(traj_path, 'pose.npy')
        poses = np.load(pose_path)  # Shape: (num_frames, 4, 4) - 这个pose是在相机坐标系中的pose
        
        # Convert poses from camera coordinate system to world coordinate system
        # cam_extrinsics is the transformation matrix from camera to world (4x4)
        poses_world = []
        for pose in poses:
            # Transform pose from camera coordinates to world coordinates
            # pose_world = cam_extrinsics @ pose
            pose_world = np.dot(np.linalg.inv(cam_extrinsics), pose)
            poses_world.append(pose_world)
        poses = np.array(poses_world)
        
        # Load RGB images and create camera info
        rgb_dir = os.path.join(traj_path, 'rgb')
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        
        dt = 0.3
        t_start = control_data[0]["ros_timestamp"]
        t_end = control_data[-1]["ros_timestamp"]
        valid_frame_id = [i for i, frame in enumerate(frame_data) if t_start - dt <= frame["ros_timestamp"] <= t_end + dt]

        rgb_files = [rgb_files[i] for i in valid_frame_id]
        images = []
        for rgb_file in rgb_files:
            image_path = os.path.join(rgb_dir, rgb_file)
            image = Image.open(image_path)
            # Convert PIL image to tensor (C, H, W) with values in [0, 1]
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            images.append(image_tensor)

        
        # Organize trajectory data
        trajectory_data = {
            'cam_K': cam_K,
            'cam_extrinsics': cam_extrinsics,
            'pose_datas': poses[valid_frame_id],
            'frame_data': [frame_data[i] for i in valid_frame_id],
            'init_data': init_data,
            'control_data': control_data,
            'rgb_images': images
        }
        camera_info = create_camera_info_from_trajectory_data(trajectory_data, frame_idx=0)
        trajectory_data['cam_infos'] = [camera_info]
        datasets.append(trajectory_data)
    
    # Split into train and test based on data_args
    train_ratio = getattr(data_args, 'train_ratio', 1.0)
    num_train = int(len(datasets) * train_ratio)
    num_test = len(datasets) - num_train
    
    train_datasets = datasets[:num_train]
    test_datasets = datasets[num_train:num_train + num_test] if num_test > 0 else []
    
    return train_datasets, test_datasets

def create_camera_info_from_trajectory_data(trajectory_data, frame_idx, cam_id=0):
    """
    Create a CameraInfo object from trajectory data for rendering
    
    Args:
        trajectory_data: Dictionary containing trajectory information
        frame_idx: Index of the frame to create camera for
        cam_id: Camera ID (default: 0)
    
    Returns:
        CameraInfo object that can be used with the rendering pipeline
    """
    # Get camera intrinsics
    cam_K = trajectory_data['cam_K']
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    
    # Convert pose to camera extrinsics (R, T)
    # pose is object-to-world, we need world-to-camera
    c2w = trajectory_data["cam_extrinsics"]
    w2c = np.linalg.inv(c2w)  # world-to-camera
    
    R = w2c[:3, :3].T  # Transpose for the expected format
    T = w2c[:3, 3]
    
    # Get RGB image
    rgb_image = trajectory_data['rgb_images'][frame_idx]
    
    # Convert tensor to PIL Image if needed
    if torch.is_tensor(rgb_image):
        if rgb_image.dim() == 3 and rgb_image.shape[0] == 3:  # (C, H, W)
            rgb_array = (rgb_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            rgb_array = (rgb_image.numpy() * 255).astype(np.uint8)
        image = Image.fromarray(rgb_array)
    else:
        image = rgb_image
    
    # Get image dimensions
    width, height = image.size
    
    # Calculate Field of View from intrinsics
    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    
    # Create CameraInfo
    camera_info = CameraInfo(
        uid=cam_id,
        frameid=frame_idx,
        R=R,
        T=T,
        FovY=fovy,
        FovX=fovx,
        image=image,
        image_path=f"frame_{frame_idx:06d}.png",
        image_name=f"frame_{frame_idx:06d}",
        width=width,
        height=height
    )
    
    return camera_info



