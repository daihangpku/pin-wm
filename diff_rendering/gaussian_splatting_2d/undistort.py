#!/usr/bin/env python3
"""
Image Undistortion Program
Reads camera parameters and distortion coefficients from transforms.json file to perform image undistortion

Main Features:
1. Read camera calibration file containing distortion parameters (OpenCV format)
2. Perform image undistortion processing
3. Generate new calibration files after undistortion:
   - OpenCV format: camera_model set to "PINHOLE", distortion coefficients set to 0
   - NeRF format: Compatible with readCamerasFromTransforms function, uses camera_angle_x

Note: Images after undistortion conform to ideal pinhole camera model with no lens distortion
"""

import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def focal2fov(focal, pixels):
    """Convert focal length to field of view"""
    return 2 * np.arctan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    """Convert field of view to focal length"""
    return pixels / (2 * np.tan(fov / 2))


class ImageUndistorter:
    def __init__(self, transforms_json_path):
        """
        Initialize undistorter
        
        Args:
            transforms_json_path (str): Path to transforms.json file
        """
        self.transforms_json_path = Path(transforms_json_path)
        self.base_dir = self.transforms_json_path.parent
        
        # Read transforms.json
        with open(transforms_json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data['frames'])} frame data")
        print(f"Camera model: {self.data.get('camera_model', 'UNKNOWN')}")
    
    
    
    def get_camera_matrix_and_distortion(self):
        """
        Get OpenCV format camera matrix and distortion coefficients
        
        Returns:
            tuple: (camera_matrix, dist_coeffs)
        """
        # Camera intrinsic matrix
        camera_matrix = np.array([
            [self.camera_params['fl_x'], 0, self.camera_params['cx']],
            [0, self.camera_params['fl_y'], self.camera_params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients [k1, k2, p1, p2, k3]
        # Assume no k3 here, if available can extract from data
        dist_coeffs = np.array([
            self.camera_params['k1'],
            self.camera_params['k2'],
            self.camera_params['p1'],
            self.camera_params['p2'],
            0.0  # k3
        ], dtype=np.float32)
        
        return camera_matrix, dist_coeffs
    
    def get_camera_matrix_and_distortion_from_frame(self, frame):
        """
        Get OpenCV format camera matrix and distortion coefficients from specific frame
        
        Args:
            frame (dict): Frame data containing camera parameters
            
        Returns:
            tuple: (camera_matrix, dist_coeffs)
        """
        # Camera intrinsic matrix
        camera_matrix = np.array([
            [frame['fl_x'], 0, frame['cx']],
            [0, frame['fl_y'], frame['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([
            frame['k1'],
            frame['k2'],
            frame['p1'],
            frame['p2'],
            0.0  # k3
        ], dtype=np.float32)
        
        return camera_matrix, dist_coeffs
    
    def undistort_image(self, image_path, frame_params=None, output_path=None, show_comparison=False):
        """
        Undistort a single image
        
        Args:
            image_path (str): Input image path
            frame_params (dict): Specific frame camera parameters, use default if None
            output_path (str): Output image path, do not save if None
            show_comparison (bool): Whether to show comparison images
            
        Returns:
            tuple: (original_image, undistorted_image, new_camera_matrix, roi)
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Get camera parameters
        if frame_params is not None:
            camera_matrix, dist_coeffs = self.get_camera_matrix_and_distortion_from_frame(frame_params)
        else:
            camera_matrix, dist_coeffs = self.get_camera_matrix_and_distortion()
        
        # Calculate optimal new camera matrix
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Crop image (optional)
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]
        
        # Save result
        if output_path:
            cv2.imwrite(str(output_path), undistorted)
        
        return img, undistorted, new_camera_matrix, roi
    
    def undistort_all_images(self, output_dir=None, max_images=None):
        """
        Undistort all images
        
        Args:
            output_dir (str): Output directory, create undistort_images subdirectory if None
            max_images (int): Maximum number of images to process, None means process all
        """
        if output_dir is None:
            output_dir = self.base_dir / "undistort_images"
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Process images
        frames_to_process = self.data['frames']
        if max_images:
            frames_to_process = frames_to_process[:max_images]
        
        print(f"Starting undistortion processing for {len(frames_to_process)} images")
        
        for i, frame in enumerate(tqdm(frames_to_process, desc="Undistortion processing")):
            # Build input and output paths
            file_path = frame['file_path']
            
            # Build input path
            input_path = self.base_dir / file_path
            if not input_path.exists():
                # If file doesn't exist, try adding .png extension
                if not file_path.endswith(('.png', '.jpg', '.jpeg')):
                    input_path = self.base_dir / (file_path + '.png')
            
            # Build output path - use original filename directly without adding extra extensions
            output_filename = Path(file_path).name
            # Ensure output file has extension
            if not output_filename.endswith(('.png', '.jpg', '.jpeg')):
                output_filename += '.png'
            output_path = output_dir / output_filename
            
            if not input_path.exists():
                print(f"Warning: Image file does not exist: {input_path}")
                continue
            
            try:
                # Use specific frame parameters for undistortion
                self.undistort_image(input_path, frame, output_path)
            except Exception as e:
                print(f"Error processing image {input_path}: {e}")
        
        print(f"Undistortion completed, results saved to: {output_dir}")
    
    def create_nerf_transforms(self, output_path=None, test_ratio=0.1, test_output_path=None):
        """
        Create NeRF format transforms.json file (compatible with readCamerasFromTransforms function)
        
        Args:
            output_path (str): Output file path for training data
            test_ratio (float): Ratio of frames to use for test set (default: 0.1, meaning 10%)
            test_output_path (str): Output file path for test data
            
        Description:
            Images after undistortion have corrected lens distortion and conform to ideal pinhole camera model.
            Only keep camera_angle_x and transform_matrix, no distortion parameters.
            Automatically splits data into training and test sets.
        """
        if output_path is None:
            output_path = self.base_dir / "transforms_train.json"
        if test_output_path is None:
            test_output_path = self.base_dir / "transforms_test.json"
        
        # Copy original data
        new_data = self.data.copy()
        new_data['frames'] = []
        
        # Copy for test data
        test_data = self.data.copy()
        test_data['frames'] = []
        
        print(f"Calculating undistortion parameters individually for {len(self.data['frames'])} frames...")
        
        # Calculate number of test frames
        total_frames = len(self.data['frames'])
        num_test_frames = max(1, int(total_frames * test_ratio))
        
        # Select test frame indices (evenly distributed)
        test_indices = set()
        if num_test_frames < total_frames:
            step = total_frames // num_test_frames
            for i in range(0, total_frames, step):
                if len(test_indices) < num_test_frames:
                    test_indices.add(i)
        else:
            test_indices = set(range(total_frames))
        
        print(f"Selected {len(test_indices)} frames for test set out of {total_frames} total frames")
        
        # Get parameters from first frame to calculate camera_angle_x (assume all frames have same camera intrinsics)
        first_frame = self.data['frames'][0]
        camera_matrix, dist_coeffs = self.get_camera_matrix_and_distortion_from_frame(first_frame)
        h, w = first_frame['h'], first_frame['w']
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # Calculate new focal length and camera_angle_x
        new_fl_x = new_camera_matrix[0, 0]
        new_w = roi[2] if roi[2] > 0 else w
        camera_angle_x = 2 * np.arctan(new_w / (2 * new_fl_x))
        new_data['camera_angle_x'] = float(camera_angle_x)
        test_data['camera_angle_x'] = float(camera_angle_x)
        # Remove opencv specific fields
        if 'camera_model' in new_data:
            del new_data['camera_model']
        
        # Update parameters for each frame
        for i, frame in enumerate(tqdm(self.data['frames'], desc="Calculating new parameters")):
            new_frame = {}
            
            # Basic fields
            # Ensure correct file path, remove extension (readCamerasFromTransforms will add automatically)
            original_file_path = frame['file_path']
            if original_file_path.startswith('images/'):
                # Remove extension since readCamerasFromTransforms will add automatically
                base_path = original_file_path.replace('images/', 'undistort_images/')
                # Remove extension
                if base_path.endswith(('.png', '.jpg', '.jpeg')):
                    base_path = str(Path(base_path).with_suffix(''))
                new_file_path = base_path
            else:
                # If original path doesn't have images/ prefix, use undistort_images/ directly with filename (no extension)
                filename = Path(original_file_path).stem  # Use stem to get filename without extension
                new_file_path = f"undistort_images/{filename}"
            
            new_frame['file_path'] = new_file_path
            new_frame['transform_matrix'] = frame['transform_matrix']
            
            image_filename = Path(original_file_path).stem
            new_frame['mask_path'] = f"masks/{image_filename}"

            # Add to appropriate dataset
            if i in test_indices:
                test_data['frames'].append(new_frame)
            else:
                new_data['frames'].append(new_frame)
        
        # Save new transforms.json files
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        test_output_path = Path(test_output_path)
        test_output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        with open(test_output_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"NeRF format transforms_train.json saved to: {output_path}")
        print(f"NeRF format transforms_test.json saved to: {test_output_path}")
        print(f"Training set: {len(new_data['frames'])} frames")
        print(f"Test set: {len(test_data['frames'])} frames")


def main():
    parser = argparse.ArgumentParser(description='Image undistortion tool')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of frames to use for test set (default: 0.1)')
    args = parser.parse_args()


    base_dir = Path(args.dataset_dir)
    transforms_json = base_dir / "transforms.json"
    
    print("=== Image Undistortion Processing ===")
    print(f"Reading configuration file: {transforms_json}")
    
    # Check if configuration file exists
    if not transforms_json.exists():
        print(f"Error: Configuration file does not exist {transforms_json}")
        return
    
    # Create undistorter
    undistorter = ImageUndistorter(transforms_json)
    
    # Process all images
    undistorter.undistort_all_images()
    
    # Create NeRF format transforms.json
    print("Creating NeRF format transforms_train.json and transforms_test.json...")
    undistorter.create_nerf_transforms(test_ratio=args.test_ratio)
    
    print("\n=== Processing Complete ===")
    print("Generated files:")
    print("- transforms_train.json (NeRF format, compatible with readCamerasFromTransforms)")
    print("- transforms_test.json (NeRF format, test set for evaluation)")


if __name__ == "__main__":
    main()

