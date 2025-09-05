import argparse
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 创建保存图像的目录
output_dir = os.path.join(os.path.dirname(__file__), "output/images")

def transform(pos, quaternion):
    rotation_matrix = quaternion_to_rotation_matrix(quaternion) 

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # Set the rotation part
    transformation_matrix[:3, 3] = pos  # Set the translation part

    return transformation_matrix
def save_rgbd(rgb_data, depth_data, output_dir = output_dir, frame_count = 0):
    os.makedirs(output_dir, exist_ok=True)
    if rgb_data is not None and rgb_data.shape[0] > 0:
        #print(rgb_data)
        #input()
        rgb_image = (rgb_data[0] ).astype(np.uint8)
                # OpenCV使用BGR格式，需要转换颜色通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.join(output_dir,"rgb"), exist_ok=True)
        cv2.imwrite(os.path.join(os.path.join(output_dir,"rgb"), f"frame_{frame_count:05d}.png"), bgr_image)

            # 处理并保存深度图像
    if depth_data is not None and depth_data.shape[0] > 0:
        depth_image = np.uint16(depth_data[0]*1000)
        # 归一化深度值到0-255范围
        #depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
        os.makedirs(os.path.join(output_dir,"depth"), exist_ok=True)
        cv2.imwrite(os.path.join(os.path.join(output_dir,"depth"), f"frame_{frame_count:05d}.png"), depth_image)


import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):
    q = np.array(q) 
    """
    将四元数 (w, x, y, z) 转换为 3x3 旋转矩阵
    输入:
        q: 四元数，np数组形状为 (4,) 或 (n, 4)，代表 (w, x, y, z)
    输出:
        返回对应的旋转矩阵 (3x3) 或 (n, 3, 3)
    """
    if q.ndim == 1:  # 如果输入是一个四元数 (w, x, y, z)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
        ])
        return R
    else:  # 如果输入是多个四元数 (n, 4)
        matrices = []
        for i in range(q.shape[0]):
            w, x, y, z = q[i]
            R = quaternion_to_rotation_matrix([w, x, y, z])
            matrices.append(R)
        return np.array(matrices)
def rotation_matrix_to_quaternion(matrix):
    """
    将 3×3 旋转矩阵转换为四元数 (w,x, y, z, )
    
    参数:
    - matrix: np.array, 形状为 (3,3) 的旋转矩阵
    
    返回:
    - quaternion: np.array, 形状为 (4,) 的四元数 (w,x, y, z, )
    """
    rotation = R.from_matrix(matrix)  # 从旋转矩阵创建 Rotation 对象
    quaternion = rotation.as_quat()   # 转换为四元数 (x, y, z, w)
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    
    return quaternion


def invert_homogeneous_matrix(matrix):
    """
    Invert a 4x4 homogeneous transformation matrix.

    Parameters:
    - matrix: np.array, shape (4, 4), the homogeneous transformation matrix.

    Returns:
    - inverted_matrix: np.array, shape (4, 4), the inverted homogeneous matrix.
    """
    R = matrix[:3, :3]  # Extract rotation matrix
    T = matrix[:3, 3]   # Extract translation vector

    # Compute the inverse
    R_inv = R.T  # Transpose of rotation matrix
    T_inv = -R_inv @ T  # Inverse translation

    # Construct the inverted matrix
    inverted_matrix = np.eye(4)
    inverted_matrix[:3, :3] = R_inv
    inverted_matrix[:3, 3] = T_inv

    return inverted_matrix
def rotate_around_world_z_axis(axis_matrix, angle_degrees):
    """
    Rotate a 3x3 coordinate axis matrix around the world z-axis by a specified angle.

    Parameters:
    - axis_matrix (np.ndarray): A 3x3 matrix representing the coordinate axes.
                                Each column is a unit vector (x-axis, y-axis, z-axis).
    - angle_degrees (float): The angle to rotate around the world z-axis, in degrees.

    Returns:
    - rotated_axis_matrix (np.ndarray): The rotated 3x3 matrix.
    """
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Define the rotation matrix around the z-axis
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0,                     0,                     1]
    ])

    # Apply the rotation to the axis matrix
    rotated_axis_matrix = rotation_matrix @ axis_matrix

    return rotated_axis_matrix
def rotate_quaternion_around_world_z_axis(quaternion, angle_degrees):
    """
    Rotate a quaternion representing a coordinate axis around the world z-axis by a specified angle.

    Parameters:
    - quaternion (np.ndarray): A quaternion representing the orientation (shape: [4], format: [x,y,z,w]).
    - angle_degrees (float): The angle to rotate around the world z-axis, in degrees.

    Returns:
    - rotated_quaternion (np.ndarray): The rotated quaternion (shape: [4], format: [x, y, z, w]).
    """
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Define the rotation quaternion around the z-axis
    rotation_quat = Rotation.from_euler('z', angle_degrees, degrees=True).as_quat()  # [x, y, z, w]

    # Convert input quaternion to a Rotation object
    input_matrix = Rotation.from_quat(quaternion).as_matrix()  # [x, y, z, w]
    output_matrix = input_matrix
    for i in range(3):
        axis = input_matrix[:, i].reshape(-1)
        x_ = axis[0]*np.cos(angle_radians) - axis[1]*np.sin(angle_radians)
        y_ = axis[0]*np.sin(angle_radians) + axis[1]*np.cos(angle_radians)
        z_ = axis[2]
        output_matrix[:,i] = np.array([x_,y_,z_]).reshape(-1,3)


    rotated_quaternion = Rotation.from_matrix(output_matrix).as_quat() 

    # visualize_axes(quaternion, rotated_quaternion)
    return rotated_quaternion

def visualize_axes(quaternion, rotated_quaternion):
    """
    Visualize the original and rotated axes in 3D.

    Parameters:
    - quaternion (np.ndarray): Original quaternion (shape: [4], format: [w, x, y, z]).
    - rotated_quaternion (np.ndarray): Rotated quaternion (shape: [4], format: [w, x, y, z]).
    """
    # Convert quaternions to rotation matrices
    original_rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    rotated_rotation_matrix = Rotation.from_quat(rotated_quaternion).as_matrix()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original axes
    ax.quiver(0, 0, 0, original_rotation_matrix[0, 0], original_rotation_matrix[1, 0], original_rotation_matrix[2, 0], color='r', label='Original X-axis')
    ax.quiver(0, 0, 0, original_rotation_matrix[0, 1], original_rotation_matrix[1, 1], original_rotation_matrix[2, 1], color='g', label='Original Y-axis')
    ax.quiver(0, 0, 0, original_rotation_matrix[0, 2], original_rotation_matrix[1, 2], original_rotation_matrix[2, 2], color='b', label='Original Z-axis')

    # Plot rotated axes
    ax.quiver(0, 0, 0, rotated_rotation_matrix[0, 0], rotated_rotation_matrix[1, 0], rotated_rotation_matrix[2, 0], color='r', linestyle='dashed', label='Rotated X-axis')
    ax.quiver(0, 0, 0, rotated_rotation_matrix[0, 1], rotated_rotation_matrix[1, 1], rotated_rotation_matrix[2, 1], color='g', linestyle='dashed', label='Rotated Y-axis')
    ax.quiver(0, 0, 0, rotated_rotation_matrix[0, 2], rotated_rotation_matrix[1, 2], rotated_rotation_matrix[2, 2], color='b', linestyle='dashed', label='Rotated Z-axis')

    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show the plot
    plt.show()
if __name__ == "__main__":
    # 原始四元数 (格式: [w, x, y, z])
    quaternion = np.array([0.707, 0.707, 0, 0])  # 初始四元数，表示绕世界 z 轴旋转 45 度
    random_values = np.random.randn(4)

    # Normalize the quaternion to ensure it is a unit quaternion
    quaternion = random_values / np.linalg.norm(random_values)

    # 绕世界坐标系的 z 轴再旋转 90 度
    angle_degrees = 30
    rotated_quaternion = rotate_quaternion_around_world_z_axis(quaternion, angle_degrees)

    print("Original Quaternion:", quaternion)
    print("Rotated Quaternion:", rotated_quaternion)

    # 可视化旋转前和旋转后的轴
    visualize_axes(quaternion, rotated_quaternion)