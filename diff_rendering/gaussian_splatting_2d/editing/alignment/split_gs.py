from urdfpy import URDF
import os
import trimesh
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
import torch
import typing
from sklearn import neighbors

def write_ply(
    filename: str,
    count: int,
    map_to_tensors: typing.OrderedDict[str, np.ndarray],
):
    """
    Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
    Note: All float values will be converted to float32 for writing.

    Parameters:
    filename (str): The name of the file to write.
    count (int): The number of vertices to write.
    map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
        Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
    """

    # Ensure count matches the length of all tensors
    if not all(len(tensor) == count for tensor in map_to_tensors.values()):
        raise ValueError("Count does not match the length of all tensors")

    # Type check for numpy arrays of type float or uint8 and non-empty
    if not all(
        isinstance(tensor, np.ndarray)
        and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
        and tensor.size > 0
        for tensor in map_to_tensors.values()
    ):
        raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

    with open(filename, "wb") as ply_file:
        # Write PLY header
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")

        ply_file.write(f"element vertex {count}\n".encode())

        # Write properties, in order due to OrderedDict
        for key, tensor in map_to_tensors.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            ply_file.write(f"property {data_type} {key}\n".encode())

        ply_file.write(b"end_header\n")

        # Write binary data
        # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
        for i in range(count):
            for tensor in map_to_tensors.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    ply_file.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    ply_file.write(value.tobytes())
    
def split_gs(input_urdf_path, device="cuda:0"):
    urdf_robot = URDF.load(str(input_urdf_path))
    sdfstudio_path = os.path.dirname(input_urdf_path)    

    plydata = PlyData.read(os.path.join(sdfstudio_path, "object_3dgs.ply"))
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"]),
    ), axis=1)
    X_train = []
    y_train = []
    
    # here we use knn algorithm like editing/alignment/align_robot_gs_knn.py
    knn = neighbors.KNeighborsClassifier(n_neighbors=args.knn_neighbors)
    for link_idx, link in enumerate(urdf_robot.links):
        link_name = link.name
        forward_kinematics = urdf_robot.link_fk()[link]
        part_mesh = trimesh.load(os.path.join(sdfstudio_path, "parts", link_name + ".ply"))
        o3d_part_mesh = part_mesh.as_open3d
        o3d_part_mesh.transform(forward_kinematics)
        part_pcd = o3d_part_mesh.sample_points_uniformly(number_of_points=5000)
        part_pcd = np.asarray(part_pcd.points)
        X_train.append(part_pcd)
        y_train.append(np.ones((part_pcd.shape[0],), dtype=int) * link_idx)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(xyz)
    for link_idx, link in enumerate(urdf_robot.links):
        link_name = link.name
        valid_part_gs_idxs = y_pred == link_idx
        new_ply_data = dict()
        for prop in plydata.elements[0].properties:
            new_ply_data[prop.name] = np.asarray(plydata.elements[0][prop.name])[valid_part_gs_idxs]
        write_ply(os.path.join(sdfstudio_path, "parts", f'{link_name}_3dgs.ply'), len(np.where(valid_part_gs_idxs)[0]), new_ply_data)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split 3DGS for articulated object')
    parser.add_argument('--input', type=str, default="/home/hwfan/workspace/twinmanip/outputs/microwave-20240831_020222_766/object.urdf",
                        help='Input URDF File')
    parser.add_argument("-k", "--knn-neighbors", type=int, default=20)
    args = parser.parse_args()
    split_gs(args.input)
    