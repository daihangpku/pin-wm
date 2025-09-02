import argparse
import open3d as o3d
import os
from sklearn import neighbors
import numpy as np
import torch
import matplotlib
import matplotlib.colors as colors
from plyfile import PlyData, PlyElement
from functools import partial

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from preprocessing.align_meshes_o3d_cpu import align_meshes
from editing.forward_kinematics.urdf_fk import urdf_fk
# from editing.forward_kinematics.mjcf_fk import mjcf_fk
from editing.forward_kinematics.usd_fk import usd_fk
from editing.alignment.split_gs import write_ply
from simulation.fast_gaussian_model_manager import construct_from_ply, matrix_to_quaternion, save_to_ply

def remove_nan_inf(pcd):
    points = np.asarray(pcd.points)
    mask = np.isfinite(points).all(axis=1)  # 过滤掉 NaN 和 Inf
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(points[mask])
    return clean_pcd

def main(args):
    if args.target.endswith(".urdf"):
        fk_func = urdf_fk
    # elif args.target.endswith(".mjcf"):
    #     fk_func = mjcf_fk
    elif args.target.endswith(".usd"):
        fk_func = partial(usd_fk, flange_prim_paths=args.flange_prim_paths)
    else:
        raise NotImplementedError(f"Unknown format {os.path.splitext(os.path.basename(args.target))}")
    fk_dict, all_fk = fk_func(args.target, save=False, clean_vertex_normals=True)
    source_point_cloud = o3d.io.read_point_cloud(args.gaussian_path)
    source_point_cloud = remove_nan_inf(source_point_cloud)
    if args.mesh_path is not None:
        try:
            source_mesh = o3d.io.read_triangle_mesh(args.mesh_path)
            source_mesh_point_cloud = source_mesh.sample_points_uniformly(number_of_points=50000)
        except:
            source_mesh_point_cloud = o3d.io.read_point_cloud(args.mesh_path)
        source_mesh_point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_point_cloud = all_fk.sample_points_uniformly(number_of_points=50000)
    source_point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if args.mesh_path is not None:
        s2t, rmse = align_meshes(source_mesh_point_cloud, target_point_cloud, show_fitting=args.visualize, num_init_trials=50)
    else:
        s2t, rmse = align_meshes(source_point_cloud, target_point_cloud, show_fitting=args.visualize, num_init_trials=50)
    scale = np.ones((3,)) * np.linalg.svd(s2t[:3, :3])[1][0]
    rotation = s2t[:3, :3] / scale
    translation = s2t[:3, 3]
    if args.save_txt:
        s2t_txt_path = os.path.join(os.path.dirname(os.path.abspath(args.mesh_path)), "robot_gs2cad.txt")
        np.savetxt(s2t_txt_path, s2t, fmt="%.8f", delimiter=" ")
        print(f"Saved transformation matrix to {s2t_txt_path}")

    source_point_cloud.transform(s2t)
    knn = neighbors.KNeighborsClassifier(n_neighbors=args.knn_neighbors)
    link_names = []
    X_train = []
    y_train = []
    cnt = 0
    for link_name, part in fk_dict.items():
        each_part_pcd = part.sample_points_uniformly(number_of_points=5000)
        each_part_pcd = np.asarray(each_part_pcd.points)
        X_train.append(each_part_pcd)
        y_train.append(np.ones((each_part_pcd.shape[0],), dtype=int) * cnt)
        link_names.append(link_name)
        cnt += 1
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    knn.fit(X_train, y_train)
    if args.visualize:
        cmap = matplotlib.colormaps['plasma']
        normalized_labels = y_train / np.max(y_train)
        colors = cmap(normalized_labels)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X_train)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
    X_pred = np.asarray(source_point_cloud.points)
    y_pred = knn.predict(X_pred)
    
    gaussian_part_dir = os.path.join(os.path.dirname(os.path.abspath(args.target)), "parts")
    os.makedirs(gaussian_part_dir, exist_ok=True)
    plydata = PlyData.read(args.gaussian_path)
    
    for link_idx, link_name in enumerate(link_names):
        valid_part_gs_idxs = y_pred == link_idx
        new_ply_data = dict()
        for prop in plydata.elements[0].properties:
            new_ply_data[prop.name] = np.asarray(plydata.elements[0][prop.name])[valid_part_gs_idxs]
        part_3dgs_path = os.path.join(gaussian_part_dir, f'{link_name}_3dgs.ply')

        write_ply(part_3dgs_path, len(np.where(valid_part_gs_idxs)[0]), new_ply_data)
        gs_model = construct_from_ply(part_3dgs_path, torch.device(args.device))
        gs_model.scale(torch.from_numpy(scale).to(gs_model._scaling.dtype).to(args.device))
        gs_model.rotate(matrix_to_quaternion(torch.from_numpy(rotation).to(gs_model._rotation.dtype).to(args.device)))
        gs_model.translate(torch.from_numpy(translation).to(gs_model._xyz.dtype).to(args.device))
        save_to_ply(gs_model, part_3dgs_path.replace(".ply", "_abs.ply"))
        
    if args.visualize:
        cmap = matplotlib.colormaps['plasma']
        normalized_labels = y_pred / np.max(y_pred)
        colors = cmap(normalized_labels)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X_pred)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
    
    return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser("prepare franka.urdf, meshes for extra")
    parser.add_argument('-s', '--gaussian-path', type=str, default="", help='source 3dgs path')
    parser.add_argument('-ms', '--mesh-path', type=str, default="", help='source mesh path')
    parser.add_argument('-t', '--target', type=str, default="assets/franka/franka.usd")
    parser.add_argument('-vis', "--visualize", action="store_true")
    parser.add_argument("-k", "--knn-neighbors", type=int, default=3)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument('--save_txt', action='store_true')
    parser.add_argument("--flange-prim-paths", type=str, default=None,
                        help="flange prim paths, to ignore in blender transformation, splitted with ';'")
    args = parser.parse_args()
    main(args)