import open3d as o3d
import numpy as np
import argparse
from PyQt5.QtWidgets import QApplication, QDialog, QFormLayout, QLabel, QLineEdit, QPushButton
import os
import copy

class InputDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Absolute Scale")

        layout = QFormLayout()

        self.x_label = QLabel("X:")
        self.x_edit = QLineEdit()
        self.y_label = QLabel("Y:")
        self.y_edit = QLineEdit()
        self.z_label = QLabel("Z:")
        self.z_edit = QLineEdit()

        layout.addRow(self.x_label, self.x_edit)
        layout.addRow(self.y_label, self.y_edit)
        layout.addRow(self.z_label, self.z_edit)

        button = QPushButton("OK")
        button.clicked.connect(self.accept)
        layout.addWidget(button)

        self.setLayout(layout)

    def get_input(self):
        return float(self.x_edit.text()), float(self.y_edit.text()), float(self.z_edit.text())
    
def main(args, save=False):
    meshfile = args.input
    mesh = o3d.io.read_triangle_mesh(meshfile)
    ori_mesh = copy.deepcopy(mesh)
    ori_center = mesh.get_center()
    T_transmat_all = np.eye(4)
    T_transmat_all[:3, 3] = -ori_center
    mesh.translate(-ori_center)
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(axis_pcd)
    
    R_all = np.eye(3)
    r_axis_angles = np.zeros((3,))
    scale_x = None
    scale_y = None
    scale_z = None
    
    def rotate_mesh_YZ_pos(vis):
        nonlocal mesh
        nonlocal R_all
        delta = 0.5 * np.pi / 180
        r_axis_angles[0] += delta
        R_ori = R_all.copy()
        R_all = o3d.geometry.get_rotation_matrix_from_axis_angle(r_axis_angles)
        mesh.rotate(R_all @ np.linalg.inv(R_ori), center=[0, 0, 0])
        vis.update_geometry(mesh)

    def rotate_mesh_YZ_neg(vis):
        nonlocal mesh
        nonlocal R_all
        delta = 0.5 * np.pi / 180
        r_axis_angles[0] -= delta
        R_ori = R_all.copy()
        R_all = o3d.geometry.get_rotation_matrix_from_axis_angle(r_axis_angles)
        mesh.rotate(R_all @ np.linalg.inv(R_ori), center=[0, 0, 0])
        vis.update_geometry(mesh)

    def rotate_mesh_XZ_pos(vis):
        nonlocal mesh
        nonlocal R_all
        delta = 0.5 * np.pi / 180
        r_axis_angles[1] += delta
        R_ori = R_all.copy()
        R_all = o3d.geometry.get_rotation_matrix_from_axis_angle(r_axis_angles)
        mesh.rotate(R_all @ np.linalg.inv(R_ori), center=[0, 0, 0])
        vis.update_geometry(mesh)

    def rotate_mesh_XZ_neg(vis):
        nonlocal mesh
        nonlocal R_all
        delta = 0.5 * np.pi / 180
        r_axis_angles[1] -= delta
        R_ori = R_all.copy()
        R_all = o3d.geometry.get_rotation_matrix_from_axis_angle(r_axis_angles)
        mesh.rotate(R_all @ np.linalg.inv(R_ori), center=[0, 0, 0])
        vis.update_geometry(mesh)

    def rotate_mesh_XY_pos(vis):
        nonlocal mesh
        nonlocal R_all
        delta = 0.5 * np.pi / 180
        r_axis_angles[2] += delta
        R_ori = R_all.copy()
        R_all = o3d.geometry.get_rotation_matrix_from_axis_angle(r_axis_angles)
        mesh.rotate(R_all @ np.linalg.inv(R_ori), center=[0, 0, 0])
        vis.update_geometry(mesh)

    def rotate_mesh_XY_neg(vis):
        nonlocal mesh
        nonlocal R_all
        delta = 0.5 * np.pi / 180
        r_axis_angles[2] -= delta
        R_ori = R_all.copy()
        R_all = o3d.geometry.get_rotation_matrix_from_axis_angle(r_axis_angles)
        mesh.rotate(R_all @ np.linalg.inv(R_ori), center=[0, 0, 0])
        vis.update_geometry(mesh)
    
    def XY_view(vis):
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_front([0, 0, -1])

    def XZ_view(vis):
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 0, 1])
        view_control.set_front([0, 1, 0])
        
    def YZ_view(vis):
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_front([1, 0, 0])

    def annot_scale(vis):        
        nonlocal scale_x
        nonlocal scale_y
        nonlocal scale_z
        app = QApplication([])
        dialog = InputDialog()
        if dialog.exec_() == QDialog.Accepted:
            scale_x, scale_y, scale_z = dialog.get_input()

    XY_view(vis)
    vis.register_key_callback(ord('1'), XY_view)
    vis.register_key_callback(ord('2'), rotate_mesh_XY_pos)
    vis.register_key_callback(ord('3'), rotate_mesh_XY_neg)
    vis.register_key_callback(ord('4'), XZ_view)
    vis.register_key_callback(ord('5'), rotate_mesh_XZ_pos)
    vis.register_key_callback(ord('6'), rotate_mesh_XZ_neg)
    vis.register_key_callback(ord('7'), YZ_view)
    vis.register_key_callback(ord('8'), rotate_mesh_YZ_pos)
    vis.register_key_callback(ord('9'), rotate_mesh_YZ_neg)
    vis.register_key_callback(ord('0'), annot_scale)
    vis.run()
    vis.destroy_window()
    vertices = np.asarray(mesh.vertices)
    
    R_transmat_all = np.eye(4)
    R_transmat_all[:3, :3] = R_all
    
    normalized_x_scale = vertices[:, 0].max() - vertices[:, 0].min()
    normalized_y_scale = vertices[:, 1].max() - vertices[:, 1].min()
    normalized_z_scale = vertices[:, 2].max() - vertices[:, 2].min()
    x_ratio = scale_x / normalized_x_scale
    y_ratio = scale_y / normalized_y_scale
    z_ratio = scale_z / normalized_z_scale
    max_ratio = (x_ratio + y_ratio + z_ratio) / 3
    S_transmat_all = np.eye(4)
    S_transmat_all[0, 0] = max_ratio
    S_transmat_all[1, 1] = max_ratio
    S_transmat_all[2, 2] = max_ratio
    
    transmat_all = S_transmat_all @ R_transmat_all @ T_transmat_all
    ori_mesh.transform(transmat_all)
    output_dir = os.path.dirname(meshfile)
    if save:
        np.savez_compressed(os.path.join(output_dir, 'rel2abs.npz'), 
                            rel2abs=transmat_all,
                            scale=np.array([max_ratio, max_ratio, max_ratio]),
                            rot=r_axis_angles,
                            trans=-ori_center,)
        o3d.io.write_triangle_mesh(meshfile.replace(".ply", "_abs.ply"), ori_mesh)
    return transmat_all
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/hwfan/workspace/twinmanip/outputs/component-20240914_175058_247/mesh_w_vertex_color.ply")
    args = parser.parse_args()
    main(args, save=True)
    