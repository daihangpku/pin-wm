import xml.etree.ElementTree as ET
import numpy as np
import argparse
import os
import sys
import open3d as o3d
from urdfpy import URDF
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from editing.convertion.ply2obj import blender_ply2obj

def do_abs_urdf(args):
    directory = os.path.dirname(args.input)
    rel2abs = np.load(os.path.join(directory, "rel2abs.npz"))
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(rel2abs['rot'])
    tree = ET.parse(args.input)
    root = tree.getroot()
    
    robot = URDF.load(args.input)
    baselink_name = robot.base_link.name
    for link in root.iter("link"):
        link_name = link.get("name")
        for keyword in ["visual", "collision"]:
            for keywordele in link.iter(keyword):
                for meshele in keywordele.iter('mesh'):
                    if keyword == "visual":
                        link_filename = os.path.join(directory, meshele.get("filename"))
                        link_new_filename = link_filename.replace(".obj", "_abs.obj")
                        part_mesh = o3d.io.read_triangle_mesh(link_filename.replace(".obj", ".ply"))
                        if link_name == baselink_name:
                            part_mesh.transform(rel2abs["rel2abs"])
                        else:
                            rel2abs_R = rel2abs["rel2abs"].copy()
                            rel2abs_R[:3, 3] = 0
                            part_mesh.transform(rel2abs_R)
                        o3d.io.write_triangle_mesh(link_new_filename.replace(".obj", ".ply"), part_mesh)
                        blender_ply2obj(link_new_filename.replace(".obj", ".ply"), link_new_filename)
                        meshele.set("filename", meshele.get("filename").replace(".obj", "_abs.obj"))
                    else:
                        meshele.set("filename", meshele.get("filename").replace(".obj", "_abs.obj"))
    for joint in root.iter("joint"):
        for origin in joint.iter("origin"):
            xyz = np.array([float(x) for x in origin.get('xyz').split(" ")])
            xyz = rel2abs['rel2abs'][:3, :3] @ xyz + rel2abs['rel2abs'][:3, 3]
            origin.set('xyz', ' '.join(str(x) for x in xyz))
        for axis in joint.iter("axis"):
            direction = np.array([float(x) for x in axis.get('xyz').split(" ")])
            direction = rot @ direction
            axis.set('xyz', ' '.join(str(x) for x in direction))
    tree.write(args.input.replace(".urdf", "_abs.urdf"), xml_declaration=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/home/hwfan/workspace/twinmanip/outputs/microwave-20240831_020222_766/object.urdf")
    args = parser.parse_args()
    do_abs_urdf(args)
