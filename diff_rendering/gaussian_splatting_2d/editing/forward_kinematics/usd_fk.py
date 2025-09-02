import numpy as np
import argparse
import os
import bpy
import subprocess
from collections import defaultdict
import trimesh
import copy
import open3d as o3d
def trimesh_to_open3d(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    return o3d_mesh

def get_object_path(obj):
    path = []
    parent = obj.parent
    while parent:
        path.append(parent.name)
        parent = parent.parent
    path.reverse()
    path.append(obj.name)
    return '/' + '/'.join(path)

def select_only(obj):
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)

def usd_fk(usd_path, save=True, clean_vertex_normals=True, flange_prim_paths="/panda/panda_link8"):#"/A02L_MP/Robotiq_2F_85"): 
    # NOTE: bpy cannot be imported in isaac sim due to version problem of LibUSD
    # we use subprocess calling instead
    mesh_dir = os.path.join(os.path.dirname(usd_path), "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    
    subprocess.call(['python', os.path.join(os.path.dirname(__file__), "usd_fk_transforms.py"), '--input', usd_path])   
    link_transforms_npz = np.load(os.path.join(os.path.dirname(usd_path), "link_transforms.npz"))
    link_paths = link_transforms_npz["link_paths"].tolist()
    link_transforms = link_transforms_npz["link_transforms"]
    link_names = link_transforms_npz["link_names"].tolist()
    for idx in range(len(link_paths)):
        if "link0" in link_paths[idx] or "rightfinger" in link_paths[idx]:
            link_paths[idx] = link_paths[idx] + '.001'
        if "link8" in link_paths[idx]:
            link_paths[idx] = "/panda/panda_hand_joint"
    bpy.ops.wm.read_homefile()
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        bpy.data.meshes.remove(mesh)
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
        bpy.data.objects.remove(camera)
    if "Light" in bpy.data.objects:
        light = bpy.data.objects["Light"]
        bpy.data.objects.remove(light)
    bpy.ops.wm.usd_import(filepath=usd_path)
    link_to_obj = defaultdict(list)
    for obj in bpy.data.objects:
        obj_path = get_object_path(obj)
        for link_path in link_paths:
            if obj_path == link_path:
                obj.rotation_mode = 'QUATERNION'
                link_transform = link_transforms[link_paths.index(obj_path)]
                link_name = link_names[link_paths.index(obj_path)]
                obj.location = link_transform[:3]
                obj.rotation_quaternion = link_transform[[6,3,4,5]]
                break
        if obj.type == "MESH":
            for link_name in link_names:
                if link_name in obj_path.split("/"):
                    break
            link_to_obj[link_name].append(obj.name)
    # bpy.ops.wm.save_as_mainfile(filepath="debug.blend")
    bpy.ops.wm.obj_export(filepath=usd_path.replace(".usd", ".obj"), forward_axis='Y', up_axis='Z', )
    bpy.ops.wm.read_homefile()
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        bpy.data.meshes.remove(mesh)
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
        bpy.data.objects.remove(camera)
    if "Light" in bpy.data.objects:
        light = bpy.data.objects["Light"]
        bpy.data.objects.remove(light)
    bpy.ops.wm.obj_import(filepath=usd_path.replace(".usd", ".obj"), forward_axis='Y', up_axis='Z', )
    for link_name, obj_names in link_to_obj.items():
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.name in obj_names]
        active_object = mesh_objects[0]
        for obj in mesh_objects[1:]:
            bpy.ops.object.select_all(action='DESELECT')
            active_object.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = active_object
            bpy.ops.object.join()
        active_object.name = link_name
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            select_only(obj)
            bpy.ops.wm.obj_export(filepath=os.path.join(mesh_dir, f"{obj.name}.obj"), forward_axis='Y', up_axis='Z', 
                                  export_selected_objects=True, export_normals=False, export_uv=False, export_materials=False)

    fk_dict = dict()
    combined = None
    for link_name in link_to_obj.keys():
        filepath = os.path.join(mesh_dir, f"{link_name}.obj")
        mesh = trimesh.load(filepath)
        o3d_mesh = trimesh_to_open3d(mesh)
        fk_dict[link_name] = copy.deepcopy(o3d_mesh)
        if combined is None:
            combined = copy.deepcopy(o3d_mesh)
        else:
            combined += o3d_mesh
    return fk_dict, combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="assets/franka/franka.usd",
                        help='Input USD File')
    parser.add_argument("--flange-prim-paths", type=str, default=None,
                        help="flange prim paths, to ignore in blender transformation, splitted with ';'")
    args = parser.parse_args()
    usd_fk(args.input, flange_prim_paths=args.flange_prim_paths)
