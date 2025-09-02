import mmcv
import torch
import numpy as np
import random
import os,sys
import json
from tqdm import tqdm
import trimesh
import time
import argparse
import yaml
cur_work_path = os.getcwd()
sys.path.append(cur_work_path)

from builder import Builder
from utils.sys_utils import prepare_output_and_logger
from utils.data_utils import load_pybullet_dynamic_dataset, load_twinaligner_dataset
from utils.metric_utils import mean_psnr
from utils.quaternion_utils import wxyz2xyzw,rotmat_to_quaternion, dq_to_omega_tensor

from diff_rendering.gaussian_splatting_2d.utils.camera_utils import cameraList_from_camInfos
from diff_rendering.gaussian_splatting_2d.utils.render_utils import save_img_u8
from diff_rendering.gaussian_splatting_2d.scene.gaussian_model import GaussianModel
from diff_rendering.gaussian_splatting_2d.utils.loss_utils import rendering_loss_batch,pose_loss_batch

from diff_simulation.simulator import Simulator

from diff_simulation.physical_material import Physical_Materials
from diff_simulation.constraints.base import Joint_Type


def train_physical_materials(output_path,builder,obj_id,ee_id,dataset,simulator:Simulator,
                             gaussians:GaussianModel,gaussian_renderer,random_init_index,
                             optimizer,logger=None):
    n_substeps = round(builder.sim_args.frame_dt / builder.sim_args.dtime)
    f_max = len(dataset["frame_data"])
    dynamic_train_camera = cameraList_from_camInfos(dataset["cam_infos"], 1.0, builder.gaussian_args.dataset)
    gt_images = dataset["rgb_images"]

    timestamps = [frame["ros_timestamp"] for frame in dataset["frame_data"]]
    ee_trans = [torch.tensor(frame["ee_trans"], device=simulator.device) for frame in dataset["frame_data"]]
    ee_quat = [torch.tensor(frame["ee_quat_wxyz"], device=simulator.device) for frame in dataset["frame_data"]]

    frame_data = dataset["frame_data"]
    
    train_iteration = builder.sim_args.train_iteration


    progress_bar = tqdm(range(0, train_iteration), desc="Dynamic Train Progress")
    for iter in range(0,train_iteration + 1):
        simulator.reset()
        images = []
        poses = []
        for f in range(f_max-1):
            print(f"Processing frame {f}")
            # import ipdb
            # ipdb.set_trace()
            if f >= 0:
                ee_world_position,ee_world_rotation = simulator.get_body_pose_clone(ee_id)
                dt = timestamps[f+1] - timestamps[f]
                ee_linear_velocity = (ee_trans[f+1] - ee_world_position)/dt
                ee_angular_velocity = dq_to_omega_tensor(ee_world_rotation, ee_quat[f+1], dt, device=simulator.device)

                # ee_linear_velocity should remain 3D, don't add extra dimension
                simulator.change_body_vel(ee_id,linear_velocity=ee_linear_velocity,angular_velocity=torch.zeros(3,device=simulator.device))
                for i in range(int(dt/simulator.dtime)):

                    simulator.step()

            obj_position,obj_rotation = simulator.get_body_pose(obj_id)
            gaussians.reset_position_rotation(obj_position,obj_rotation)

            render_pkg = gaussian_renderer.render(dynamic_train_camera[0], gaussians)


            image = render_pkg['render'] 
            images.append(image)
            poses.append(torch.cat((obj_position.clone(),wxyz2xyzw(obj_rotation.clone())),dim=-1))

        loss = rendering_loss_batch(images[:],gt_images[:])

        loss.backward(retain_graph=True)
        
        psnr = mean_psnr(images,gt_images)

        all_physical_materials = simulator.get_all_physical_materials()
        if logger is not None:
            logger.record("metric/loss"+ str(random_init_index),loss)
            logger.record("metric/psnr"+ str(random_init_index),psnr)
            logger.dump(iter)


        info_dict = {"loss": loss.item(),"psnr": psnr.item()}
        progress_bar.set_postfix(info_dict)
        progress_bar.update(1)

        if (iter % builder.sim_args.save_iteration) == 0:
            save_path = os.path.join(output_path, "iteration_{}".format(iter))
            os.makedirs(save_path, exist_ok = True)
            image_path = os.path.join(save_path, "images")
            os.makedirs(image_path, exist_ok = True)
            for index,image in enumerate(images):
                save_img_u8(image.permute(1,2,0).cpu().detach().numpy(), os.path.join(image_path, 'sim_{}.png'.format(index)))
            physical_materials_original_json_list = []
            physical_materials_activate_json_list = []
            for physical_materials in all_physical_materials:
                physical_materials_original_json_list.append(physical_materials.get_original_json_dict())
                physical_materials_activate_json_list.append(physical_materials.get_activate_json_dict())
            with open(os.path.join(save_path,'physical_materials_iter.json'), 'w') as json_file:
                json.dump({
                           "activate":physical_materials_activate_json_list}, fp=json_file ,indent=4)

        optimizer.step()
        optimizer.zero_grad()
    progress_bar.close()

    return loss

def create_push_scene(builder,simulator:Simulator, obj_trans, obj_quat, ee_trans, ee_quat):

    plane_mesh_path = "./envs/asset/plane/plane_collision.obj"
    obj_mesh_path = "envs/asset/banana/mesh_w_vertex_color_abs.obj"
    ee_mesh_path = "./envs/asset/ee/ee.obj"
    plane_urdf_path = "./envs/asset/plane/plane.urdf"
    obj_urdf_path = "envs/asset/banana/object.urdf"
    ee_urdf_path = "./envs/asset/franka_ee/ee.urdf"

    plane_mesh = trimesh.load(plane_mesh_path)
    plane_mesh.apply_scale([30,30,10])
    plane_physical_material = Physical_Materials(requires_grad = True,device = simulator.device)
    plane_physical_material.no_optimize("mass")
    plane_physical_material.no_optimize('inertia')
    plane_id = simulator.create_mesh_body(plane_mesh,plane_physical_material,requires_grad=True,
                                            urdf = plane_urdf_path,
                                            world_position=torch.tensor([0.0,0.0,-5.011],device=simulator.device),
                                            world_rotation=torch.tensor([1.0,0.0,0.0,0.0],device=simulator.device))

    simulator.create_joint(plane_id,Joint_Type.FIX_CONSTRAINT)
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_physical_material = Physical_Materials(requires_grad = True,device=simulator.device)
    obj_physical_material.set_material("inertia",[
                [
                    0.01,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.01,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.01
                ]
            ])   
    obj_id = simulator.create_mesh_body(obj_mesh,obj_physical_material,requires_grad=True,
                                        urdf = obj_urdf_path,
                                        world_position=obj_trans,
                                        world_rotation=obj_quat) 

    ee_mesh = trimesh.load(ee_mesh_path)
    ee_physical_material = Physical_Materials(requires_grad = True,device = simulator.device)
    ee_physical_material.set_material("mass",1000.0)
    ee_physical_material.no_optimize("inertia")
    ee_physical_material.no_optimize("mass")
    ee_id = simulator.create_mesh_body(ee_mesh,ee_physical_material,requires_grad=True,
                                        urdf = ee_urdf_path,
                                        world_position=ee_trans,
                                        world_rotation=ee_quat)
    simulator.create_joint(ee_id,Joint_Type.NO_TRANS_Z_CONSTRATNT)
    #simulator.create_joint(ee_id,Joint_Type.NO_ROT_CONSTRATNT)

    return obj_id,ee_id



def test_dynamic_train(all_args):
    builder = Builder(all_args)

    dynamic_train_datasets,_ = load_twinaligner_dataset(builder.data_args)
    all_args, loggers = prepare_output_and_logger(all_args, dynamic_train_datasets, need_logger=True)
    

    train_path = os.path.join(builder.sys_args.output_path, "dynamic","train",str(int(time.time())))
    os.makedirs(train_path, exist_ok = True)    

    random_init_num = 1
    for random_init_index in range(random_init_num):
        train_sub_path = os.path.join(train_path, "random_init" + str(random_init_index))
        os.makedirs(train_sub_path, exist_ok = True)
        loss_mean = 0.0
        for dataset_index,dynamic_train_dataset in enumerate(dynamic_train_datasets):
            output_path = os.path.join(train_sub_path, "dataset" + str(dataset_index))
            os.makedirs(output_path, exist_ok = True)
            print("Save folder:",train_path)
            sim_device = "cuda"
            vis = True
            simulator = Simulator(builder.sim_args.dtime,device=sim_device,vis=vis)
            transformation_matrix = dynamic_train_dataset["pose_datas"][0]
            obj_quat = rotmat_to_quaternion(torch.tensor(transformation_matrix[:3,:3],device=simulator.device, dtype=torch.float32))
            obj_trans = torch.tensor(transformation_matrix[:3,3],device=simulator.device, dtype=torch.float32)
            ee_quat = torch.tensor(dynamic_train_dataset["frame_data"][0]["ee_quat_wxyz"],device=simulator.device, dtype=torch.float32)
            ee_trans = torch.tensor(dynamic_train_dataset["frame_data"][0]["ee_trans"],device=simulator.device, dtype=torch.float32)
            obj_id, ee_id = create_push_scene(builder, simulator, obj_trans, obj_quat, ee_trans, ee_quat)
            obj_position,obj_rotation = simulator.get_body_pose_clone(obj_id)
            gaussians = builder.build_static_2dgs()
            gaussians.translate2localframe(obj_position,obj_rotation)
            gaussian_renderer = builder.build_renderer()
            optimizer = builder.build_optimizer(simulator)
            optimizer.param_groups[4]['lr'] = 4e-2
            optimizer.param_groups[5]['lr'] = 1e-4
            loss = train_physical_materials(output_path,builder,obj_id,ee_id,dynamic_train_dataset,
                                    simulator,gaussians,gaussian_renderer,random_init_index,
                                    optimizer,loggers["dynamic"][dataset_index])
            simulator.close()
            loss_mean += loss / len(dynamic_train_datasets)

        print("loss_mean",loss_mean)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/banana.yaml')
    args = parser.parse_args()
    all_args = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    test_dynamic_train(all_args)





