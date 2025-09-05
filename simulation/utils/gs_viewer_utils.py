'''
Modified from https://github.com/yzslab/gaussian-splatting-lightning
'''
from gsplat import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
import time
import threading
import traceback
import numpy as np
import torch
import viser
import viser.transforms as vtf
import math
import sys
import os
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
sys.path.insert(0, os.getcwd())
from simulation.fast_gaussian_model_manager import GaussianModel
from simulation.utils.sh_utils import eval_sh
from typing import Any, Union, List, Tuple, Optional
from dataclasses import dataclass, field
from torch import Tensor

def colmap_to_pyrender(extrinsic):
    """
    将COLMAP的相机参数转换为pyrender所需的参数
    
    参数:
        extrinsic: COLMAP外参矩阵 (3x4) World --> Camera
    输出:
        camera_pose: Camera --> World
    """
    # 提取旋转和平移
    R_colmap = extrinsic[:3, :3]
    t_colmap = extrinsic[:3, 3]
    
    transform_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    # 构建4x4变换矩阵 (世界到相机)
    extrinsic_4x4 = np.eye(4)
    extrinsic_4x4[:3, :3] = R_colmap
    extrinsic_4x4[:3, 3] = t_colmap
    
    # 应用坐标系转换
    camera_pose = transform_matrix @ extrinsic_4x4
    
    # 转换为相机到世界的变换
    camera_pose = np.linalg.inv(camera_pose)
    
    return camera_pose

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def load_camera_params():
    extr_path = os.path.join("assets", "realsense", "cam_extr.txt")
    camera_extr = np.loadtxt(extr_path)
    
    intr_path = os.path.join("assets", "realsense", "cam_K.txt")
    camera_intr = np.loadtxt(intr_path)
    camera_intr_dict = dict()
    camera_intr_dict["fx"] = camera_intr[0, 0]
    camera_intr_dict["fy"] = camera_intr[1, 1]
    camera_intr_dict["cx"] = camera_intr[0, 2]
    camera_intr_dict["cy"] = camera_intr[1, 2]
    camera_intr_dict["image_width"] = 1280
    camera_intr_dict["image_height"] = 720
    return camera_extr, camera_intr_dict

class CameraType:
    PERSPECTIVE: int = 0
    FISHEYE: int = 1

@dataclass
class Camera:
    R: Tensor  # [3, 3]
    T: Tensor  # [3]
    fx: Tensor
    fy: Tensor
    fov_x: Tensor
    fov_y: Tensor
    cx: Tensor
    cy: Tensor
    width: Tensor
    height: Tensor
    appearance_id: Tensor
    normalized_appearance_id: Tensor
    time: Tensor
    distortion_params: Optional[Tensor]  # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    camera_type: Tensor

    world_to_camera: Tensor
    projection: Tensor
    full_projection: Tensor
    camera_center: Tensor

    def to_device(self, device):
        for field in Camera.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))

        return self

@dataclass
class Cameras:
    """
    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor = field(init=False)  # [n_cameras]
    fov_y: Tensor = field(init=False)  # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]
    appearance_id: Tensor  # [n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [n_cameras]
    distortion_params: Optional[Union[Tensor, list[Tensor]]]  # [n_cameras, 2 or 4 or 5 or 8 or 12 or 14], (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    camera_type: Tensor  # Int[n_cameras]

    world_to_camera: Tensor = field(init=False)  # [n_cameras, 4, 4], transposed
    projection: Tensor = field(init=False)
    full_projection: Tensor = field(init=False)
    camera_center: Tensor = field(init=False)

    time: Optional[Tensor] = None  # [n_cameras]

    def _calculate_fov(self):
        # calculate fov
        self.fov_x = 2 * torch.atan((self.width / 2) / self.fx)
        self.fov_y = 2 * torch.atan((self.height / 2) / self.fy)

    def _calculate_w2c(self):
        # build world-to-camera transform matrix
        self.world_to_camera = torch.zeros((self.R.shape[0], 4, 4))
        self.world_to_camera[:, :3, :3] = self.R
        self.world_to_camera[:, :3, 3] = self.T
        self.world_to_camera[:, 3, 3] = 1.
        self.world_to_camera = torch.transpose(self.world_to_camera, 1, 2)

    def _calculate_ndc_projection_matrix(self):
        """
        calculate ndc projection matrix
        http://www.songho.ca/opengl/gl_projectionmatrix.html

        TODO:
            1. support colmap refined principal points
            2. the near and far here are ignored in diff-gaussian-rasterization
        """
        zfar = 100.0
        znear = 0.01

        tanHalfFovY = torch.tan((self.fov_y / 2))
        tanHalfFovX = torch.tan((self.fov_x / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(self.fov_y.shape[0], 4, 4)

        z_sign = 1.0

        P[:, 0, 0] = 2.0 * znear / (right - left)  # = 1 / tanHalfFovX = 2 * fx / width
        P[:, 1, 1] = 2.0 * znear / (top - bottom)  # = 2 * fy / height
        P[:, 0, 2] = (right + left) / (right - left)  # = 0, right + left = 0
        P[:, 1, 2] = (top + bottom) / (top - bottom)  # = 0, top + bottom = 0
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        self.projection = torch.transpose(P, 1, 2)

        self.full_projection = self.world_to_camera.bmm(self.projection)

    def _calculate_camera_center(self):
        self.camera_center = torch.linalg.inv(self.world_to_camera)[:, 3, :3]

    def __post_init__(self):
        self._calculate_fov()
        self._calculate_w2c()
        self._calculate_ndc_projection_matrix()
        self._calculate_camera_center()

        if self.time is None:
            self.time = torch.zeros(self.R.shape[0])
        if self.distortion_params is None:
            self.distortion_params = torch.zeros(self.R.shape[0], 4)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Camera:
        return Camera(
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            fov_x=self.fov_x[index],
            fov_y=self.fov_y[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=self.normalized_appearance_id[index],
            distortion_params=self.distortion_params[index],
            time=self.time[index],
            camera_type=self.camera_type[index],
            world_to_camera=self.world_to_camera[index],
            projection=self.projection[index],
            full_projection=self.full_projection[index],
            camera_center=self.camera_center[index],
        )

class VanillaRenderer:
    def __init__(self, compute_cov3D_python: bool = False, convert_SHs_python: bool = False):
        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

    def render(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python is True:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python is True:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        # print("radii device:", radii.device)
        # print("radii shape:", radii.shape)
        # print("radii contains NaN:", torch.isnan(radii).any())

        # radii = radii.to('cuda:0')
        # rendered_image = rendered_image.to('cuda:0')
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    
DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = False
INVALID_DEPTH: float = 10.0
class GSPlatRenderer():
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS) -> None:

        self.block_size = block_size
        self.anti_aliased = anti_aliased

    def render(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        if render_types is None:
            render_types = ["rgb"]

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means3d=pc.get_xyz,
            scales=pc.get_scaling,
            glob_scale=scaling_modifier,
            quats=pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True),
            viewmat=viewpoint_camera.world_to_camera.T[:3, :],
            # projmat=viewpoint_camera.full_projection.T,
            fx=viewpoint_camera.fx.item(),
            fy=viewpoint_camera.fy.item(),
            cx=viewpoint_camera.cx.item(),
            cy=viewpoint_camera.cy.item(),
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
        )

        try:
            xys.retain_grad()
        except:
            pass
        #import pdb; pdb.set_trace()
        viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
        # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        rgb = None
        alpha = None
        if "rgb" in render_types:
            rgb, alpha = rasterize_gaussians(  # type: ignore
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,  # type: ignore
                rgbs,
                opacities,
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=bg_color,
                return_alpha=True,
            )  # type: ignore
            rgb = rgb.permute(2, 0, 1)
            alpha = alpha

            # 合并 RGB 和 Alpha 通道为 RGBA 格式
            rgba = torch.cat([rgb, alpha.unsqueeze(0)], dim=0) 
        depth_im = None
        if "depth" in render_types:
            depth_im, depth_alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths.unsqueeze(-1).repeat(1, 3),
                opacities,
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=torch.zeros_like(bg_color),
                return_alpha=True,
            )  # type: ignore
            depth_alpha = depth_alpha[..., None]
            depth_im = torch.where(depth_alpha > 0, depth_im / depth_alpha, INVALID_DEPTH)
            depth_im = depth_im.permute(2, 0, 1)

        return {
            "render": rgba,
            "depth": depth_im,
            "alpha": alpha,
            "viewspace_points": xys,
            "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

class ClientThread(threading.Thread):
    def __init__(self, viewer, renderer, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.client = client
        self.render_trigger = threading.Event()
        self.last_move_time = 0
        self.last_camera = None  # store camera information
        self.state = "low"  # low or high render resolution
        self.stop_client = False  # whether stop this thread
        self.client.camera.up_direction = np.asarray([0., 0., 1.])
        @self.client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()

    def render_and_send(self):
        with self.client.atomic():
            cam = self.last_camera

            self.last_move_time = time.time()

            # get camera pose
            R = vtf.SO3(wxyz=self.client.camera.wxyz)
            R = R @ vtf.SO3.from_x_radians(np.pi)
            R = torch.tensor(R.as_matrix())
            pos = torch.tensor(self.client.camera.position, dtype=torch.float64)
            c2w = torch.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = pos

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            # calculate resolution
            aspect_ratio = cam.aspect
            max_res, jpeg_quality = self.get_render_options()
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)

            # construct camera
            fx = torch.tensor([fov2focal(cam.fov, max_res)], dtype=torch.float)
            camera = Cameras(
                R=R.unsqueeze(0),
                T=T.unsqueeze(0),
                fx=fx,
                fy=fx,
                cx=torch.tensor([(image_width // 2)], dtype=torch.int),
                cy=torch.tensor([(image_height // 2)], dtype=torch.int),
                width=torch.tensor([image_width], dtype=torch.int),
                height=torch.tensor([image_height], dtype=torch.int),
                appearance_id=torch.tensor([0], dtype=torch.int),
                normalized_appearance_id=torch.tensor([0], dtype=torch.float),
                time=torch.tensor([0], dtype=torch.float),
                distortion_params=None,
                camera_type=torch.tensor([0], dtype=torch.int),
            )[0].to_device(self.viewer.device)

            with torch.no_grad():
                image = self.renderer.get_outputs(camera, scaling_modifier=1.0)["render"]
                image = torch.clamp(image, max=1.)
                image = torch.permute(image, (1, 2, 0))
                self.client.set_background_image(
                    image.cpu().numpy(),
                    format=self.viewer.image_format,
                    jpeg_quality=jpeg_quality,
                )

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)  # TODO: avoid wasting CPU
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:
                # skip if camera is none
                if self.last_camera is None:
                    continue

                # if we haven't received a trigger in a while, switch to high resolution
                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                else:
                    continue  # skip if already in high resolution mode

            self.render_trigger.clear()
            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                break

        self._destroy()
        
    def stop(self):
        self.stop_client = True
        # self.render_trigger.set()  # TODO: potential thread leakage?
        
    def get_render_options(self):
        return 1920, 100

    def _destroy(self):
        self.viewer = None
        self.renderer = None
        self.client = None
        self.last_camera = None

class ViewerRenderer:
    def __init__(
            self,
            gaussian_model,
            renderer,
            background_color,
    ):
        super().__init__()

        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.background_color = background_color

    def get_outputs(self, camera, scaling_modifier: float = 1., render_depth=False):
        return self.renderer.render(
            camera,
            self.gaussian_model,
            self.background_color,
            scaling_modifier=scaling_modifier,
            render_types=["rgb", "depth"] if render_depth else ["rgb"],
        )
