# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
'''
Credit: https://github.com/nexuslrf/SAM-3D-Selector/blob/main/app.py
Credit: https://github.com/isl-org/Open3D/blob/main/examples/python/visualization/vis_gui.py
'''

import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import cv2
import argparse
import copy
import seaborn
from collections import defaultdict
import shutil
import pyautogui
import time

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.getcwd())
from editing.convertion.ply2obj import blender_ply2obj

isMacOS = (platform.system() == "Darwin")

'''
Credit: https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
'''

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow_from_ori_and_dir(origin, vec):
    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005,
                                            cone_radius=0.01,
                                            cylinder_height=0.1,
                                            cone_height=0.06,
                                            resolution=10,
                                            cylinder_split=4,
                                            cone_split=1)
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh

def RotateAnyAxis(v1, v2, step):
    ROT = np.identity(4)

    axis = v2 - v1
    axis = axis / np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)

    step_cos = np.cos(step)
    step_sin = np.sin(step)

    ROT[0][0] = axis[0] * axis[0] + (axis[1] * axis[1] + axis[2] * axis[2]) * step_cos
    ROT[0][1] = axis[0] * axis[1] * (1 - step_cos) + axis[2] * step_sin
    ROT[0][2] = axis[0] * axis[2] * (1 - step_cos) - axis[1] * step_sin
    ROT[0][3] = 0

    ROT[1][0] = axis[1] * axis[0] * (1 - step_cos) - axis[2] * step_sin
    ROT[1][1] = axis[1] * axis[1] + (axis[0] * axis[0] + axis[2] * axis[2]) * step_cos
    ROT[1][2] = axis[1] * axis[2] * (1 - step_cos) + axis[0] * step_sin
    ROT[1][3] = 0

    ROT[2][0] = axis[2] * axis[0] * (1 - step_cos) + axis[1] * step_sin
    ROT[2][1] = axis[2] * axis[1] * (1 - step_cos) - axis[0] * step_sin
    ROT[2][2] = axis[2] * axis[2] + (axis[0] * axis[0] + axis[1] * axis[1]) * step_cos
    ROT[2][3] = 0

    ROT[3][0] = (v1[0] * (axis[1] * axis[1] + axis[2] * axis[2]) - axis[0] * (v1[1] * axis[1] + v1[2] * axis[2])) * (1 - step_cos) + \
                (v1[1] * axis[2] - v1[2] * axis[1]) * step_sin

    ROT[3][1] = (v1[1] * (axis[0] * axis[0] + axis[2] * axis[2]) - axis[1] * (v1[0] * axis[0] + v1[2] * axis[2])) * (1 - step_cos) + \
                (v1[2] * axis[0] - v1[0] * axis[2]) * step_sin

    ROT[3][2] = (v1[2] * (axis[0] * axis[0] + axis[1] * axis[1]) - axis[2] * (v1[0] * axis[0] + v1[1] * axis[1])) * (1 - step_cos) + \
                (v1[0] * axis[1] - v1[1] * axis[0]) * step_sin
    ROT[3][3] = 1

    return ROT.T

class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Clay [default]"
    PREFAB = {
        "Polished ceramic": {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    FREE_MODE = 1
    SEG_MODE = 3
    PART_MODE = 5
    JOINT_MODE = 7
    SEG_POS_LABEL = "a"
    SEG_NEG_LABEL = "d"
    RECOVER = "z"
    SET_PART = "s"
    SAVE_ALL = 0
    ANIMATE = "p"
    JOINT_X_UP = "h"
    JOINT_X_DOWN = "j"
    JOINT_Y_UP = "k"
    JOINT_Y_DOWN = "l"
    JOINT_Z_UP = ";"
    JOINT_Z_DOWN = "'"
    JOINT_X_ROT_POS = "b"
    JOINT_X_ROT_NEG = "n"
    JOINT_Y_ROT_POS = "m"
    JOINT_Y_ROT_NEG = ","
    JOINT_Z_ROT_POS = "."
    JOINT_Z_ROT_NEG = "/"
    def __init__(self, use_sam, part_only_mode):
        self.use_sam = use_sam
        self.part_only_mode = part_only_mode
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = False
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 13580
        self.sun_intensity = 13580
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)
        self.rot_steps = 20
        self.rot_delta = np.pi / 180
        self.trans_delta = 0.001
        self.animate_steps = 100
        self.animate_steptime = 0.02
        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.UNLIT]

        self.ui_mode = Settings.FREE_MODE
        if self.use_sam:
            self.seg_points = {
                "points": [],
                "labels": [],
            }
            self.seg_label = Settings.SEG_POS_LABEL

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "park"

    MATERIAL_NAMES = ["Unlit", "Lit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.UNLIT, Settings.LIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height, use_sam, part_only_mode):
        self._do_state_init(use_sam, part_only_mode)
        self.settings = Settings(use_sam, part_only_mode)

        if use_sam:
            import time
            print("==> Loading SAM...")
            sam_load_start_time = time.time()
            from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor
            # sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
            # sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
            # sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
            # self.sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
            sam_checkpoint = "checkpoints/sam_hq_vit_h.pth"
            self.sam = sam_hq_model_registry['vit_h'](checkpoint=sam_checkpoint)
            self.sam.to("cuda")

            self.sam_predictor = SamPredictor(self.sam)
            sam_load_end_time = time.time()
            print("==> Finished. Load time: {:.2f} s".format(sam_load_end_time - sam_load_start_time))
        
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)
        self._scene.enable_scene_caching(False)
        self._scene.visible = True
        self._image = gui.ImageWidget()
        self._image.visible = False
        self._make_settings_panel()
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._image)
        self.window.add_child(self._settings_panel)
        
        self._apply_settings()
        self.window.set_on_key(self._on_keyboard_event)
        self._scene.set_on_mouse(self._on_mouse_event)
        self._image.set_on_mouse(self._on_mouse_event)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        
    def _make_settings_panel(self):
        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        # ----
        self._about_button = gui.Button("Help")
        self._about_button.horizontal_padding_em = 0.5
        self._about_button.vertical_padding_em = 0
        self._about_button.set_on_clicked(self._on_menu_about)
        h = gui.Horiz(0.25 * em)  # row 0
        # h.add_stretch()
        h.add_child(self._about_button)
        # h.add_stretch()
        self._settings_panel.add_child(h)

        self._open_button = gui.Button("Open")
        self._open_button.horizontal_padding_em = 0.5
        self._open_button.vertical_padding_em = 0
        self._open_button.set_on_clicked(self._on_menu_open)
        self.file_io_ctrls = gui.CollapsableVert("File", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        h = gui.Horiz(0.25 * em)  # row 1
        # h.add_stretch()
        h.add_child(self._open_button)
        self.file_io_ctrls.add_child(h)
        
        self.file_io_ctrls_part_or_art = gui.CollapsableVert("File", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        self._save_button = gui.Button("Save")
        self._save_button.horizontal_padding_em = 0.5
        self._save_button.vertical_padding_em = 0
        self._save_button.set_on_clicked(self.save_all)
        h = gui.Horiz(0.25 * em)  # row 1
        # h.add_stretch()
        h.add_child(self._save_button)
        # h.add_stretch()
        self.file_io_ctrls_part_or_art.add_child(h)
        
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.file_io_ctrls)
        self._settings_panel.add_child(self.file_io_ctrls_part_or_art)
        self.file_io_ctrls_part_or_art.visible = False
        
        self.limit_ctrl = gui.CollapsableVert("Joint", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        self.limit_ctrl.add_child(gui.Label("Lower Bound"))
        self.slider_low = gui.Slider(gui.Slider.DOUBLE)
        self.slider_low.set_limits(-180, 180)
        self.limit_ctrl.add_child(self.slider_low)
        self.limit_ctrl.add_child(gui.Label("Upper Bound"))
        self.slider_up = gui.Slider(gui.Slider.DOUBLE)
        self.slider_up.set_limits(-180, 180)
        self.slider_low.double_value = -180
        self.slider_up.double_value = 180
        self.slider_low.set_on_value_changed(self._on_change_joint_range)
        self.slider_up.set_on_value_changed(self._on_change_joint_range)
        self.limit_ctrl.add_child(self.slider_up)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.limit_ctrl)

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        self.view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        self.view_ctrls.set_is_open(False)
        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        # self._fly_button = gui.Button("Fly")
        # self._fly_button.horizontal_padding_em = 0.5
        # self._fly_button.vertical_padding_em = 0
        # self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        # self._model_button = gui.Button("Model")
        # self._model_button.horizontal_padding_em = 0.5
        # self._model_button.vertical_padding_em = 0
        # self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Env")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)

        self.view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 2
        # h.add_stretch()
        h.add_child(self._arcball_button)
        # h.add_child(self._fly_button)
        # h.add_child(self._model_button)
        # h.add_stretch()
        # self.view_ctrls.add_child(h)
        # h = gui.Horiz(0.25 * em)  # row 2
        # h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        # h.add_stretch()
        self.view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        self.view_ctrls.add_fixed(separation_height)
        self.view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        self.view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        self.view_ctrls.add_fixed(separation_height)
        self.view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        self.view_ctrls.add_fixed(separation_height)
        self.view_ctrls.add_child(gui.Label("Lighting profiles"))
        self.view_ctrls.add_child(self._profiles)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.view_ctrls)

        self.advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        self.advanced.set_is_open(False)
        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        self.advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        self.advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        self.advanced.add_fixed(separation_height)
        self.advanced.add_child(gui.Label("Environment"))
        self.advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        self.advanced.add_fixed(separation_height)
        self.advanced.add_child(gui.Label("Sun (Directional light)"))
        self.advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.advanced)

        self.material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        self.material_settings.set_is_open(False)
        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        self.material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self.material_settings)

        self._settings_panel.visible = True
        self.limit_ctrl.visible = False
        self.slider_low.enabled = False
        self.slider_up.enabled = False


    def _do_state_init(self, use_sam, part_only_mode):
        self.use_sam = use_sam
        self.part_only_mode = part_only_mode
        self.mouse_state = []
        self.geometry_dict = []
        self.cluster_tri_num = None
        self.cluster_pertri_index = None
        self.active_cluster_id = None
        self.active_articulation_joint = None
        self.articulation_joints = []
        self.geometry_parts = []
        self.img_parts = []
        self.path = None
        self.curr_image = None

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def set_free_mode(self):
        if self.settings.ui_mode != Settings.FREE_MODE:
            self.mouse_state = []
            self.settings.ui_mode = Settings.FREE_MODE
            self._scene.visible = True
            self._image.visible = False
            self.limit_ctrl.visible = False
            self.file_io_ctrls.visible = True
            self.view_ctrls.visible = True
            self.advanced.visible = True
            self.material_settings.visible = True
            self._settings_panel.visible = True
            self.file_io_ctrls_part_or_art.visible = False
            self.redraw_anything()
            self.curr_image = None
            self._shader.selected_index = 0
            self.settings.set_material(AppWindow.MATERIAL_SHADERS[self._shader.selected_index])
            self._apply_settings()
            if len(self.geometry_dict) >= 1:
                curr_mesh = copy.deepcopy(self.geometry_dict[-1])
                self._scene.scene.clear_geometry()
                self._scene.scene.add_geometry("__model__", curr_mesh, self.settings.material)
                self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            if self.use_sam:
                self.settings.seg_points = {
                    "points": [],
                    "labels": [],
                }
                self.settings.seg_mode = Settings.SEG_POS_LABEL
                self.curr_logits = None
            

    def set_seg_mode(self):
        if self.settings.ui_mode != Settings.SEG_MODE:
            self.mouse_state = []
            self.settings.ui_mode = Settings.SEG_MODE
            self._shader.selected_index = 0
            self.settings.set_material(AppWindow.MATERIAL_SHADERS[self._shader.selected_index])
            self._apply_settings()
            if self.use_sam:
                self.settings.seg_points = {
                    "points": [],
                    "labels": [],
                }
                self.settings.seg_mode = Settings.SEG_POS_LABEL
            def on_image(image):
                img = np.asarray(image) 
                self._image.update_image(image)
                self._settings_panel.visible = True
                self._scene.visible = False
                self._image.visible = True
                self.limit_ctrl.visible = False
                self.file_io_ctrls.visible = False
                self.view_ctrls.visible = False
                self.advanced.visible = False
                self.material_settings.visible = False
                self._settings_panel.visible = True
                self.file_io_ctrls_part_or_art.visible = False
                self.redraw_anything()
                self.curr_image = img.copy()
                if self.use_sam:
                    self.curr_logits = None
                    self.sam_predictor.set_image(img)
            if len(self.geometry_dict) >= 1:
                curr_mesh = copy.deepcopy(self.geometry_dict[-1])
                self._scene.scene.clear_geometry()
                self._scene.scene.add_geometry("__model__", curr_mesh, self.settings.material)
            self._scene.scene.scene.render_to_image(on_image)

    def set_part_mode(self):
        if self.settings.ui_mode != Settings.PART_MODE:
            self.mouse_state = []
            self.settings.ui_mode = Settings.PART_MODE
            self.active_cluster_id = None
            self._shader.selected_index = 1
            self.settings.set_material(AppWindow.MATERIAL_SHADERS[self._shader.selected_index])
            self._apply_settings()
            if len(self.geometry_dict) >= 1:
                def on_image(image):
                    img = np.asarray(image) 
                    self._image.update_image(image)
                    self._scene.visible = False
                    self._image.visible = True
                    self.limit_ctrl.visible = False
                    self.file_io_ctrls.visible = False
                    self.view_ctrls.visible = False
                    self.advanced.visible = False
                    self.material_settings.visible = False
                    self._settings_panel.visible = True
                    self.file_io_ctrls_part_or_art.visible = True
                    self.redraw_anything()
                    self.curr_image = img.copy()
                curr_mesh = copy.deepcopy(self.geometry_dict[-1])
                vertices = np.asarray(curr_mesh.vertices).copy()
                faces = np.asarray(curr_mesh.triangles).copy()
                vertex_colors = np.asarray(curr_mesh.vertex_colors).copy()
                self.cluster_pertri_index, self.cluster_tri_num, _ = curr_mesh.cluster_connected_triangles()
                color_palette = seaborn.color_palette(n_colors=len(self.cluster_tri_num))
                for cluster_id in range(len(self.cluster_tri_num)):
                    this_cluster_faces = faces[np.asarray(self.cluster_pertri_index)==cluster_id]
                    this_cluster_faces_flatten = this_cluster_faces.reshape(-1)
                    vertex_colors[this_cluster_faces_flatten] = color_palette[cluster_id]
                new_mesh = copy.deepcopy(curr_mesh)
                new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                self._scene.scene.clear_geometry()
                self._scene.scene.add_geometry("__model__", new_mesh, self.settings.material)
                self._scene.scene.scene.render_to_image(on_image)
            else:
                def on_image(image):
                    img = (np.ones_like(np.asarray(image)) * 255).astype(np.uint8)
                    self._image.update_image(o3d.geometry.Image(img))
                    self._scene.visible = False
                    self._image.visible = True
                    self.limit_ctrl.visible = False
                    self.file_io_ctrls.visible = False
                    self.view_ctrls.visible = False
                    self.advanced.visible = False
                    self.material_settings.visible = False
                    self._settings_panel.visible = True
                    self.file_io_ctrls_part_or_art.visible = True
                    self.redraw_anything()
                    self.curr_image = img.copy()
                self._scene.scene.scene.render_to_image(on_image)
                
    def set_joint_mode(self):
        if self.settings.ui_mode != Settings.JOINT_MODE:
            self.mouse_state = []
            self.settings.ui_mode = Settings.JOINT_MODE
            self.active_cluster_id = None
            self.active_articulation_joint = None
            self._shader.selected_index = 1
            self.settings.set_material(AppWindow.MATERIAL_SHADERS[self._shader.selected_index])
            self._apply_settings()
            def on_image(image):
                img = np.asarray(image) 
                self._image.update_image(image)
                self._scene.visible = False
                self._image.visible = True
                self.limit_ctrl.visible = True
                self.file_io_ctrls.visible = False
                self.view_ctrls.visible = False
                self.advanced.visible = False
                self.material_settings.visible = False
                self._settings_panel.visible = True
                self.redraw_anything()
                self.curr_image = img.copy()
            if len(self.geometry_parts) >= 1:
                self._scene.scene.clear_geometry()
                color_palette = seaborn.color_palette(n_colors=len(self.geometry_parts))
                for part_idx, part in enumerate(self.geometry_parts):
                    vertex_colors = np.asarray(part.vertex_colors).copy()
                    vertex_colors[:] = color_palette[part_idx]
                    new_part = copy.deepcopy(part)
                    new_part.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                    self._scene.scene.add_geometry(f"__model__.{part_idx}", new_part, self.settings.material)
            if len(self.articulation_joints) >= 1:
                for joint_idx, joint in enumerate(self.articulation_joints):
                    hitted_point, primitive_normal = joint[:2]
                    arrow = get_arrow_from_ori_and_dir(hitted_point, primitive_normal)
                    vertex_colors = np.zeros_like(np.asarray(arrow.vertices))
                    vertex_colors[:, 0] = 1.0
                    arrow.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                    arrow.compute_vertex_normals()
                    self._scene.scene.add_geometry(f"__model__.articulation{joint_idx}", arrow, self.settings.material)
            if len(self.geometry_parts) >= 1 or len(self.articulation_joints) >= 1:
                self._scene.scene.scene.render_to_image(on_image)
            else:
                def on_image(image):
                    img = (np.ones_like(np.asarray(image)) * 255).astype(np.uint8)
                    self._image.update_image(o3d.geometry.Image(img))
                    self._scene.visible = False
                    self._image.visible = True
                    self.limit_ctrl.visible = True
                    self.file_io_ctrls.visible = False
                    self.view_ctrls.visible = False
                    self.advanced.visible = False
                    self.material_settings.visible = False
                    self._settings_panel.visible = True
                    self.redraw_anything()
                    self.curr_image = img.copy()
                self._scene.scene.scene.render_to_image(on_image)

    def change_seg_label_mode(self, is_pos=True):
        if is_pos:
            if self.settings.ui_mode == Settings.SEG_MODE:
                self.settings.seg_mode = Settings.SEG_POS_LABEL
        else:
            if self.settings.ui_mode == Settings.SEG_MODE:
                self.settings.seg_mode = Settings.SEG_NEG_LABEL

    def do_recover(self):
        if self.settings.ui_mode == Settings.SEG_MODE:
            if len(self.geometry_dict) > 1:
                def on_image(image):
                    img = np.asarray(image) 
                    self._image.update_image(image)
                    self.curr_image = img.copy()
                    self.redraw_anything()
                curr_mesh = copy.deepcopy(self.geometry_dict[-2])
                self._scene.scene.clear_geometry()
                self._scene.scene.add_geometry("__model__", curr_mesh, self.settings.material)
                del self.geometry_dict[-1]
                self._scene.scene.scene.render_to_image(on_image)
        elif self.settings.ui_mode == Settings.PART_MODE:
            if len(self.geometry_parts) >= 1 and len(self.img_parts) >= 1:
                self._on_recover_part()
        elif self.settings.ui_mode == Settings.JOINT_MODE:
            if len(self.articulation_joints) >= 1:
                self._on_recover_joint()
                
    def do_forget_part(self):
        if len(self.articulation_joints) >= 1:
            affected_joint_idxs = []
            affected_part_id = len(self.geometry_parts) - 1
            for joint_idx, joint in enumerate(self.articulation_joints):
                cluster_ids = [x["cluster_id"] for x in joint[2]]
                if affected_part_id in cluster_ids:
                    affected_joint_idxs.append(joint_idx)
            affected_joint_idxs = sorted(affected_joint_idxs)[::-1]
            for affected_joint_idx in affected_joint_idxs:
                del self.articulation_joints[affected_joint_idx]
        del self.geometry_parts[-1]
        del self.img_parts[-1]
        self.window.close_dialog()
        pyautogui.moveRel(1, 0)
        if self._scene.visible:
            self.window.set_focus_widget(self._scene)
        if self._image.visible:
            self.window.set_focus_widget(self._image)
    
    def do_forget_joint(self):
        del self.articulation_joints[-1]
        self.window.close_dialog()
        pyautogui.moveRel(1, 0)
        if self._scene.visible:
            self.window.set_focus_widget(self._scene)
        if self._image.visible:
            self.window.set_focus_widget(self._image)
        def on_image(image):
            img = np.asarray(image) 
            self._image.update_image(image)
            self.curr_image = img.copy()
            self.redraw_anything()
        self.active_cluster_id = None
        self.active_articulation_joint = None
        self._scene.scene.clear_geometry()
        color_palette = seaborn.color_palette(n_colors=len(self.geometry_parts))
        for part_idx, part in enumerate(self.geometry_parts):
            vertex_colors = np.asarray(part.vertex_colors).copy()
            vertex_colors[:] = color_palette[part_idx]
            new_part = copy.deepcopy(part)
            new_part.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            self._scene.scene.add_geometry(f"__model__.{part_idx}", new_part, self.settings.material)
        if len(self.articulation_joints) >= 1:
            for joint_idx, joint in enumerate(self.articulation_joints):
                hitted_point, primitive_normal = joint[:2]
                arrow = get_arrow_from_ori_and_dir(hitted_point, primitive_normal)
                vertex_colors = np.zeros_like(np.asarray(arrow.vertices))
                vertex_colors[:, 0] = 1.0
                arrow.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                arrow.compute_vertex_normals()
                self._scene.scene.add_geometry(f"__model__.articulation{joint_idx}", arrow, self.settings.material)
        self._scene.scene.scene.render_to_image(on_image)

    def deselect_part(self):
        if self.settings.ui_mode == Settings.PART_MODE:
            if self.active_cluster_id is not None:
                self.active_cluster_id = None
                self._shader.selected_index = 0
                self.settings.set_material(AppWindow.MATERIAL_SHADERS[self._shader.selected_index])
                self._apply_settings()
                self._image.update_image(o3d.geometry.Image(self.curr_image))

    def deselect_part_for_joint(self):
        if self.settings.ui_mode == Settings.JOINT_MODE:
            if isinstance(self.active_cluster_id, list) and len(self.active_cluster_id) >= 1:
                if self.active_articulation_joint is not None:
                    self.active_articulation_joint = None
                else:
                    if len(self.active_cluster_id) == 2:
                        del self.active_cluster_id[-1]
                    else:
                        self.active_cluster_id = None
                self.slider_low.enabled = False
                self.slider_up.enabled = False
                self._scene.scene.clear_geometry()
                color_palette = seaborn.color_palette(n_colors=len(self.geometry_parts))
                for part_idx, part in enumerate(self.geometry_parts):
                    vertex_colors = np.asarray(part.vertex_colors).copy()
                    vertex_colors[:] = color_palette[part_idx]
                    new_part = copy.deepcopy(part)
                    new_part.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                    self._scene.scene.add_geometry(f"__model__.{part_idx}", new_part, self.settings.material)
                if len(self.articulation_joints) >= 1:
                    for joint_idx, joint in enumerate(self.articulation_joints):
                        hitted_point, primitive_normal = joint[:2]
                        arrow = get_arrow_from_ori_and_dir(hitted_point, primitive_normal)
                        vertex_colors = np.zeros_like(np.asarray(arrow.vertices))
                        vertex_colors[:, 0] = 1.0
                        arrow.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                        arrow.compute_vertex_normals()
                        self._scene.scene.add_geometry(f"__model__.articulation{joint_idx}", arrow, self.settings.material)
                def on_image(image):
                    img = np.asarray(image) 
                    if self.active_cluster_id is not None:
                        for part_dict in self.active_cluster_id:
                            cv2.circle(img, part_dict["mouse_state"], 3, (255, 0, 0), -1)
                    self._image.update_image(image)
                    self.redraw_anything()
                self._scene.scene.scene.render_to_image(on_image)
                self._scene.visible = False
                self._image.visible = True
                self.redraw_anything()
    def move_joint(self, key):
        if self.active_articulation_joint is not None:
            hitted_point, primitive_normal = copy.deepcopy(self.active_articulation_joint[:2])
        else:
            return
        if key == Settings.JOINT_X_UP:
            hitted_point[0] += self.settings.trans_delta
        elif key == Settings.JOINT_X_DOWN:
            hitted_point[0] -= self.settings.trans_delta
        elif key == Settings.JOINT_Y_UP:
            hitted_point[1] += self.settings.trans_delta
        elif key == Settings.JOINT_Y_DOWN:
            hitted_point[1] -= self.settings.trans_delta
        elif key == Settings.JOINT_Z_UP:
            hitted_point[2] += self.settings.trans_delta
        elif key == Settings.JOINT_Z_DOWN:
            hitted_point[2] -= self.settings.trans_delta
        elif key == Settings.JOINT_X_ROT_POS:
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz([self.settings.rot_delta, 0, 0])
            primitive_normal = (rot_mat @ np.expand_dims(primitive_normal, -1))[:, 0]
        elif key == Settings.JOINT_X_ROT_NEG:
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz([-self.settings.rot_delta, 0, 0])
            primitive_normal = (rot_mat @ np.expand_dims(primitive_normal, -1))[:, 0]
        elif key == Settings.JOINT_Y_ROT_POS:
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz([0, self.settings.rot_delta, 0])
            primitive_normal = (rot_mat @ np.expand_dims(primitive_normal, -1))[:, 0]
        elif key == Settings.JOINT_Y_ROT_NEG:
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz([0, -self.settings.rot_delta, 0])
            primitive_normal = (rot_mat @ np.expand_dims(primitive_normal, -1))[:, 0]
        elif key == Settings.JOINT_Z_ROT_POS:
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, self.settings.rot_delta])
            primitive_normal = (rot_mat @ np.expand_dims(primitive_normal, -1))[:, 0]
        elif key == Settings.JOINT_Z_ROT_NEG:
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, -self.settings.rot_delta])
            primitive_normal = (rot_mat @ np.expand_dims(primitive_normal, -1))[:, 0]
        self.active_articulation_joint[:2] = hitted_point, primitive_normal
        def on_image(image):
            self.active_articulation_joint[3] = image
            self._image.update_image(image)
            self.redraw_anything()
        arrow = get_arrow_from_ori_and_dir(hitted_point, primitive_normal)
        vertex_colors = np.zeros_like(np.asarray(arrow.vertices))
        vertex_colors[:, 0] = 1.0
        arrow.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        arrow.compute_vertex_normals()
        self._scene.scene.remove_geometry(f"__model__.articulation")
        self._scene.scene.add_geometry(f"__model__.articulation", arrow, self.settings.material)
        cluster_id = self.active_cluster_id[1]["cluster_id"]
        ori_transform = copy.deepcopy(self._scene.scene.get_geometry_transform(f"__model__.{cluster_id}"))
        child_transform_1 = RotateAnyAxis(self.active_articulation_joint[0], self.active_articulation_joint[0]+self.active_articulation_joint[1], np.pi * self.slider_low.double_value / 180)
        child_transform_2 = RotateAnyAxis(self.active_articulation_joint[0], self.active_articulation_joint[0]+self.active_articulation_joint[1], np.pi * self.slider_up.double_value / 180)
        self._scene.scene.set_geometry_transform(f"__model__.{cluster_id}.lower", child_transform_1 @ ori_transform)
        self._scene.scene.set_geometry_transform(f"__model__.{cluster_id}.upper", child_transform_2 @ ori_transform)
        self._scene.scene.scene.render_to_image(on_image)
        
    def set_part(self):
        if self.settings.ui_mode == Settings.PART_MODE and self.active_cluster_id is not None:
            curr_mesh = self.geometry_dict[-1]
            new_mesh = copy.deepcopy(curr_mesh)
            new_mesh.remove_triangles_by_mask(np.asarray(self.cluster_pertri_index)!=self.active_cluster_id)
            new_mesh.remove_unreferenced_vertices()
            new_mesh = o3d.t.geometry.TriangleMesh.from_legacy(new_mesh)
            new_mesh = new_mesh.to_legacy()
            self.geometry_parts.append(new_mesh)
            def on_image(image):
                self.img_parts.append(image)
            self._scene.scene.scene.render_to_image(on_image)
            self.active_cluster_id = None
            self._image.update_image(o3d.geometry.Image(self.curr_image))
            self.redraw_anything()

    def set_joint(self):
        if self.settings.ui_mode == Settings.JOINT_MODE and self.active_articulation_joint is not None:
            self.articulation_joints.append(copy.deepcopy(self.active_articulation_joint))
            self.active_cluster_id = None
            self.active_articulation_joint = None
            self.slider_low.enabled = False
            self.slider_up.enabled = False

            self._scene.scene.clear_geometry()
            color_palette = seaborn.color_palette(n_colors=len(self.geometry_parts))
            def on_image(image):
                img = np.asarray(image) 
                self._image.update_image(image)
                self.redraw_anything()
            for part_idx, part in enumerate(self.geometry_parts):
                vertex_colors = np.asarray(part.vertex_colors).copy()
                vertex_colors[:] = color_palette[part_idx]
                new_part = copy.deepcopy(part)
                new_part.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                self._scene.scene.add_geometry(f"__model__.{part_idx}", new_part, self.settings.material)
            if len(self.articulation_joints) >= 1:
                for joint_idx, joint in enumerate(self.articulation_joints):
                    hitted_point, primitive_normal = joint[:2]
                    arrow = get_arrow_from_ori_and_dir(hitted_point, primitive_normal)
                    vertex_colors = np.zeros_like(np.asarray(arrow.vertices))
                    vertex_colors[:, 0] = 1.0
                    arrow.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                    arrow.compute_vertex_normals()
                    self._scene.scene.add_geometry(f"__model__.articulation{joint_idx}", arrow, self.settings.material)
            self._scene.scene.scene.render_to_image(on_image)

    def save_all(self):
        if self.path is not None:
            if self.settings.ui_mode == Settings.JOINT_MODE:
                self.export_urdf()
            elif self.settings.ui_mode == Settings.PART_MODE:
                self.export_parts()
            self._on_menu_save()
    
    def export_parts(self):
        dirname = os.path.dirname(self.path)
        parts_dirname = os.path.join(dirname, "parts")
        if os.path.exists(parts_dirname):
            shutil.rmtree(parts_dirname)
        os.makedirs(parts_dirname, exist_ok=True)
        for partidx, partimg in enumerate(zip(self.geometry_parts, self.img_parts)):
            part, img = partimg
            o3d.io.write_triangle_mesh(os.path.join(parts_dirname, f"part_{partidx}.ply"), part)
            o3d.io.write_image(os.path.join(parts_dirname, f"part_{partidx}.png"), img)
            blender_ply2obj(os.path.join(parts_dirname, f"part_{partidx}.ply"), os.path.join(parts_dirname, f"part_{partidx}.obj"))
            
    def build_articulation_tree(self):
        from treelib import Node, Tree
        tree = Tree()
        tree.create_node("root", "root")
        joints = []
        if len(self.articulation_joints) >= 1:
            joints.extend(self.articulation_joints)
        if self.active_articulation_joint is not None:
            joints.append(self.active_articulation_joint)

        has_parent_list = [joint[2][1]["cluster_id"] for joint in joints]
        root_lists = set(list(range(len(self.geometry_parts)))) - set(has_parent_list)
        for root in root_lists:
            tree.create_node(str(root), str(root), parent="root", data=None)
        
        for joint_idx, joint in enumerate(joints):
            tree.create_node(str(joint[2][1]["cluster_id"]), str(joint[2][1]["cluster_id"]), parent=str(joint[2][0]["cluster_id"]), \
                data={
                    "joint": copy.deepcopy(joint),
                    "joint_id": copy.deepcopy(joint_idx)})
        return tree

    # TODO: prismatic
    def calc_animate_step(self, tree, node, transform=np.eye(4)):
        for child in tree.children(node.tag):
            if child.data is not None:
                joint = child.data["joint"]
                joint[0] = (transform @ np.expand_dims(np.concatenate([joint[0], np.array([1])]), 1))[:3, 0]
                joint[1] = (transform @ np.expand_dims(np.concatenate([joint[1], np.array([1])]), 1))[:3, 0]
                child_transform = RotateAnyAxis(joint[0], joint[0]+joint[1], (joint[4][1] - joint[4][0]) / self.settings.rot_steps * np.pi / 180)
                curr_transform = self._scene.scene.get_geometry_transform(f"__model__.{int(child.tag)}")
                self._scene.scene.set_geometry_transform(f"__model__.{int(child.tag)}", \
                                                         child_transform @ transform @ curr_transform)
                curr_transform = self._scene.scene.get_geometry_transform(f"__model__.articulation{int(child.data['joint_id'])}")
                self._scene.scene.set_geometry_transform(f"__model__.articulation{int(child.data['joint_id'])}", \
                                                         child_transform @ transform @ curr_transform)
                self.calc_animate_step(tree, child, child_transform @ transform)
            else:
                self.calc_animate_step(tree, child, transform)

    def animate(self):
        self._scene.visible = True
        self._image.visible = False
        self.redraw_anything()
        gui.Application.instance.run_one_tick()
        art_tree = self.build_articulation_tree()
        ori_transforms = [copy.deepcopy(self._scene.scene.get_geometry_transform(f"__model__.{part_idx}")) for part_idx in range(len(self.geometry_parts))]
        ori_transforms_joint = [copy.deepcopy(self._scene.scene.get_geometry_transform(f"__model__.articulation{joint_idx}")) for joint_idx in range(len(self.articulation_joints))]
        for step_iter in range(self.settings.rot_steps):
            self.calc_animate_step(art_tree, art_tree.get_node("root"))
            self.redraw_anything()
            gui.Application.instance.run_one_tick()
            time.sleep(0.1)
        for part_idx in range(len(self.geometry_parts)):
            self._scene.scene.set_geometry_transform(f"__model__.{part_idx}", ori_transforms[part_idx])
        for joint_idx in range(len(self.articulation_joints)):
            self._scene.scene.set_geometry_transform(f"__model__.articulation{joint_idx}", ori_transforms_joint[joint_idx])
        self._image.visible = True
        self._scene.visible = False
        self.redraw_anything()
        gui.Application.instance.run_one_tick()

    def _on_change_joint_range(self, arg0):
        if self.active_articulation_joint is not None:
            def on_image(image):
                self.active_articulation_joint[3] = image
                self.active_articulation_joint[4] = (self.slider_low.double_value, self.slider_up.double_value)
                self._image.update_image(image)
                self.redraw_anything()
            cluster_id = self.active_cluster_id[1]["cluster_id"]
            ori_transform = copy.deepcopy(self._scene.scene.get_geometry_transform(f"__model__.{cluster_id}"))
            child_transform_1 = RotateAnyAxis(self.active_articulation_joint[0], self.active_articulation_joint[0]+self.active_articulation_joint[1], np.pi * self.slider_low.double_value / 180)
            child_transform_2 = RotateAnyAxis(self.active_articulation_joint[0], self.active_articulation_joint[0]+self.active_articulation_joint[1], np.pi * self.slider_up.double_value / 180)
            self._scene.scene.set_geometry_transform(f"__model__.{cluster_id}.lower", child_transform_1 @ ori_transform)
            self._scene.scene.set_geometry_transform(f"__model__.{cluster_id}.upper", child_transform_2 @ ori_transform)
            self._scene.scene.scene.render_to_image(on_image)

    def redraw_anything(self):
        self.window.set_needs_layout()
        self._scene.force_redraw()
        self.window.post_redraw()
        
    def parse_mouse_sam(self, event):
        # NOTE: WIP
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
            self.settings.seg_points["points"].append([int(event.x), int(event.y)])
            if self.settings.seg_mode == Settings.SEG_POS_LABEL:
                self.settings.seg_points["labels"].append(1)
            elif self.settings.seg_mode == Settings.SEG_NEG_LABEL:
                self.settings.seg_points["labels"].append(0)
            masks, _, logits = self.sam_predictor.predict(
                point_coords=np.array(self.settings.seg_points["points"]),
                point_labels=np.array(self.settings.seg_points["labels"]),
                mask_input=self.curr_logits,
                box=None,
                multimask_output=False,
            )
            self.curr_logits = logits
            transposed_mask = np.transpose(masks, (1, 2, 0)).astype(float) * np.expand_dims(np.expand_dims(np.array([0, 123 / 255, 167 / 255]), 0), 0) * 0.7 + 0.3
            masked_image = (transposed_mask * self.curr_image.copy().astype(float)).astype(np.uint8)
            for point, label in zip(self.settings.seg_points["points"], self.settings.seg_points["labels"]):
                point_x, point_y = point
                if label == 0:
                    cv2.circle(masked_image, (point_x, point_y), 5, (255, 0, 0), -1)
                elif label == 1:
                    cv2.circle(masked_image, (point_x, point_y), 5, (0, 255, 0), -1)
            masked_image = o3d.geometry.Image(masked_image)
            self._image.update_image(masked_image)

    def parse_mouse_split(self, event):
        if len(self.geometry_dict) >= 1:
            if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_DOWN:
                self.mouse_state = [(int(event.x), int(event.y))]
            elif event.type == o3d.visualization.gui.MouseEvent.Type.DRAG:
                self.mouse_state = [self.mouse_state[0], (int(event.x), int(event.y))]
                masked_image = self.curr_image.copy()
                cv2.line(masked_image, (self.mouse_state[0][0], self.mouse_state[0][1]), (self.mouse_state[1][0], self.mouse_state[1][1]), (0, 255, 0), 2)
                masked_image = o3d.geometry.Image(masked_image)
                self._image.update_image(masked_image)
            elif event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
                if len(self.mouse_state) > 1:
                    camera = self._scene.scene.camera
                    width = int(self._scene.frame.width)
                    height = int(self._scene.frame.height)
                    mouse_state_0 = np.array(self.mouse_state[0])
                    mouse_state_1 = np.array(self.mouse_state[1])
                    segment_direction = mouse_state_1 - mouse_state_0
                    segment_length = np.linalg.norm(segment_direction)
                    point1 = camera.unproject(self.mouse_state[0][0], self.mouse_state[0][1], 0, view_width=width, view_height=height)
                    point2 = camera.unproject(self.mouse_state[1][0], self.mouse_state[1][1], 0, view_width=width, view_height=height)
                    point3 = camera.unproject(self.mouse_state[1][0], self.mouse_state[1][1], 1000, view_width=width, view_height=height)
                    split_plane_normal = np.cross((point2 - point3), (point1 - point2))
                    split_plane_point = (point1 + point2) / 2
                    curr_mesh = copy.deepcopy(self.geometry_dict[-1])
                    vertices = np.asarray(curr_mesh.vertices)    
                    faces = np.asarray(curr_mesh.triangles)
                    vertices_edges_loc = vertices[faces]
                    first_edges_loc = vertices_edges_loc[:, [0, 1]]
                    second_edges_loc = vertices_edges_loc[:, [1, 2]]
                    third_edges_loc = vertices_edges_loc[:, [2, 0]]
                    is_intersect = None
                    for loc in [first_edges_loc, second_edges_loc, third_edges_loc]:
                        line_direction = loc[:, 1] - loc[:, 0]
                        dot_product = np.sum(np.expand_dims(split_plane_normal, axis=0) * line_direction, axis=1)
                        distance_to_plane = np.sum((split_plane_point - loc[:, 0]) * split_plane_normal, axis=1) / dot_product
                        is_intersect_thisturn = np.logical_and(distance_to_plane >= 0, distance_to_plane <= 1)
                        intersection_point = loc[:, 0] + np.expand_dims(distance_to_plane, axis=1) * line_direction
                        projview = np.expand_dims(camera.get_projection_matrix() @ camera.get_view_matrix(), axis=0)
                        intersect_pt_add_one = np.expand_dims(np.concatenate((intersection_point, np.ones_like(intersection_point[:, :1])), axis=1), axis=2)
                        NDCw = (projview @ intersect_pt_add_one)[:,:,0]
                        screenX = (NDCw[:, 0] / NDCw[:, 3] + 1) * 0.5 * width
                        screenY = (1 - NDCw[:, 1] / NDCw[:, 3]) * 0.5 * height
                        screenX = np.expand_dims(screenX, axis=1)
                        screenY = np.expand_dims(screenY, axis=1)
                        screenXY = np.concatenate([screenX, screenY], axis=1)
                        projection_length = np.sum((screenXY - np.expand_dims(mouse_state_0, axis=0)) * np.expand_dims(segment_direction, axis=0), axis=1) / (segment_length ** 2)
                        is_in_length = np.logical_and(projection_length >= 0, projection_length <= 1)
                        is_intersect_thisturn = np.logical_and(is_intersect_thisturn, is_in_length)
                        if is_intersect is None:
                            is_intersect = is_intersect_thisturn
                        else:
                            is_intersect = np.logical_or(is_intersect, is_intersect_thisturn)
                    curr_mesh.remove_triangles_by_mask(is_intersect)

                    self._scene.scene.clear_geometry()
                    self._scene.scene.add_geometry("__model__", curr_mesh, self.settings.material)
                    self.geometry_dict.append(curr_mesh)
                    def on_image(image):
                        img = np.asarray(image) 
                        self._image.update_image(image)
                        self.curr_image = img.copy()
                        self.redraw_anything()
                    self._scene.scene.scene.render_to_image(on_image)

    def do_raycast(self, meshes, mouse_state):
        width = int(self._scene.frame.width)
        height = int(self._scene.frame.height)
        endpoint = self._scene.scene.camera.unproject(mouse_state[0], mouse_state[1], 0, view_width=width, view_height=height)
        startpoint = np.linalg.inv(self._scene.scene.camera.get_view_matrix())[:3, 3]
        raydir = endpoint - startpoint
        raydir = raydir / np.linalg.norm(raydir)
        scene = o3d.t.geometry.RaycastingScene()
        for curr_mesh in meshes:
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(copy.deepcopy(curr_mesh)))
        rays = np.expand_dims(np.concatenate([startpoint, raydir]), 0)
        o3d_rays = o3d.core.Tensor(rays,
                dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(o3d_rays)
        return ans, rays

    def export_urdf(self):
        dirname = os.path.dirname(self.path)
        parts_dirname = os.path.join(dirname, "parts")
        if os.path.exists(parts_dirname):
            shutil.rmtree(parts_dirname)
        os.makedirs(parts_dirname, exist_ok=True)

        import xml.etree.ElementTree as ET
        from xml.etree.ElementTree import Element, tostring, SubElement, Comment, ElementTree, XML
        import xml.dom.minidom
        root = Element('robot', name="root")
        link_list = []
        joint_list = []
        logged_link_id = []

        tree = self.build_articulation_tree()
        for joint_idx, joint_info in enumerate(self.articulation_joints):
            parent_part_id = joint_info[2][0]["cluster_id"]
            child_part_id = joint_info[2][1]["cluster_id"]
            joint_type = "revolute"
            parent_name = f"part_{parent_part_id}"
            child_name = f"part_{child_part_id}"
            parent_node = tree.get_node(str(parent_part_id))
            if parent_node.data is not None:
                child_origin_xyz_np = joint_info[0] - parent_node.data["joint"][0]
            else:
                child_origin_xyz_np = joint_info[0]
            child_origin_xyz = " ".join([str(x) for x in child_origin_xyz_np])
            if not parent_part_id in logged_link_id:
                part, img = copy.deepcopy(self.geometry_parts[parent_part_id]), self.img_parts[parent_part_id]
                o3d.io.write_triangle_mesh(os.path.join(parts_dirname, f"part_{parent_part_id}.ply"), part)
                o3d.io.write_image(os.path.join(parts_dirname, f"part_{parent_part_id}.png"), img)
                blender_ply2obj(os.path.join(parts_dirname, f"part_{parent_part_id}.ply"), os.path.join(parts_dirname, f"part_{parent_part_id}.obj"))
                
                parent_element = Element('link', name=parent_name)
                parent_visual = SubElement(parent_element, 'visual')
                parent_origin = SubElement(parent_visual, 'origin', rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
                parent_geometry = SubElement(parent_visual, 'geometry')
                parent_mesh = SubElement(parent_geometry, 'mesh', filename=f"parts/part_{parent_part_id}.obj", scale="1 1 1")
                parent_collision = SubElement(parent_element, 'collision')
                parent_collision_origin = SubElement(parent_collision, 'origin', rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
                parent_collision_geometry = SubElement(parent_collision, 'geometry')
                parent_collision_mesh = SubElement(parent_collision_geometry, 'mesh', filename=f"parts/part_{parent_part_id}.obj", scale="1 1 1")
                parent_inertial = SubElement(parent_element, 'inertial')
                node_inertial = XML(
                    '''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="3.0"/><inertia ixx="100" ixy="100" ixz="100" iyy="100" iyz="100" izz="100"/></inertial>''')
                parent_inertial.extend(node_inertial)
                for mass in parent_inertial.iter('mass'):
                    mass.set('value', "0.0")
                for inertia in parent_inertial.iter('inertia'):
                    inertia.set('ixx', "0.0")
                    inertia.set('ixy', "0.0")
                    inertia.set('ixz', "0.0")
                    inertia.set('iyy', "0.0")
                    inertia.set('iyz', "0.0")
                    inertia.set('izz', "0.0")
                link_list.append(parent_element)
                logged_link_id.append(parent_part_id)

            joint = Element('joint', name=f"joint_{joint_idx}", type=joint_type)
            joint_parent = SubElement(joint, "parent", link=parent_name)
            joint_child = SubElement(joint, "child", link=child_name)

            axis_xyz = ' '.join(str(x) for x in joint_info[1])

            origin = SubElement(joint, "origin",
                                xyz=child_origin_xyz,
                                rpy="0 0 0")
            axis = SubElement(joint, "axis", xyz=axis_xyz)
            limit = SubElement(joint, "limit", effort="1.0", lower=str(joint_info[4][0] * np.pi / 180), upper=str(joint_info[4][1] * np.pi / 180), velocity="1000")
            joint_list.append(joint)

            if not child_part_id in logged_link_id:
                part, img = copy.deepcopy(self.geometry_parts[child_part_id]), self.img_parts[child_part_id]
                vertices = np.asarray(part.vertices)
                vertices = vertices - np.expand_dims(child_origin_xyz_np, axis=0)
                part.vertices = o3d.utility.Vector3dVector(vertices)
                o3d.io.write_triangle_mesh(os.path.join(parts_dirname, f"part_{child_part_id}.ply"), part)
                o3d.io.write_image(os.path.join(parts_dirname, f"part_{child_part_id}.png"), img)
                blender_ply2obj(os.path.join(parts_dirname, f"part_{child_part_id}.ply"), os.path.join(parts_dirname, f"part_{child_part_id}.obj"))
                child_element = Element('link', name=child_name)
                child_visual = SubElement(child_element, 'visual')
                child_origin = SubElement(child_visual, 'origin', rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
                child_geometry = SubElement(child_visual, 'geometry')
                child_mesh = SubElement(child_geometry, 'mesh', filename=f"parts/part_{child_part_id}.obj", scale="1 1 1")
                child_collision = SubElement(child_element, 'collision')
                child_collision_origin = SubElement(child_collision, 'origin', rpy="0.0 0.0 0.0", xyz="0.0 0.0 0.0")
                child_collision_geometry = SubElement(child_collision, 'geometry')
                child_collision_mesh = SubElement(child_collision_geometry, 'mesh', filename=f"parts/part_{child_part_id}.obj", scale="1 1 1")
                child_inertial = SubElement(child_element, 'inertial')
                node_inertial = XML(
                    '''<inertial><origin rpy="0 0 0" xyz="0 0 0"/><mass value="3.0"/><inertia ixx="100" ixy="100" ixz="100" iyy="100" iyz="100" izz="100"/></inertial>''')
                child_inertial.extend(node_inertial)
                for mass in child_inertial.iter('mass'):
                    mass.set('value', "0.0")
                for inertia in child_inertial.iter('inertia'):
                    inertia.set('ixx', "0.0")
                    inertia.set('ixy', "0.0")
                    inertia.set('ixz', "0.0")
                    inertia.set('iyy', "0.0")
                    inertia.set('iyz', "0.0")
                    inertia.set('izz', "0.0")
                link_list.append(child_element)
                logged_link_id.append(child_part_id)

        root.extend(link_list)
        root.extend(joint_list)
        xml_string = xml.dom.minidom.parseString(tostring(root))
        xml_pretty_str = xml_string.toprettyxml()
        tree = ET.ElementTree(root)
        with open(os.path.join(dirname, "object.urdf"), "w") as f:
            f.write(xml_pretty_str)

    def parse_mouse_part_select(self, event):
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP and self.active_cluster_id is None and len(self.geometry_dict) >= 1:
            mouse_state = (int(event.x), int(event.y))
            ans, _ = self.do_raycast([self.geometry_dict[-1]], mouse_state)
            t_hit = ans['t_hit'].numpy()[0]
            if not np.isinf(t_hit):
                def on_image(image):
                    img = np.asarray(image) 
                    self._image.update_image(image)
                    self.redraw_anything()
                face_id = int(ans['primitive_ids'].numpy()[0])
                cluster_id = int(np.asarray(self.cluster_pertri_index)[face_id])
                self.active_cluster_id = cluster_id
                curr_mesh = self.geometry_dict[-1]
                new_mesh = copy.deepcopy(curr_mesh)
                new_mesh.remove_triangles_by_mask(np.asarray(self.cluster_pertri_index)!=cluster_id)
                self._shader.selected_index = 0
                self.settings.set_material(AppWindow.MATERIAL_SHADERS[self._shader.selected_index])
                self._apply_settings()
                self._scene.scene.clear_geometry()
                self._scene.scene.add_geometry("__model__", new_mesh, self.settings.material)
                self._scene.scene.scene.render_to_image(on_image)

    def parse_mouse_part_select_for_joint(self, event):
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP and not (isinstance(self.active_cluster_id, list) and len(self.active_cluster_id) >= 2) and len(self.geometry_parts) >= 1:
            mouse_state = (int(event.x), int(event.y))
            ans, _ = self.do_raycast(self.geometry_parts, mouse_state)
            t_hit = ans['t_hit'].numpy()[0]
            if not np.isinf(t_hit):
                def on_image(image):
                    img = np.asarray(image) 
                    for part_dict in self.active_cluster_id:
                        cv2.circle(img, part_dict["mouse_state"], 3, (255, 0, 0), -1)
                    self._image.update_image(image)
                    self.redraw_anything()
                cluster_id = int(ans['geometry_ids'].numpy()[0])
                if self.active_cluster_id is None:
                    self.active_cluster_id = [{"cluster_id": cluster_id, "mouse_state": mouse_state}]
                elif cluster_id not in [x["cluster_id"] for x in self.active_cluster_id]:
                    no_add_joint = False
                    if len(self.articulation_joints) >= 1:
                        possible_all_cluster_ids = [self.active_cluster_id[0]["cluster_id"], cluster_id]
                        for joint in self.articulation_joints:
                            if set(possible_all_cluster_ids) == set([x["cluster_id"] for x in joint[2]]):
                                no_add_joint = True
                                break
                            if possible_all_cluster_ids[1] == joint[2][1]["cluster_id"] and possible_all_cluster_ids[0] != joint[2][0]["cluster_id"]:
                                no_add_joint = True
                                break
                    if not no_add_joint:
                        self.active_cluster_id.append({"cluster_id": cluster_id, "mouse_state": mouse_state})
                if isinstance(self.active_cluster_id, list) and len(self.active_cluster_id) == 2:
                    self._scene.scene.clear_geometry()
                    color_palette = seaborn.color_palette(n_colors=2)
                    for part_idx_idx, part_dict in enumerate(self.active_cluster_id):
                        part = self.geometry_parts[part_dict["cluster_id"]]
                        vertex_colors = np.asarray(part.vertex_colors).copy()
                        vertex_colors[:] = color_palette[part_idx_idx]
                        new_part = copy.deepcopy(part)
                        new_part.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                        self._scene.scene.add_geometry(f"__model__.{part_dict['cluster_id']}", new_part, self.settings.material)
                self._scene.scene.scene.render_to_image(on_image)

    def parse_mouse_build_joint(self, event):
        if event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP and isinstance(self.active_cluster_id, list) and len(self.active_cluster_id) >= 2 and self.active_articulation_joint is None and len(self.geometry_parts) >= 1:
            mouse_state = (int(event.x), int(event.y))
            ans, rays = self.do_raycast([self.geometry_parts[part_dict["cluster_id"]] for part_dict in self.active_cluster_id], mouse_state)
            rayori = rays[0][:3]
            raydir = rays[0][3:]
            t_hit = ans['t_hit'].numpy()[0]
            primitive_normal = ans['primitive_normals'].numpy()[0]
            if not np.isinf(t_hit):
                hitted_point = rayori + raydir * t_hit
                self.active_articulation_joint = [hitted_point, primitive_normal, copy.deepcopy(self.active_cluster_id), None, None]
                def on_image(image):
                    self.active_articulation_joint[3] = image
                    self.active_articulation_joint[4] = (self.slider_low.double_value, self.slider_up.double_value)
                    self._image.update_image(image)
                    self.redraw_anything()
                arrow = get_arrow_from_ori_and_dir(hitted_point, primitive_normal)
                vertex_colors = np.zeros_like(np.asarray(arrow.vertices))
                vertex_colors[:, 0] = 1.0
                arrow.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                arrow.compute_vertex_normals()
                self._scene.scene.add_geometry(f"__model__.articulation", arrow, self.settings.material)
                self.slider_low.enabled = True
                self.slider_up.enabled = True
                self.slider_low.double_value = 0
                self.slider_up.double_value = 90
                color_palette = seaborn.color_palette(n_colors=5)
                cluster_id = self.active_cluster_id[1]["cluster_id"]
                part = self.geometry_parts[cluster_id]
                vertex_colors = np.asarray(part.vertex_colors).copy()
                vertex_colors[:] = color_palette[-1]
                new_part1 = copy.deepcopy(part)
                new_part1.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                new_part2 = copy.deepcopy(new_part1)
                vertex_colors[:] = color_palette[-2]
                new_part2.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                self._scene.scene.show_geometry(f"__model__.{cluster_id}", False)
                self._scene.scene.add_geometry(f"__model__.{cluster_id}.lower", new_part1, self.settings.material)
                self._scene.scene.add_geometry(f"__model__.{cluster_id}.upper", new_part2, self.settings.material)
                ori_transform = copy.deepcopy(self._scene.scene.get_geometry_transform(f"__model__.{cluster_id}"))
                child_transform_1 = RotateAnyAxis(self.active_articulation_joint[0], self.active_articulation_joint[0]+self.active_articulation_joint[1], np.pi * self.slider_low.double_value / 180)
                child_transform_2 = RotateAnyAxis(self.active_articulation_joint[0], self.active_articulation_joint[0]+self.active_articulation_joint[1], np.pi * self.slider_up.double_value / 180)
                self._scene.scene.set_geometry_transform(f"__model__.{cluster_id}.lower", child_transform_1 @ ori_transform)
                self._scene.scene.set_geometry_transform(f"__model__.{cluster_id}.upper", child_transform_2 @ ori_transform)
                self._scene.scene.scene.render_to_image(on_image)
                self._scene.visible = True
                self._image.visible = False
                self.redraw_anything()
    def _on_keyboard_event(self, event):
        if event.type == event.Type.UP:
            if event.key == 4294967295:
                return True
            key = chr(event.key).lower()
            if key == str(Settings.FREE_MODE):
                self.set_free_mode()
            elif key == str(Settings.SEG_MODE):
                self.set_seg_mode()
            elif key == str(Settings.PART_MODE):
                self.set_part_mode()
            elif key == str(Settings.JOINT_MODE) and not self.part_only_mode:
                self.set_joint_mode()
            elif key == str(Settings.SEG_POS_LABEL) and self.use_sam and self.settings.ui_mode == Settings.SEG_MODE:
                self.change_seg_label_mode(is_pos=True)
            elif key == str(Settings.SEG_NEG_LABEL) and self.use_sam and self.settings.ui_mode == Settings.SEG_MODE:
                self.change_seg_label_mode(is_pos=False)
            elif key == str(Settings.RECOVER):
                self.do_recover()
            elif key == str(Settings.SET_PART) and self.settings.ui_mode == Settings.PART_MODE:
                self.set_part()
            elif key == str(Settings.SET_PART) and self.settings.ui_mode == Settings.JOINT_MODE:
                self.set_joint()
            elif key in [str(x) for x in \
                                [Settings.JOINT_X_UP, \
                                 Settings.JOINT_X_DOWN, \
                                 Settings.JOINT_Y_UP, \
                                 Settings.JOINT_Y_DOWN, \
                                 Settings.JOINT_Z_UP, \
                                 Settings.JOINT_Z_DOWN, \
                                 Settings.JOINT_X_ROT_POS, \
                                 Settings.JOINT_X_ROT_NEG, \
                                 Settings.JOINT_Y_ROT_POS, \
                                 Settings.JOINT_Y_ROT_NEG, \
                                 Settings.JOINT_Z_ROT_POS, \
                                 Settings.JOINT_Z_ROT_NEG]]:
                self.move_joint(key)
            elif key == str(Settings.SAVE_ALL):
                self.save_all()
            elif key == str(Settings.ANIMATE) and self.settings.ui_mode == Settings.JOINT_MODE \
                                              and self.active_cluster_id is None \
                                              and self.active_articulation_joint is None:
                self.animate()
            return True
        else:
            return False

    def _on_mouse_event(self, event):
        if event.buttons is None:
            return gui.Widget.EventCallbackResult.HANDLED
        else:
            if event.buttons == o3d.visualization.gui.MouseButton.LEFT.value:
                if self.settings.ui_mode == Settings.SEG_MODE and event.type in [
                    o3d.visualization.gui.MouseEvent.Type.BUTTON_UP, \
                    o3d.visualization.gui.MouseEvent.Type.BUTTON_DOWN, \
                    o3d.visualization.gui.MouseEvent.Type.DRAG
                ]:
                    if self.use_sam:
                        self.parse_mouse_sam(event)
                    else:
                        self.parse_mouse_split(event)
                    return gui.Widget.EventCallbackResult.CONSUMED
                elif self.settings.ui_mode == Settings.PART_MODE and event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
                    self.parse_mouse_part_select(event)
                    return gui.Widget.EventCallbackResult.CONSUMED
                elif self.settings.ui_mode == Settings.JOINT_MODE and event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP:
                    if not (isinstance(self.active_cluster_id, list) and len(self.active_cluster_id) >= 2):
                        self.parse_mouse_part_select_for_joint(event)
                        return gui.Widget.EventCallbackResult.CONSUMED
                    elif self.active_articulation_joint is None:
                        self.parse_mouse_build_joint(event)
                        return gui.Widget.EventCallbackResult.CONSUMED
                    else:
                        return gui.Widget.EventCallbackResult.HANDLED
                else:
                    return gui.Widget.EventCallbackResult.HANDLED
            elif event.buttons == o3d.visualization.gui.MouseButton.RIGHT.value:
                if self.settings.ui_mode == Settings.PART_MODE and event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP: 
                    self.deselect_part()
                    return gui.Widget.EventCallbackResult.CONSUMED
                elif self.settings.ui_mode == Settings.JOINT_MODE and event.type == o3d.visualization.gui.MouseEvent.Type.BUTTON_UP \
                                                                  and isinstance(self.active_cluster_id, list) \
                                                                  and len(self.active_cluster_id) >= 1: 
                    self.deselect_part_for_joint()
                    return gui.Widget.EventCallbackResult.CONSUMED
                else:
                    return gui.Widget.EventCallbackResult.CONSUMED
            else:
                return gui.Widget.EventCallbackResult.HANDLED
    
    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        width = 17 * layout_context.theme.font_size
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, r.height)
        self._scene.frame = gui.Rect(r.x, r.y, r.width - width, r.height)
        self._image.frame = gui.Rect(r.x, r.y, r.width - width, r.height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)
        pyautogui.moveRel(1, 0)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_recover_part(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("Recover Part")

        # Add the text
        dlg_layout = gui.Vert(0, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Forgot the following part:"))
        _image = gui.ImageWidget()
        _image.update_image(self.img_parts[-1])
        dlg_layout.add_child(_image)
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        yes = gui.Button("Yes")
        yes.set_on_clicked(self.do_forget_part)
        no = gui.Button("No")
        no.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(yes)
        h.add_child(no)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
        pyautogui.moveRel(1, 0)

    def _on_recover_joint(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("Recover Joint")

        # Add the text
        dlg_layout = gui.Vert(0, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Forgot the following joint:"))
        _image = gui.ImageWidget()
        _image.update_image(self.articulation_joints[-1][3])
        dlg_layout.add_child(_image)
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        yes = gui.Button("Yes")
        yes.set_on_clicked(self.do_forget_joint)
        no = gui.Button("No")
        no.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(yes)
        h.add_child(no)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
        pyautogui.moveRel(1, 0)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        # gui.Application.instance.menubar.set_checked(
        #     AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("Help")

        # Add the text
        dlg_layout = gui.Vert(0, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Object Part/Articulation Annotation Tool"))
        dlg_layout.add_child(gui.Label("Author: Hongwei Fan, PKU-Agibot Lab"))
        dlg_layout.add_child(gui.Label("=================== Keyboard ==================="))
        dlg_layout.add_child(gui.Label(f"Press {Settings.FREE_MODE}: Viewpoint mode, viewpoint adjustment"))
        dlg_layout.add_child(gui.Label(f"Press {Settings.SEG_MODE}: Split mode, splitting the mesh to parts"))
        dlg_layout.add_child(gui.Label(f"Press {Settings.PART_MODE}: Part mode, selecting which part to save"))
        if not self.part_only_mode:
            dlg_layout.add_child(gui.Label(f"Press {Settings.JOINT_MODE}: Joint mode, add a joint for two parts"))
            dlg_layout.add_child(gui.Label(f"Press {Settings.RECOVER}: cancel seg, forgot the logged part / articulation"))
        else:
            dlg_layout.add_child(gui.Label(f"Press {Settings.RECOVER}: cancel seg, forgot the logged part"))
        dlg_layout.add_child(gui.Label(f"Press {Settings.SET_PART}: log the selected part"))
        dlg_layout.add_child(gui.Label(f"Press {Settings.SAVE_ALL}: save all info to the disk"))
        dlg_layout.add_child(gui.Label(f"Press {Settings.ANIMATE}: animate all joints!"))
        dlg_layout.add_child(gui.Label("===================== Mouse ===================="))
        dlg_layout.add_child(gui.Label("Viewpoint mode: drag the viewpoint to segment"))
        dlg_layout.add_child(gui.Label("Split mode: drag the line, segment the mesh"))
        dlg_layout.add_child(gui.Label("Part mode: select one part, log it in memory"))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
        pyautogui.moveRel(1, 0)

    def _on_menu_save(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("Saved")

        # Add the text
        dlg_layout = gui.Vert(0, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("All parts saved!"))
        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
        pyautogui.moveRel(1, 0)

    def _on_about_ok(self):
        self.window.close_dialog()
        pyautogui.moveRel(1, 0)
        if self._scene.visible:
            self.window.set_focus_widget(self._scene)
        if self._image.visible:
            self.window.set_focus_widget(self._image)

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
        if mesh is None:
            print("[Info]", path, "appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                print("[Info] Successfully read", path)
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                print("[WARNING] Failed to read points", path)

        if geometry is not None or mesh is not None:
            try:
                
                if mesh is not None:
                    # Triangle model
                    self._scene.scene.add_model("__model__", mesh)
                    self.geometry_dict = [mesh.meshes[0].mesh]
                    self.path = path
                    self._scene.scene.update_material(self.settings.material)
                else:
                    # Point cloud
                    self._scene.scene.add_geometry("__model__", geometry,
                                                   self.settings.material)
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)


def main(args):
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768, args.use_sam, args.part_only_mode)
    if args.load_file is not None and os.path.exists(args.load_file):
        w.load(args.load_file)
        
    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-sam', action='store_true', default=False)
    parser.add_argument('--part-only-mode', action='store_true', default=False)
    parser.add_argument('--load-file', type=str, default=None)
    args = parser.parse_args()
    main(args)