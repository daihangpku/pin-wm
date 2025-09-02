# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``omni.isaac.urdf_importer``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links that are connected by fixed joints. (default: False)
  --fix-base                Fix the base to where it is imported. (default: False)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument("-i", "--input", type=str, help="The path to the input URDF file.")
parser.add_argument("-o", "--output", type=str, default=None, help="The path to store the USD file.")
parser.add_argument("-d", "--default-prim-name", type=str, default="link0")
parser.add_argument("-s", "--set-default-prim", action='store_true')
parser.add_argument("-sdf", "--use-sdf", action='store_true')
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import carb
import omni.isaac.core.utils.stage as stage_utils
from omni.physx.scripts import utils
import omni.kit.app
from pxr import Usd, UsdGeom, Gf
from pxr import PhysxSchema

from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict

import shutil

def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if dest_path is None:
        dest_path = urdf_path.replace(".urdf", ".usd")
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Create Urdf converter config
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        make_instanceable=False,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Urdf converter and import the file
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    # print output
    print("URDF importer output:")
    print(f"Generated USD file: {urdf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)
    
    texture_dir = os.path.join(os.path.dirname(urdf_converter.usd_path), "materials")
    for texture_filename in os.listdir(texture_dir):
        texture_filepath = os.path.join(texture_dir, texture_filename)
        real_texture_filepath = os.path.abspath(os.path.join(texture_dir, "..", texture_filename))
        shutil.copy(real_texture_filepath, texture_filepath)
    
    stage_utils.open_stage(urdf_converter.usd_path)
    stage = omni.usd.get_context().get_stage()
    curr_prim = stage.GetPrimAtPath("/")
    for prim in Usd.PrimRange(curr_prim):
        if prim.IsA(UsdGeom.Mesh) and prim.GetName() == "collisions":
            if args_cli.use_sdf:
                # TODO: don't know why unstable for isaac lab, but work for isaac sim
                utils.setCollider(prim, approximationShape="sdf")
                meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                meshCollision.CreateSdfResolutionAttr().Set(256)
            else:
                utils.setCollider(prim, approximationShape="convexDecomposition")
    stage_utils.save_stage(urdf_converter.usd_path)
    
    if args_cli.set_default_prim:
        stage_utils.open_stage(urdf_converter.usd_path)
        current_stage = omni.usd.get_context().get_stage()
        default_prim = UsdGeom.Xform.Define(current_stage, Sdf.Path(f"/{args_cli.default_prim_name}")).GetPrim()
        current_stage.SetDefaultPrim(default_prim)
        stage_utils.save_stage(urdf_converter.usd_path)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
