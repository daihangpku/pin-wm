# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a MJCF into USD format.

MuJoCo XML Format (MJCF) is an XML file format used in MuJoCo to describe all elements of a robot. For more information, see: http://www.mujoco.org/book/XMLreference.html

This script uses the MJCF importer extension from Isaac Sim (``omni.isaac.mjcf_importer``) to convert a MJCF asset into USD format. It is designed as a convenience script for command-line use. For more information on the MJCF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_mjcf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                Show this help message and exit
  --fix-base                Fix the base to where it is imported. (default: False)
  --import-sites            Import sites by parse <site> tag. (default: True)
  --make-instanceable       Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a MJCF into USD format.")
parser.add_argument("-i", "--input", type=str, help="The path to the input MJCF file.")
parser.add_argument("-o", "--output", type=str, help="The path to store the USD file.")
parser.add_argument("-d", "--default-prim-name", type=str, default="link0")
parser.add_argument("-s", "--set-default-prim", action='store_true')

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
import omni.kit.app

from omni.isaac.lab.sim.converters import MjcfConverter, MjcfConverterCfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict

from pxr import UsdGeom, Sdf, UsdPhysics

def main():
    # check valid file path
    mjcf_path = args_cli.input
    if not os.path.isabs(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    # create destination path
    dest_path = args_cli.output
    if dest_path is None:
        dest_path = urdf_path.replace(".mjcf", ".usd")
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # create the converter configuration
    mjcf_converter_cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=False,
        import_sites=False,
        force_usd_conversion=True,
        make_instanceable=False,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input MJCF file: {mjcf_path}")
    print("MJCF importer config:")
    print_dict(mjcf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create mjcf converter and import the file
    mjcf_converter = MjcfConverter(mjcf_converter_cfg)
    # print output
    print("MJCF importer output:")
    print(f"Generated USD file: {mjcf_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)

    if args_cli.set_default_prim:
        stage_utils.open_stage(mjcf_converter.usd_path)
        current_stage = omni.usd.get_context().get_stage()
        default_prim = UsdGeom.Xform.Define(current_stage, Sdf.Path(f"/{args_cli.default_prim_name}")).GetPrim()
        # NOTE: deprecated
        # joints = UsdGeom.Xform.Define(current_stage, Sdf.Path(f"/{args_cli.default_prim_name}/joints")).GetPrim()
        # for joint in joints.GetAllChildren():   
        #     attr_name_dict = dict()
        #     for attr in joint.GetAttributes():    
        #         attr_name = attr.GetName()
        #         if "drive:X" in attr_name:
        #             attr_name_dict[attr_name] = attr_name.replace("drive:X", "drive:angular")
        #     for old_attr_name, new_attr_name in attr_name_dict.items():
        #         new_attr = joint.CreateAttribute(new_attr_name, joint.GetAttribute(old_attr_name).GetTypeName())
        #         new_attr.Set(joint.GetAttribute(old_attr_name).Get())
        #     for applied_schema in joint.GetAppliedSchemas():
        #         if applied_schema == 'PhysicsDriveAPI:X':
        #             joint.RemoveAppliedSchema(applied_schema)
        #             joint.AddAppliedSchema(applied_schema.replace(":X", ":angular"))
        #         if applied_schema == 'PhysxLimitAPI:X':
        #             joint.RemoveAppliedSchema(applied_schema)
        #             joint.AddAppliedSchema(applied_schema.replace(":X", ":angular"))
        current_stage.SetDefaultPrim(default_prim)
        stage_utils.save_stage(mjcf_converter.usd_path)
    
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
