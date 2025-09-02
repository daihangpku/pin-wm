import numpy as np
import argparse
import os

from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.world import World

import sys
sys.path.insert(0, os.path.dirname(__file__))
from constants import PANDA_DEFAULT_CFG

def usd_fk_transforms(usd_path, root_prim_path, cfg=None):
    assert "franka" in usd_path, "Currently we don't support other robots."
    if cfg is None:
        cfg = PANDA_DEFAULT_CFG
    if World.instance():
        World.instance().clear_instance()
    world = World()
    add_reference_to_stage(usd_path, root_prim_path)
    articulation_prim_view = ArticulationView(prim_paths_expr=root_prim_path, name="robot")
    world.scene.add(articulation_prim_view)
    world.reset()
    dof_names = articulation_prim_view.dof_names
    dof_positions = np.expand_dims(np.array([cfg[dof_name] if dof_name in cfg else 0.0 for dof_name in dof_names]), axis=0)
    articulation_prim_view.set_joint_positions(dof_positions)
    link_transforms = articulation_prim_view._physics_view.get_link_transforms()[0].copy()
    link_names = articulation_prim_view.body_names
    np.savez_compressed(os.path.join(os.path.dirname(usd_path), "link_transforms.npz"), 
                        link_transforms=link_transforms,
                        link_paths=articulation_prim_view._physics_view.link_paths[0],
                        link_names=link_names)
    simulation_app.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="assets/franka/franka.usd",
                        help='Input USD File')
    parser.add_argument("--root-prim-path", type=str, default="/panda")

    args = parser.parse_args()
    usd_fk_transforms(args.input, args.root_prim_path)
