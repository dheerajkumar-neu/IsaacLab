# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.isaaclab_int import mdp
from isaaclab_tasks.manager_based.isaaclab_int.mdp import franka_pack_events
from isaaclab_tasks.manager_based.isaaclab_int.packing_env_cfg import IsaaclabIntEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# Local candidate-object assets (see repo `assets/` folder). Repo root is 7 levels above this file.
_ISAACLAB_ASSETS_DIR = Path(__file__).resolve().parents[7] / "assets"


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_pack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_pack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Packing bin stays at a fixed position
    reset_packing_bin_pose = EventTerm(
        func=franka_pack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (-0.5, -0.5), "z": (0.1, 0.1), "yaw": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0)},
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("packing_bin")],
        },
    )

    # Objects are randomized on the table surface
    reset_objects_pose = EventTerm(
        func=franka_pack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.2), "y": (0.45, 0.75), "z": (0.2, 0.2), "yaw": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [
                SceneEntityCfg("object_01"),
                SceneEntityCfg("object_02"),
                SceneEntityCfg("object_03"),
            ],
        },
    )


@configclass
class FrankaPackEnvCfg(IsaaclabIntEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set actions for joint position control
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Gripper state tracking attributes consumed by mdp observation/termination functions
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Shared rigid body properties for all packed objects
        object_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Packing bin — kinematic so it stays fixed during the episode
        self.scene.packing_bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PackingBin",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
                scale=(2.5, 2.5, 1.5),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
                mass_props=MassPropertiesCfg(mass=1.0),
                collision_props=CollisionPropertiesCfg(),
            ),
        )

        # Object 1 — stainless-steel spatula/ladle (local SimReady asset)
        # Physics (rigid body, per-part convex-hull colliders, mass=0.18kg, friction/restitution
        # material) is already authored on this asset, so no collision_props/mass_props override
        # is needed here — same as objects 02/03 below.
        self.scene.object_01 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object01",
            # Identity rotation = upright/standing, matching the orientation the
            # reset_objects_pose event samples (roll/pitch/yaw all zero) so fixed
            # and randomized placements agree.
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.8, 0.20), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                usd_path=str(
                    _ISAACLAB_ASSETS_DIR
                    / "N00327 Sterilisingcontainer System-IsaacSim"
                    / "IsaacSim_asset_69e7433d7d5208d3e791ee70.usd"
                ),
                scale=(1.0, 1.0, 1.0),
                rigid_props=object_properties,
                semantic_tags=[("class", "object_01")],
            ),
        )

        # Object 2 — mustard bottle (YCB physics-enabled)
        self.scene.object_02 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object02",
            # Identity rotation = upright/standing (see object_01 note).
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.5, 0.20), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
                usd_path=str(
                    _ISAACLAB_ASSETS_DIR
                    / "N02017 Whitepackerbottle-IsaacSim"
                    / "IsaacSim_asset_69eca8e8c917ed99cd3ad807.usd"
                ),
                scale=(1.0, 1.0, 1.0),
                rigid_props=object_properties,
                semantic_tags=[("class", "object_02")],
            ),
        )

        # Object 3 — sugar box (YCB physics-enabled)
        self.scene.object_03 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object03",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.7, 0.10), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=UsdFileCfg(
                # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
                usd_path=str(
                    _ISAACLAB_ASSETS_DIR
                    / "D00160 Coffee Mug-IsaacSim" 
                    / "IsaacSim_assets_69fef61ff3726e6ad05387d4" 
                    / "IsaacSim_asset_69fef61ff3726e6ad05387d4.usd"
                ),
                scale=(1.0, 1.0, 1.0),
                rigid_props=object_properties,
                semantic_tags=[("class", "object_03")],
            ),
        )

        # End-effector frame transformer — tracks panda_hand and both fingers
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )
