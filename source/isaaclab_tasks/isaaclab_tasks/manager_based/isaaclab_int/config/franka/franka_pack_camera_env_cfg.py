# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Packing environment variants equipped with three RealSense D435-like cameras for dataset recording.

Camera layout (all per-environment):
  - wrist_cam    : mounted on panda_hand, looks at the grasped object
  - table_top_cam: fixed above the workspace, straight-down bird's-eye view
  - front_cam    : fixed in front of the robot, tilted-front overview (~30° below horizontal)

RealSense D435 RGB module approximate intrinsics (640×480):
  focal_length=24.0 mm, horizontal_aperture≈33.0 mm  →  HFOV≈69°
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.isaaclab_int import mdp

from . import franka_pack_joint_pos_env_cfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


##
# Observation configuration
##


@configclass
class ObservationsCfg:
    """Observation specifications for the camera-equipped packing environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """State-based observations (identical to base packing env)."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_positions = ObsTerm(func=mdp.object_positions_in_world_frame)
        object_orientations = ObsTerm(func=mdp.object_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """RGB and depth image observations from all three cameras."""

        # Wrist camera
        wrist_cam_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
        )
        # Table top camera
        table_top_cam_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_top_cam"), "data_type": "rgb", "normalize": False},
        )
        table_top_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_top_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
        )
        # Front overview camera
        front_cam_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False},
        )
        front_cam_depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_cam"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask-tracking observations (identical to base packing env)."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object_01"),
            },
        )
        place_1 = ObsTerm(
            func=mdp.object_packed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object_01"),
                "bin_cfg": SceneEntityCfg("packing_bin"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object_02"),
            },
        )
        place_2 = ObsTerm(
            func=mdp.object_packed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object_02"),
                "bin_cfg": SceneEntityCfg("packing_bin"),
            },
        )
        grasp_3 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object_03"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


##
# Environment configurations
##


@configclass
class FrankaPackCameraEnvCfg(franka_pack_joint_pos_env_cfg.FrankaPackEnvCfg):
    """Packing env with joint position control and three RealSense D435-like cameras."""

    observations: ObservationsCfg = ObservationsCfg()

    # Scene camera names exposed to dataset-recording scripts
    image_obs_list = ["wrist_cam", "table_top_cam", "front_cam"]

    def __post_init__(self):
        super().__post_init__()

        # Stiffer PD gains: dense joint-position waypoints from the motion planner need
        # tight tracking — the default (soft) gains lag and oscillate on large steps.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # ------------------------------------------------------------------ #
        # Camera 1 — Wrist camera (per env, attached to panda_hand)
        #
        # Mimics an Intel RealSense D435 mounted on the robot wrist.
        # Positioned 13 cm forward and 15 cm below the hand frame so the
        # fingertips and grasped object are centred in the field of view.
        #
        # Rotation quaternion (-0.70614, 0.03701, 0.03701, -0.70614) in ROS
        # convention tilts the camera ~90° downward from the hand z-axis,
        # pointing toward the workspace.
        # ------------------------------------------------------------------ #
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=33.0,  # RealSense D435 RGB HFOV ≈ 69°
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, -0.09),
                rot=(-0.707, 0.0, 0.0, -0.80),
                convention="ros",
            ),
        )

        # ------------------------------------------------------------------ #
        # Camera 2 — Table top camera (per env, fixed above the workspace)
        #
        # Centred at (x=0.1, y=0.1, z=1.2) in env-local coordinates so that
        # at height 1.2 m with a 69° HFOV the entire workspace is visible:
        #   x ∈ [-0.73, 0.93] m   (covers robot and both side margins)
        #   y ∈ [-0.52, 0.72] m   (covers bin at y=-0.5 and objects at y≈0.6)
        #
        # Rotation (0, 0, 1, 0) = 180° around world Y → camera z-axis points
        # world -Z (straight down).  Image rows run along world +Y.
        # ------------------------------------------------------------------ #
        self.scene.table_top_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_top_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=33.0,  # RealSense D435 RGB HFOV ≈ 69°
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.1, 0.6, 1.2),
                rot=(0.0, 0.0, 1.0, 0.0),  # 180° around Y → camera_z = world -Z (looking down)
                convention="ros",
            ),
        )

        # ------------------------------------------------------------------ #
        # Camera 3 — Front overview camera (per env, tilted front view)
        #
        # Positioned at (0, -1.0, 1.0) — 1 m in front of the robot and 1 m
        # above the table — looking toward the workspace.
        #
        # Rotation (0.5, -0.866, 0, 0) = -120° around X in ROS convention:
        #   camera_z (optical axis) = (0, +0.866, -0.5) in world space
        #   → looking in the +Y direction, 30° below the horizontal plane.
        # At 1 m range the centre of the frame lands on the robot wrist;
        # at 2 m it covers the full object workspace.
        # ------------------------------------------------------------------ #
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=33.0,  # RealSense D435 RGB HFOV ≈ 69°
                clipping_range=(0.1, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.2, 0.0, 1.0),
                rot=(-0.35355, 0.61237, 0.61237, -0.35355),  # GUI: x=0°, y=60°, z=90°
                convention="ros",
            ),
        )

        # Re-render on reset so cameras capture the freshly reset scene
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"


@configclass
class FrankaPackIKRelCameraEnvCfg(FrankaPackCameraEnvCfg):
    """Packing env with IK relative pose control and three RealSense D435-like cameras."""

    def __post_init__(self):
        super().__post_init__()

        # Switch to a stiffer PD controller for better IK tracking accuracy
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Replace joint-position arm action with IK-relative delta-pose control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )
