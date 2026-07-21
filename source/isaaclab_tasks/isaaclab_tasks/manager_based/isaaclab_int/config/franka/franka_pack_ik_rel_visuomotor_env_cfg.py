# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Packing environment variant with wrist/table-top RGB placed inside the `policy`
observation group, so Isaac Lab Mimic's standard recorder (which only captures
``obs_buf["policy"]``) records and replays image observations end-to-end.

Deliberately a separate, minimal-camera env cfg rather than reusing
``FrankaPackIKRelCameraEnvCfg`` (8 GraspGenX-fusion cameras): those all render every
frame regardless of which obs group they're wired into, which is unnecessarily
expensive across the many parallel envs used during Mimic dataset generation. Mirrors
``stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg`` (two cameras only).
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.isaaclab_int import mdp

from . import franka_pack_ik_rel_env_cfg


@configclass
class ObservationsCfg:
    """Observation specifications for the visuomotor (Mimic-recordable) packing env."""

    @configclass
    class PolicyCfg(ObsGroup):
        """State + image observations, kept ungrouped (dict) so Mimic's recorder
        (which only reads ``obs_buf["policy"]``) captures the RGB terms too."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_positions = ObsTerm(func=mdp.object_positions_in_world_frame)
        object_orientations = ObsTerm(func=mdp.object_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        table_top_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_top_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask-tracking observations (identical to the base packing env)."""

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
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class FrankaPackIKRelVisuomotorEnvCfg(franka_pack_ik_rel_env_cfg.FrankaPackEnvCfg):
    """Packing env: IK-relative control + wrist/table-top RGB inside the policy obs
    group, so Isaac Lab Mimic can record and replay image observations end-to-end."""

    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Wrist camera — mounted on panda_hand, looks at the grasped object.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
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

        # Table-top camera — fixed above the workspace, straight-down bird's-eye view.
        self.scene.table_top_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_top_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=33.0,
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.1, 0.6, 1.2),
                rot=(0.0, 0.0, 1.0, 0.0),  # 180° around Y -> camera_z = world -Z (looking down)
                convention="ros",
            ),
        )

        # Re-render on reset so cameras capture the freshly reset scene (Mimic
        # teleport-resets during data generation otherwise leave stale frames).
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"

        # Scene camera names exposed to dataset-recording scripts.
        self.image_obs_list = ["wrist_cam", "table_top_cam"]
