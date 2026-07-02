# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def objects_packed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_1_cfg: SceneEntityCfg = SceneEntityCfg("object_1"),
    object_2_cfg: SceneEntityCfg = SceneEntityCfg("object_2"),
    object_3_cfg: SceneEntityCfg | None = SceneEntityCfg("object_3"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("packing_bin"),
    xy_threshold: float = 0.12,
    z_margin: float = 0.20,
    atol: float = 0.0001,
    rtol: float = 0.0001,
) -> torch.Tensor:
    """Return True when all objects are inside the packing bin and the gripper is open."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_1: RigidObject = env.scene[object_1_cfg.name]
    object_2: RigidObject = env.scene[object_2_cfg.name]
    packing_bin: RigidObject = env.scene[bin_cfg.name]

    bin_pos = packing_bin.data.root_pos_w

    def _in_bin(obj: RigidObject) -> torch.Tensor:
        pos = obj.data.root_pos_w
        xy_dist = torch.norm(pos[:, :2] - bin_pos[:, :2], dim=1)
        return torch.logical_and(xy_dist < xy_threshold, pos[:, 2] < (bin_pos[:, 2] + z_margin))

    packed = torch.logical_and(_in_bin(object_1), _in_bin(object_2))

    if object_3_cfg is not None:
        object_3: RigidObject = env.scene[object_3_cfg.name]
        packed = torch.logical_and(packed, _in_bin(object_3))

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        packed = torch.logical_and(suction_cup_is_open, packed)
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
            packed = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                packed,
            )
            packed = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                packed,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return packed


def all_objects_packed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obj_01_cfg: SceneEntityCfg = SceneEntityCfg("object_01"),
    obj_02_cfg: SceneEntityCfg = SceneEntityCfg("object_02"),
    obj_03_cfg: SceneEntityCfg = SceneEntityCfg("object_03"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("packing_bin"),
    xy_threshold: float = 0.12,
    z_margin: float = 0.20,
    atol: float = 0.0001,
    rtol: float = 0.0001,
) -> torch.Tensor:
    """Return True when all three objects are inside the packing bin and the gripper is open."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj_01: RigidObject = env.scene[obj_01_cfg.name]
    obj_02: RigidObject = env.scene[obj_02_cfg.name]
    obj_03: RigidObject = env.scene[obj_03_cfg.name]
    packing_bin: RigidObject = env.scene[bin_cfg.name]

    bin_pos = packing_bin.data.root_pos_w

    def _in_bin(obj: RigidObject) -> torch.Tensor:
        pos = obj.data.root_pos_w
        xy_dist = torch.norm(pos[:, :2] - bin_pos[:, :2], dim=1)
        return torch.logical_and(xy_dist < xy_threshold, pos[:, 2] < (bin_pos[:, 2] + z_margin))

    packed = torch.logical_and(torch.logical_and(_in_bin(obj_01), _in_bin(obj_02)), _in_bin(obj_03))

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        packed = torch.logical_and(suction_cup_is_open, packed)
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
            packed = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                packed,
            )
            packed = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                packed,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return packed
