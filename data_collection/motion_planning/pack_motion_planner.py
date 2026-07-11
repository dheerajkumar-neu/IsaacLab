# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
PackMotionPlanner — high-level pick-and-place orchestration for the packing task.

Wraps CuroboPlanner to provide two operations per object:
  1. plan_pick(grasp)     — plan to the GraspGenX grasp pose (fingers open)
  2. plan_place(obj_name) — plan to above the bin (fingers closed, object attached)

Waypoint execution is left to the caller (collect_packing_demos.py) so that
data recording can interleave with sim stepping.
"""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

from data_collection.grasp_client.grasp_result import GraspResult


# Gripper pointing straight down: 180° rotation around X in wxyz convention.
_DOWN_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)

# How far above the bin origin to target for placement (metres).
PLACE_HEIGHT_OFFSET: float = 0.40

# Extra push along the grasp approach axis, applied to every GraspGenX hand
# pose before planning (metres). Raw grasps can land shallow — fingers close
# on the object's edge rather than around its body, which slips on sideways
# or curved-surface grasps. Increase to bite deeper; trial-and-error knob.
GRASP_DEPTH_OFFSET: float = 0.01


class PackMotionPlanner:
    """Thin orchestration layer connecting GraspGenX output to CuRobo for the packing env.

    The planner is initialised once per run.  For each object in the episode:
      - call plan_pick()  to plan the approach to the grasp pose
      - execute waypoints via get_arm_waypoints()
      - call plan_place() to plan the carry-and-drop above the bin
      - execute waypoints via get_arm_waypoints()
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        env_id: int = 0,
    ) -> None:
        self.env = env
        self.robot = robot
        self.env_id = env_id

        cfg = CuroboPlannerCfg.franka_pack_object_bin_config()
        # Sample the interpolated plan at exactly one waypoint per env control step so that
        # executing one waypoint per env.step() reproduces cuRobo's planned velocity profile.
        # (env.step_dt = sim.dt * decimation, e.g. 1/24 s for the packing env.)
        # The raw trajopt optimized_plan (legacy default) carries its own variable time base
        # that a fixed-rate executor does not respect — that mismatch shows up as jitter.
        cfg.interpolation_dt = float(env.step_dt)
        cfg.use_interpolated_plan = True
        # Home-return motions use joint-space planning (plan_single_js); without this
        # warmup the js trajopt solver is uninitialized and home planning fails.
        cfg.warmup_js_trajopt = True
        # franka_config defaults to a single attempt — grasp poses near the table are
        # marginal and often need a retry with a different seed before giving up.
        cfg.max_planning_attempts = 4
        self.planner = CuroboPlanner(
            env=env,
            robot=robot,
            config=cfg,
            env_id=env_id,
        )

    # ------------------------------------------------------------------
    # Planning interface
    # ------------------------------------------------------------------

    def plan_pick(
        self,
        grasp: GraspResult,
        camera_to_world: torch.Tensor | None = None,
    ) -> bool:
        """Plan collision-free motion to the GraspGenX grasp pose.

        Args:
            grasp:           Filtered GraspResult to reach.
            camera_to_world: Optional (4, 4) tensor transforming from camera to
                             world frame.  Pass None if grasp is already in world
                             (env-local) frame.

        Returns:
            True if CuRobo found a valid trajectory.
        """
        target_pose = grasp.to_world_pose(camera_to_world, depth_offset=GRASP_DEPTH_OFFSET)
        return self.planner.update_world_and_plan_motion(
            target_pose=target_pose,
            expected_attached_object=None,
            env_id=self.env_id,
        )

    def plan_place(self, object_name: str) -> bool:
        """Plan collision-free motion to place the grasped object above the bin.

        The target pose is PLACE_HEIGHT_OFFSET above the bin's current world
        position, with the gripper facing straight down.

        Args:
            object_name: Isaac Lab scene name of the attached object,
                         e.g. "object_01".  Passed to CuRobo so it includes
                         the object's collision geometry when planning.

        Returns:
            True if CuRobo found a valid trajectory.
        """
        bin_obj = self.env.scene["packing_bin"]
        env_origin = self.env.scene.env_origins[self.env_id]
        bin_pos = (bin_obj.data.root_pos_w[self.env_id] - env_origin).clone()

        place_pos = bin_pos.clone()
        place_pos[2] = place_pos[2] + PLACE_HEIGHT_OFFSET

        rot_mat = math_utils.matrix_from_quat(
            _DOWN_QUAT.to(device=bin_pos.device).unsqueeze(0)
        ).squeeze(0)

        target_pose = torch.eye(4, device=bin_pos.device, dtype=torch.float32)
        target_pose[:3, :3] = rot_mat
        target_pose[:3, 3] = place_pos

        return self.planner.update_world_and_plan_motion(
            target_pose=target_pose,
            expected_attached_object=object_name,
            env_id=self.env_id,
        )

    def plan_home(self, home_joint_pos: torch.Tensor, joint_names: list[str]) -> bool:
        """Plan a joint-space motion back to the robot's home configuration.

        Called after each object is placed so every pick starts from the same known
        configuration (and the arm is out of the cameras' view of the workspace).

        Args:
            home_joint_pos: Target joint positions (may include finger joints; extras
                            beyond cuRobo's actuated joints are ignored).
            joint_names:    Joint names corresponding to ``home_joint_pos``.

        Returns:
            True if CuRobo found a valid trajectory.
        """
        return self.planner.plan_to_joint_target(home_joint_pos, joint_names)

    # ------------------------------------------------------------------
    # Waypoint access
    # ------------------------------------------------------------------

    def get_arm_waypoints(self, arm_joint_names: list[str] | None = None) -> list[torch.Tensor]:
        """Return the arm joint-position waypoints of the current plan.

        Args:
            arm_joint_names: If given, select exactly these joints (by name) from each
                             waypoint, in the given order. Otherwise the raw waypoints
                             are returned (all plan joints, robot joint order).

        Returns:
            List of (num_arm_joints,) tensors on the cuRobo device.
            Empty if no plan is available.
        """
        plan = self.planner.current_plan
        if plan is None or len(plan.position) == 0:
            return []
        if arm_joint_names is not None:
            arm_idx = [plan.joint_names.index(name) for name in arm_joint_names]
            return [plan.position[i][arm_idx].detach() for i in range(len(plan.position))]
        return [plan.position[i].detach() for i in range(len(plan.position))]

    def get_waypoint_ee_positions(self) -> list[torch.Tensor] | None:
        """Compute the commanded end-effector position for every waypoint via cuRobo FK.

        Used purely for logging/debugging: gives the EE trajectory the plan commands,
        which can be compared step-by-step against the measured EE position.

        Returns:
            List of (3,) CPU tensors (env-local frame, same frame the targets were
            given in), or None if FK fails.
        """
        plan = self.planner.current_plan
        if plan is None or len(plan.position) == 0:
            return None
        try:
            ee_positions: list[torch.Tensor] = []
            for i in range(len(plan.position)):
                kin = self.planner.motion_gen.compute_kinematics(plan[i])
                ee_pos = kin.ee_position if hasattr(kin, "ee_position") else kin.ee_pose.position
                ee_positions.append(ee_pos.reshape(-1)[:3].detach().cpu())
            return ee_positions
        except Exception:
            return None

    def reset_plan(self) -> None:
        """Clear the current plan (delegates to CuroboPlanner)."""
        self.planner.reset_plan()
