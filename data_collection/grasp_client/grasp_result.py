# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
GraspResult — abstract representation of a single grasp candidate from GraspGenX.

GraspGenX returns a list of grasps per request.  Each grasp has a 6-DOF pose
(position + quaternion) and a confidence score.  The pose may be expressed in
the camera frame or the world frame depending on the server configuration;
to_world_pose() handles both cases via an optional camera-to-world transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import isaaclab.utils.math as math_utils


@dataclass
class GraspResult:
    """Single grasp candidate returned by GraspGenX.

    Attributes:
        position:   (3,) xyz in camera or world frame.
        quaternion: (4,) wxyz in camera or world frame.
        confidence: Scalar score in [0, 1].  Higher is better.
        object_id:  Identifier of the target object (e.g. "object_01").
    """

    position: torch.Tensor
    quaternion: torch.Tensor
    confidence: float
    object_id: str

    def approach_axis(self) -> torch.Tensor:
        """Unit approach direction (grasp frame z-axis) in the pose's frame."""
        rot = math_utils.matrix_from_quat(self.quaternion.float().unsqueeze(0)).squeeze(0)
        return rot[:, 2]

    def downwardness(self) -> float:
        """1.0 = approaching straight down, 0.0 = horizontal, -1.0 = from below.

        Useful for workspaces where objects sit inside walled containers: side
        approaches collide with the walls, so only sufficiently top-down grasps
        are plannable.
        """
        return float(-self.approach_axis()[2])

    def tcp_position(self, fingertip_offset: float = 0.1034) -> torch.Tensor:
        """Fingertip-midpoint (TCP) position of this grasp, in the same frame.

        GraspGenX poses locate the panda_hand BASE frame; the fingertips sit
        ``fingertip_offset`` further along the grasp z-axis (approach direction).
        A good grasp therefore has its *position* ~0.10-0.17 m from the object
        centre while its TCP is on the object — use this point (not ``position``)
        for object-proximity checks.
        """
        rot = math_utils.matrix_from_quat(self.quaternion.float().unsqueeze(0)).squeeze(0)  # (3, 3)
        return self.position.float() + rot[:, 2] * fingertip_offset

    def to_world_pose(self, camera_to_world: torch.Tensor | None = None) -> torch.Tensor:
        """Convert grasp to a 4x4 homogeneous transform in world (env-local) frame.

        Args:
            camera_to_world: Optional (4, 4) float32 tensor that maps points from
                             the camera frame to the world frame.  If None, the
                             grasp is assumed to already be in world frame.

        Returns:
            (4, 4) float32 tensor representing the gripper target pose in world
            frame, ready to be passed to CuroboPlanner.plan_motion().
        """
        pos = self.position.float()
        quat = self.quaternion.float()  # wxyz

        rot_mat = math_utils.matrix_from_quat(quat.unsqueeze(0)).squeeze(0)  # (3, 3)

        T = torch.eye(4, device=pos.device, dtype=torch.float32)
        T[:3, :3] = rot_mat
        T[:3, 3] = pos

        if camera_to_world is not None:
            T = camera_to_world.to(device=pos.device, dtype=torch.float32) @ T

        return T


def filter_grasps(
    grasps: list[GraspResult],
    confidence_threshold: float = 0.5,
    top_k: int = 1,
) -> list[GraspResult]:
    """Filter and rank grasp candidates.

    Applies a confidence threshold, then returns the top-K remaining grasps
    sorted by descending confidence.  If fewer than top_k grasps survive the
    threshold, all surviving grasps are returned.

    Args:
        grasps:               Raw list of GraspResult from GraspGenX.
        confidence_threshold: Minimum confidence to keep a grasp.
        top_k:                Maximum number of grasps to return.

    Returns:
        Filtered and sorted list of GraspResult (best first).
    """
    surviving = [g for g in grasps if g.confidence >= confidence_threshold]
    surviving.sort(key=lambda g: g.confidence, reverse=True)
    return surviving[:top_k]
