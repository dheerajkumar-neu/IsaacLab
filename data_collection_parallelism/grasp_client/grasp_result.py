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

import math
from dataclasses import dataclass

import torch
import isaaclab.utils.math as math_utils

# GraspGenX grasp poses are expressed in the gripper description's canonical
# frame, whose root ("world" link in ext/gripper_descriptions/.../franka_panda/
# gripper.urdf) is panda_hand rotated by +90 deg about z (world_joint has
# rpy="0 0 1.5708"): finger travel is along x in the canonical frame vs y in
# panda_hand. cuRobo targets the panda_hand link, so every GraspGenX rotation
# must be post-multiplied by this Rz(+90 deg) before use as a panda_hand pose —
# otherwise the fingers close across the wrong object axis. Translation and the
# z approach axis are unaffected.
_GRASPGEN_TO_PANDA_HAND_ROT = torch.tensor(
    [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]],
    dtype=torch.float32,
)


def graspgen_rot_to_panda_hand(rot: torch.Tensor) -> torch.Tensor:
    """Convert a GraspGenX canonical gripper-frame rotation to a panda_hand rotation.

    Args:
        rot: (3, 3) rotation matrix of the GraspGenX grasp pose.

    Returns:
        (3, 3) rotation matrix for the panda_hand target with the same approach
        axis and the finger axis corrected by 90 deg about z.
    """
    return rot @ _GRASPGEN_TO_PANDA_HAND_ROT.to(device=rot.device, dtype=rot.dtype)


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

    def to_world_pose(
        self, camera_to_world: torch.Tensor | None = None, depth_offset: float = 0.0
    ) -> torch.Tensor:
        """Convert grasp to a 4x4 homogeneous transform in world (env-local) frame.

        Args:
            camera_to_world: Optional (4, 4) float32 tensor that maps points from
                             the camera frame to the world frame.  If None, the
                             grasp is assumed to already be in world frame.
            depth_offset:    Extra push along the grasp approach axis (+z, metres)
                             applied to the raw GraspGenX hand position before
                             planning. GraspGenX poses sometimes land shallow —
                             fingers close near the object's edge instead of
                             around its body — which holds under gravity but
                             slips on sideways/curved grasps. Increase this to
                             move the commanded hand pose deeper onto the object.

        Returns:
            (4, 4) float32 tensor representing the gripper target pose in world
            frame, ready to be passed to CuroboPlanner.plan_motion().
        """
        pos = self.position.float()
        quat = self.quaternion.float()  # wxyz

        rot_mat = math_utils.matrix_from_quat(quat.unsqueeze(0)).squeeze(0)  # (3, 3)
        pos = pos + rot_mat[:, 2] * depth_offset

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
    object_pos: torch.Tensor | None = None,
    centroid_decay: float = 0.05,
) -> list[GraspResult]:
    """Filter and rank grasp candidates.

    Applies a confidence threshold, then ranks survivors by a composite score
    and returns the top-K.

    GraspGenX's confidence reflects finger-contact quality but is blind to
    whether the pose is reachable on this rig — it will confidently propose
    grasps approaching from the side, or even from underneath the table (see
    downwardness()), which cuRobo then fails to plan. Ranking by
    confidence * downwardness * centroid-closeness instead of raw confidence
    pushes those candidates to the bottom without hard-rejecting them, so an
    episode never stalls just because every candidate this round happens to
    be geometrically mediocre (a hard downwardness filter tried previously
    did exactly that and was reverted).

    Args:
        grasps:               Raw list of GraspResult from GraspGenX.
        confidence_threshold: Minimum confidence to keep a grasp.
        top_k:                Maximum number of grasps to return.
        object_pos:           Optional (3,) object centroid, same frame as the
                               grasps. When given, grasps whose TCP lands
                               closer to the centroid (a fuller grip, less
                               likely to slip off the edge) are preferred.
                               Omit to skip this term.
        centroid_decay:       Distance (metres) at which the centroid-distance
                               term decays to ~37% (1/e). Only used if
                               object_pos is given.

    Returns:
        Filtered and sorted list of GraspResult (best first).
    """
    surviving = [g for g in grasps if g.confidence >= confidence_threshold]

    def score(g: GraspResult) -> float:
        downwardness_term = max(g.downwardness(), 0.0)
        if object_pos is not None:
            dist = float(torch.linalg.vector_norm(g.tcp_position() - object_pos))
            centroid_term = math.exp(-dist / centroid_decay)
        else:
            centroid_term = 1.0
        return g.confidence * downwardness_term * centroid_term

    # Tie-break on raw confidence so a round where every candidate scores 0
    # (e.g. nothing clears downwardness > 0) still degrades to the original
    # confidence-only ranking instead of an arbitrary order.
    surviving.sort(key=lambda g: (score(g), g.confidence), reverse=True)
    return surviving[:top_k]
