# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
GraspGenXDepthClient — queries the GraspGenX ZMQ server using live depth-camera
point clouds instead of USD-stage mesh vertices.

Pipeline per query()
--------------------
  1. Read raw depth image + intrinsics from the requested camera sensor.
  2. Back-project every valid pixel to a 3D point in camera frame.
  3. Transform points to world frame using the camera's live world pose.
  4. Filter to points within ``filter_radius`` metres of the target object centre.
  5. Optionally subsample to ``num_pc_points``.
  6. Subtract centroid (required by GraspGenX) and send to ZMQ server.
  7. Add centroid back, subtract env_origin, wrap each SE(3) grasp as GraspResult.

Usage::

    client = GraspGenXDepthClient(camera_name="table_top_cam", host="localhost")
    grasps = client.query("object_01", env, env_id=0)
    filtered = filter_grasps(grasps, confidence_threshold=0.5, top_k=1)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from data_collection.grasp_client.grasp_result import GraspResult

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_GRASPGENX_REPO = "/groot/data/ubuntu/GraspGenX"
_GRASPGENX_ZMQ_CLIENT_PATH = f"{_GRASPGENX_REPO}/graspgenx/serving/zmq_client.py"


def _load_graspgenx_client_class():
    if _GRASPGENX_REPO not in sys.path:
        sys.path.insert(0, _GRASPGENX_REPO)
    spec = importlib.util.spec_from_file_location(
        "graspgenx_zmq_client", _GRASPGENX_ZMQ_CLIENT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GraspGenXClient


# ---------------------------------------------------------------------------
# Depth → point cloud helpers
# ---------------------------------------------------------------------------

def _depth_to_pointcloud_camera_frame(
    depth_m: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Back-project a depth image to a point cloud in camera frame.

    Uses the standard pinhole model (ROS camera convention: x=right, y=down,
    z=forward / optical axis):

        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    Args:
        depth_m: (H, W) float32 depth in metres. Zero / inf pixels are skipped.
        K:       (3, 3) float32 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].

    Returns:
        (N, 3) float32 array of valid points in camera frame.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    H, W = depth_m.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    valid = (depth_m > 0.0) & np.isfinite(depth_m)
    d = depth_m[valid]

    x = (uu[valid] - cx) * d / fx
    y = (vv[valid] - cy) * d / fy
    z = d

    return np.stack([x, y, z], axis=1)  # (N, 3)


def _transform_camera_to_world(
    pts_cam: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w: np.ndarray,
) -> np.ndarray:
    """Rotate + translate points from camera frame to world frame.

    Args:
        pts_cam:    (N, 3) float32 in camera frame.
        cam_pos_w:  (3,) world position of the camera origin.
        cam_quat_w: (4,) wxyz quaternion: camera → world rotation.

    Returns:
        (N, 3) float32 in world frame.
    """
    quat_t = torch.tensor(cam_quat_w, dtype=torch.float32).unsqueeze(0)  # (1, 4)
    R = math_utils.matrix_from_quat(quat_t).squeeze(0).numpy()           # (3, 3)

    pts_world = pts_cam @ R.T + cam_pos_w  # broadcast over N
    return pts_world.astype(np.float32)


def _filter_by_radius(
    pts_world: np.ndarray,
    centre_world: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Keep only points within ``radius`` metres of ``centre_world``."""
    dist2 = np.sum((pts_world - centre_world) ** 2, axis=1)
    return pts_world[dist2 <= radius ** 2]


# ---------------------------------------------------------------------------
# Depth-camera GraspGenX client
# ---------------------------------------------------------------------------

class GraspGenXDepthClient:
    """Query GraspGenX using live depth-camera point clouds.

    Args:
        camera_name:     IsaacLab scene key for the depth camera to use,
                         e.g. ``"table_top_cam"`` or ``"wrist_cam"``.
        host:            GraspGenX server hostname (default ``"localhost"``).
        port:            GraspGenX server port (default ``5556``).
        gripper_name:    Gripper asset name, e.g. ``"franka_panda"``.
        num_grasps:      Number of grasp candidates the server samples.
        topk_num_grasps: Top-K grasps returned after server-side ranking.
        filter_radius:   Radius (metres) around the object centre to keep when
                         segmenting the scene point cloud. Tune this to cover
                         the largest object in your scene.
        num_pc_points:   Max points sent to server; ``None`` = all valid pixels.
        timeout_ms:      ZMQ request timeout in milliseconds.
    """

    def __init__(
        self,
        camera_name: str = "table_top_cam",
        host: str = "localhost",
        port: int = 5556,
        gripper_name: str | None = "franka_panda",
        num_grasps: int = 200,
        topk_num_grasps: int = 10,
        filter_radius: float = 0.15,
        num_pc_points: int | None = 4096,
        timeout_ms: int = 60_000,
    ) -> None:
        self.camera_name = camera_name
        self.gripper_name = gripper_name
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps
        self.filter_radius = filter_radius
        self.num_pc_points = num_pc_points

        _ClientClass = _load_graspgenx_client_class()
        self._client = _ClientClass(host=host, port=port, timeout_ms=timeout_ms)
        self._client.connect()

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GraspGenXDepthClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------

    def query(
        self,
        object_name: str,
        env: "ManagerBasedEnv",
        env_id: int = 0,
    ) -> list[GraspResult]:
        """Query GraspGenX for grasps on a scene object via live depth camera.

        Args:
            object_name: IsaacLab scene key, e.g. ``"object_01"``.
            env:         Live ManagerBasedEnv with cameras initialised.
            env_id:      Environment index (0-indexed).

        Returns:
            List of :class:`GraspResult` in env-local frame, ordered by
            server confidence (highest first). Empty on failure.
        """
        # ---- 1. Camera sensor -------------------------------------------
        camera = env.scene[self.camera_name]

        # ---- 2. Raw depth (before display flip — intrinsics match raw) --
        depth_tensor = camera.data.output.get("distance_to_image_plane")
        if depth_tensor is None:
            logging.warning("[GraspGenXDepthClient] No depth output from '%s'", self.camera_name)
            return []

        depth_m = depth_tensor[env_id, ..., 0].cpu().numpy().astype(np.float32)  # (H, W)
        depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)

        # ---- 3. Intrinsics ----------------------------------------------
        K = camera.data.intrinsic_matrices[env_id].cpu().numpy().astype(np.float32)  # (3, 3)

        # ---- 4. Back-project depth → point cloud in camera frame --------
        pts_cam = _depth_to_pointcloud_camera_frame(depth_m, K)   # (N, 3)
        if len(pts_cam) == 0:
            logging.warning("[GraspGenXDepthClient] Empty depth image from '%s'", self.camera_name)
            return []

        # ---- 5. Camera world pose ----------------------------------------
        cam_pos_w  = camera.data.pos_w[env_id].cpu().numpy().astype(np.float32)   # (3,)
        cam_quat_w = camera.data.quat_w_ros[env_id].cpu().numpy().astype(np.float32)  # (4,) wxyz

        # ---- 6. Transform → world frame ---------------------------------
        pts_world = _transform_camera_to_world(pts_cam, cam_pos_w, cam_quat_w)  # (N, 3)

        # ---- 7. Filter to target object region --------------------------
        obj = env.scene[object_name]
        obj_pos_w = obj.data.root_pos_w[env_id].cpu().numpy().astype(np.float32)  # (3,)
        pts_world = _filter_by_radius(pts_world, obj_pos_w, self.filter_radius)

        if len(pts_world) < 10:
            logging.warning(
                "[GraspGenXDepthClient] Too few points (%d) near '%s' — check filter_radius (%.3f m)",
                len(pts_world), object_name, self.filter_radius,
            )
            return []

        # ---- 8. Subsample -----------------------------------------------
        if self.num_pc_points is not None and len(pts_world) > self.num_pc_points:
            idx = np.random.choice(len(pts_world), self.num_pc_points, replace=False)
            pts_world = pts_world[idx]

        # ---- 9. Center (GraspGenX requires zero-mean input) -------------
        center = pts_world.mean(axis=0)       # (3,) world frame
        pts_centered = pts_world - center

        logging.info(
            "[GraspGenXDepthClient] Querying server for '%s': %d points (camera=%s, centroid=%s)",
            object_name, len(pts_centered), self.camera_name, np.round(center, 4).tolist(),
        )

        # ---- 10. Query GraspGenX server ---------------------------------
        try:
            grasps_c, confidences = self._client.infer(
                point_cloud=pts_centered,
                gripper_name=self.gripper_name,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.topk_num_grasps,
            )
        except Exception as exc:
            logging.warning("[GraspGenXDepthClient] Server call failed for '%s': %s", object_name, exc)
            return []

        logging.info("[GraspGenXDepthClient] Server returned %d grasps for '%s'", len(grasps_c), object_name)
        if len(grasps_c) == 0:
            return []

        # ---- 11. Un-center + convert to env-local frame -----------------
        env_origin = env.scene.env_origins[env_id].cpu().numpy()  # (3,)

        results: list[GraspResult] = []
        for T_c, conf in zip(grasps_c, confidences):
            T_w = T_c.copy()
            T_w[:3, 3] += center            # centroid-relative → world frame

            pos_local = (T_w[:3, 3] - env_origin).astype(np.float32)

            rot = torch.tensor(T_w[:3, :3], dtype=torch.float32)
            quat = math_utils.quat_from_matrix(rot.unsqueeze(0)).squeeze(0)

            results.append(
                GraspResult(
                    position=torch.tensor(pos_local, dtype=torch.float32),
                    quaternion=quat,
                    confidence=float(conf),
                    object_id=object_name,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Convenience: return the raw segmented point cloud for inspection
    # ------------------------------------------------------------------

    def get_object_pointcloud(
        self,
        object_name: str,
        env: "ManagerBasedEnv",
        env_id: int = 0,
    ) -> np.ndarray | None:
        """Return the filtered world-frame point cloud for one object.

        Useful for debugging camera coverage before running GraspGenX.

        Returns:
            (N, 3) float32 array in world frame, or ``None`` on failure.
        """
        camera = env.scene[self.camera_name]
        depth_tensor = camera.data.output.get("distance_to_image_plane")
        if depth_tensor is None:
            return None

        depth_m = depth_tensor[env_id, ..., 0].cpu().numpy().astype(np.float32)
        depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)
        K = camera.data.intrinsic_matrices[env_id].cpu().numpy().astype(np.float32)

        pts_cam = _depth_to_pointcloud_camera_frame(depth_m, K)
        if len(pts_cam) == 0:
            return None

        cam_pos_w  = camera.data.pos_w[env_id].cpu().numpy().astype(np.float32)
        cam_quat_w = camera.data.quat_w_ros[env_id].cpu().numpy().astype(np.float32)
        pts_world  = _transform_camera_to_world(pts_cam, cam_pos_w, cam_quat_w)

        obj = env.scene[object_name]
        obj_pos_w = obj.data.root_pos_w[env_id].cpu().numpy().astype(np.float32)
        return _filter_by_radius(pts_world, obj_pos_w, self.filter_radius)
