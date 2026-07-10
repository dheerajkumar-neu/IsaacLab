# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
GraspGenXDepthClient — queries the GraspGenX ZMQ server using live depth-camera
point clouds instead of USD-stage mesh vertices.

Pipeline per query()
--------------------
  1. For EACH requested camera: read raw depth image + intrinsics, and mask the
     depth to pixels whose semantic class matches the target object (the env cfg
     tags objects with semantic class == scene key, and the cameras render
     ``semantic_segmentation`` with raw IDs). Bin walls, table, robot and
     neighbouring objects never enter the cloud.
  2. Back-project every remaining pixel to a 3D point in camera frame.
  3. Transform points to world frame using the camera's live world pose.
  4. Concatenate the per-camera clouds into one fused world-frame cloud.
  5. Filter to points within ``filter_radius`` metres of the target object centre
     (safety net; mostly redundant with the semantic mask).
  6. Remove the table surface (lowest ``table_margin`` metres of the crop) —
     safety net for cameras without semantic output.
  7. Optionally subsample to ``num_pc_points``.
  8. Subtract centroid (required by GraspGenX) and send to ZMQ server.
  9. Add centroid back, subtract env_origin, rotate each grasp from GraspGenX's
     canonical gripper frame to the panda_hand frame (Rz +90 deg), and wrap as
     GraspResult.

All cameras render in the same ``env.sim.render()`` call, so the fused views are
time-consistent. Fusing multiple viewpoints (e.g. top + two lateral cameras) gives
GraspGenX complete object geometry instead of just the top surface.

Usage::

    client = GraspGenXDepthClient(
        camera_names=["table_top_cam", "table_side_cam_1", "table_side_cam_2"],
        host="localhost",
    )
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
from data_collection.grasp_client.grasp_result import GraspResult, graspgen_rot_to_panda_hand

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


def _remove_table_points(
    pts_world: np.ndarray,
    table_margin: float,
) -> np.ndarray:
    """Drop the table surface from an object-cropped point cloud.

    The radius crop around an object sitting on the table inevitably keeps a
    disc of table surface. GraspGenX (trained on object-centric clouds) then
    proposes high-confidence pinch grasps on the RIM of that disc — a ring of
    horizontal grasps around the object at table height that cuRobo rejects
    with IK_FAIL (hand collides with the table). Verified in the 2026-07-10
    debug_02 run: every chosen TCP sat at table z within ~1 cm of the crop
    radius.

    The table is the lowest flat structure in the crop, so everything within
    ``table_margin`` metres of the crop's minimum z is removed. This also trims
    up to ``table_margin`` off the object's base — harmless, since fingers
    cannot reach below that height anyway.

    Args:
        pts_world:    (N, 3) cropped cloud in world frame (z up).
        table_margin: Height band above the lowest point to remove. <= 0
                      disables the filter.

    Returns:
        (M, 3) cloud with table points removed.
    """
    if table_margin <= 0.0 or len(pts_world) == 0:
        return pts_world
    z_cut = float(pts_world[:, 2].min()) + table_margin
    return pts_world[pts_world[:, 2] > z_cut]


# ---------------------------------------------------------------------------
# Depth-camera GraspGenX client
# ---------------------------------------------------------------------------

class GraspGenXDepthClient:
    """Query GraspGenX using live depth-camera point clouds fused from one or
    more cameras.

    Args:
        camera_names:    IsaacLab scene key(s) for the depth camera(s) to fuse,
                         e.g. ``["table_top_cam", "table_side_cam_1",
                         "table_side_cam_2"]``. A single string is also accepted.
        host:            GraspGenX server hostname (default ``"localhost"``).
        port:            GraspGenX server port (default ``5556``).
        gripper_name:    Gripper asset name, e.g. ``"franka_panda"``.
        num_grasps:      Number of grasp candidates the server samples.
        topk_num_grasps: Top-K grasps returned after server-side ranking.
        filter_radius:   Radius (metres) around the object centre to keep when
                         segmenting the scene point cloud. Tune this to cover
                         the largest object in your scene.
        table_margin:    Height band (metres) above the crop's lowest point that
                         is removed as table surface, so GraspGenX sees only
                         object geometry. <= 0 disables table removal.
        num_pc_points:   Max points sent to server; ``None`` = all valid pixels.
        timeout_ms:      ZMQ request timeout in milliseconds.
    """

    def __init__(
        self,
        camera_names: str | list[str] = ("table_top_cam", "table_side_cam_1", "table_side_cam_2"),
        host: str = "localhost",
        port: int = 5556,
        gripper_name: str | None = "franka_panda",
        num_grasps: int = 200,
        topk_num_grasps: int = 10,
        filter_radius: float = 0.15,
        table_margin: float = 0.015,
        num_pc_points: int | None = 4096,
        timeout_ms: int = 60_000,
    ) -> None:
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        self.camera_names = list(camera_names)
        if not self.camera_names:
            raise ValueError("camera_names must contain at least one camera")
        self.gripper_name = gripper_name
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps
        self.filter_radius = filter_radius
        self.table_margin = table_margin
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

    @staticmethod
    def _semantic_mask(
        camera,
        camera_name: str,
        env_id: int,
        target_class: str,
    ) -> np.ndarray | None:
        """Per-pixel mask of the pixels whose semantic class is ``target_class``.

        Uses the camera's ``semantic_segmentation`` output (raw int IDs, requires
        ``colorize_semantic_segmentation=False`` in the CameraCfg) together with
        its ``idToLabels`` info to isolate one tagged object — e.g. the
        ``semantic_tags=[("class", "object_01")]`` set in the env cfg.

        Returns:
            (H, W) bool array, or ``None`` if the camera has no semantic output
            (caller falls back to geometric filtering only).
        """
        seg_tensor = camera.data.output.get("semantic_segmentation")
        if seg_tensor is None:
            return None
        info = camera.data.info[env_id].get("semantic_segmentation") or {}
        id_to_labels = info.get("idToLabels") or {}
        # idToLabels: {"3": {"class": "object_01"}, ...} — ids are per-camera.
        target_ids = [
            int(seg_id) for seg_id, label in id_to_labels.items()
            if target_class in (label or {}).values()
        ]
        if not target_ids:
            logging.debug(
                "[GraspGenXDepthClient] '%s' not in idToLabels of '%s' (labels: %s)",
                target_class, camera_name, id_to_labels,
            )
            return np.zeros(seg_tensor.shape[1:3], dtype=bool)
        seg = seg_tensor[env_id, ..., 0].cpu().numpy()  # (H, W) int ids
        return np.isin(seg, target_ids)

    def _camera_pointcloud_world(
        self,
        camera_name: str,
        env: "ManagerBasedEnv",
        env_id: int,
        target_class: str | None = None,
    ) -> np.ndarray | None:
        """Back-project one camera's depth image to a world-frame point cloud.

        Args:
            target_class: If given and the camera renders semantic segmentation,
                          only depth pixels of this semantic class are kept —
                          the returned cloud contains just that object.

        Returns:
            (N, 3) float32 array in world frame, or ``None`` if the camera has
            no depth output / no valid pixels.
        """
        camera = env.scene[camera_name]

        # Raw depth (before display flip — intrinsics match raw)
        depth_tensor = camera.data.output.get("distance_to_image_plane")
        if depth_tensor is None:
            logging.warning("[GraspGenXDepthClient] No depth output from '%s'", camera_name)
            return None

        depth_m = depth_tensor[env_id, ..., 0].cpu().numpy().astype(np.float32)  # (H, W)
        depth_m = np.where(np.isfinite(depth_m), depth_m, 0.0)

        # Semantic mask: zero out every pixel that is not the target object, so
        # bin walls / table / neighbouring objects never reach the point cloud.
        if target_class is not None:
            mask = self._semantic_mask(camera, camera_name, env_id, target_class)
            if mask is not None:
                depth_m = np.where(mask, depth_m, 0.0)
            else:
                logging.warning(
                    "[GraspGenXDepthClient] '%s' has no semantic_segmentation output — "
                    "using geometric filters only (add it to the camera data_types).",
                    camera_name,
                )

        K = camera.data.intrinsic_matrices[env_id].cpu().numpy().astype(np.float32)  # (3, 3)

        pts_cam = _depth_to_pointcloud_camera_frame(depth_m, K)   # (N, 3)
        if len(pts_cam) == 0:
            if target_class is not None:
                # Normal with semantic masking: the object is simply occluded /
                # out of view for this camera; the other cameras cover it.
                logging.debug(
                    "[GraspGenXDepthClient] No '%s' pixels visible from '%s'",
                    target_class, camera_name,
                )
            else:
                logging.warning("[GraspGenXDepthClient] Empty depth image from '%s'", camera_name)
            return None

        cam_pos_w  = camera.data.pos_w[env_id].cpu().numpy().astype(np.float32)   # (3,)
        cam_quat_w = camera.data.quat_w_ros[env_id].cpu().numpy().astype(np.float32)  # (4,) wxyz

        return _transform_camera_to_world(pts_cam, cam_pos_w, cam_quat_w)  # (N, 3)

    def _fused_pointcloud_world(
        self,
        env: "ManagerBasedEnv",
        env_id: int,
        target_class: str | None = None,
    ) -> np.ndarray | None:
        """Fuse all configured cameras into one world-frame point cloud.

        Args:
            target_class: Optional semantic class to isolate per camera (see
                          ``_camera_pointcloud_world``).

        Returns:
            (N, 3) float32 array in world frame, or ``None`` if no camera
            produced any points.
        """
        clouds: list[tuple[str, np.ndarray]] = []
        for cam_name in self.camera_names:
            pts = self._camera_pointcloud_world(cam_name, env, env_id, target_class=target_class)
            if pts is not None and len(pts) > 0:
                clouds.append((cam_name, pts))
        if not clouds:
            return None
        fused = np.concatenate([pts for _, pts in clouds], axis=0)
        logging.debug(
            "[GraspGenXDepthClient] Fused cloud: %d points from %d camera(s) (%s)",
            len(fused), len(clouds),
            ", ".join(f"{name}={len(pts)}" for name, pts in clouds),
        )
        return fused

    def query(
        self,
        object_name: str,
        env: "ManagerBasedEnv",
        env_id: int = 0,
    ) -> list[GraspResult]:
        """Query GraspGenX for grasps on a scene object via fused depth cameras.

        Args:
            object_name: IsaacLab scene key, e.g. ``"object_01"``.
            env:         Live ManagerBasedEnv with cameras initialised.
            env_id:      Environment index (0-indexed).

        Returns:
            List of :class:`GraspResult` in env-local frame, ordered by
            server confidence (highest first). Empty on failure.
        """
        # ---- 1-6. Back-project every camera and fuse into one world cloud.
        # The env cfg tags each object with semantic class == scene key
        # (e.g. ("class", "object_01")), so pixels of the bin, table, robot and
        # neighbouring objects are masked out per camera before back-projection.
        pts_world = self._fused_pointcloud_world(env, env_id, target_class=object_name)
        if pts_world is None:
            logging.warning(
                "[GraspGenXDepthClient] No '%s' points from any camera (%s)",
                object_name, ", ".join(self.camera_names),
            )
            return []

        # ---- 7. Filter to target object region --------------------------
        obj = env.scene[object_name]
        obj_pos_w = obj.data.root_pos_w[env_id].cpu().numpy().astype(np.float32)  # (3,)
        pts_world = _filter_by_radius(pts_world, obj_pos_w, self.filter_radius)

        # ---- 7b. Remove the table surface from the crop ------------------
        num_before = len(pts_world)
        pts_world = _remove_table_points(pts_world, self.table_margin)
        if num_before:
            logging.debug(
                "[GraspGenXDepthClient] Table filter for '%s': %d -> %d points "
                "(removed %d table points, margin=%.3f m)",
                object_name, num_before, len(pts_world), num_before - len(pts_world),
                self.table_margin,
            )

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
            "[GraspGenXDepthClient] Querying server for '%s': %d points (cameras=%s, centroid=%s)",
            object_name, len(pts_centered), ",".join(self.camera_names), np.round(center, 4).tolist(),
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

            # GraspGenX canonical gripper frame → panda_hand frame (Rz +90 deg,
            # see graspgen_rot_to_panda_hand) so cuRobo closes the fingers
            # across the axis GraspGenX actually chose.
            rot = graspgen_rot_to_panda_hand(torch.tensor(T_w[:3, :3], dtype=torch.float32))
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
        """Return the filtered, fused world-frame point cloud for one object.

        Useful for debugging camera coverage before running GraspGenX — this is
        exactly the cloud (pre-subsample, pre-centering) that query() segments.

        Returns:
            (N, 3) float32 array in world frame, or ``None`` on failure.
        """
        pts_world = self._fused_pointcloud_world(env, env_id, target_class=object_name)
        if pts_world is None:
            return None

        obj = env.scene[object_name]
        obj_pos_w = obj.data.root_pos_w[env_id].cpu().numpy().astype(np.float32)
        pts_world = _filter_by_radius(pts_world, obj_pos_w, self.filter_radius)
        return _remove_table_points(pts_world, self.table_margin)
