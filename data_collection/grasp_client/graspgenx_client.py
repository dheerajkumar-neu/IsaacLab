# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
GraspGenXIsaacClient — queries the GraspGenX ZMQ server using point clouds
extracted live from the running USD stage.

Usage::

    client = GraspGenXIsaacClient(host="localhost", port=5556)
    grasps = client.query("object_01", env, env_id=0)
    filtered = filter_grasps(grasps, confidence_threshold=0.5, top_k=1)

The GraspGenX server must be running and reachable before calling query().
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch
from pxr import Usd, UsdGeom

import isaaclab.utils.math as math_utils
from data_collection.grasp_client.grasp_result import GraspResult

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Path to the GraspGenX repository (parent of IsaacLab)
_GRASPGENX_REPO = "/groot/data/ubuntu/GraspGenX"
_GRASPGENX_ZMQ_CLIENT_PATH = f"{_GRASPGENX_REPO}/graspgenx/serving/zmq_client.py"


def _load_graspgenx_client_class():
    """Load GraspGenXClient directly from file, bypassing graspgenx/serving/__init__.py.

    The serving __init__ imports the server-side ZMQServer which requires omegaconf
    (not installed in IsaacSim Python). Loading zmq_client.py directly avoids this.
    """
    if _GRASPGENX_REPO not in sys.path:
        sys.path.insert(0, _GRASPGENX_REPO)
    spec = importlib.util.spec_from_file_location(
        "graspgenx_zmq_client",
        _GRASPGENX_ZMQ_CLIENT_PATH,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GraspGenXClient


def _extract_world_pointcloud(
    prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
) -> np.ndarray | None:
    """Walk the prim subtree, collect all mesh vertices transformed to world frame.

    USD uses row-vector convention: a local point p transforms to world as
        p_world = p_local @ T[:3,:3] + T[3,:3]
    where T = np.array(Gf.Matrix4d).reshape(4,4) and translation sits in row 3.

    Args:
        prim:        Root USD prim of the object (e.g. /World/envs/env_0/Object01).
        xform_cache: XformCache instance for efficient matrix lookups.

    Returns:
        (N, 3) float32 array in world frame, or None if no meshes found.
    """
    parts: list[np.ndarray] = []
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(child)
        pts_attr = mesh.GetPointsAttr()
        if pts_attr is None or not pts_attr.HasValue():
            continue
        pts_local = np.array(pts_attr.Get(), dtype=np.float64)  # (N, 3)
        if len(pts_local) == 0:
            continue
        gf_mat = xform_cache.GetLocalToWorldTransform(child)
        T = np.array(gf_mat).reshape(4, 4)  # row-vector convention
        pts_w = (pts_local @ T[:3, :3] + T[3, :3]).astype(np.float32)
        parts.append(pts_w)

    return np.concatenate(parts, axis=0) if parts else None


class GraspGenXIsaacClient:
    """Queries the GraspGenX ZMQ server using USD-stage mesh point clouds.

    Workflow per query():
      1. Locate the object prim in the live USD stage via env.scene entity.
      2. Walk its subtree, collect all mesh vertices, transform to world frame.
      3. Optionally subsample to num_pc_points.
      4. Subtract the point cloud centroid (required by GraspGenX).
      5. Send to ZMQ server; receive (K, 4, 4) grasps + (K,) confidences in
         centroid-relative frame.
      6. Add back centroid to un-center, subtract env_origin to convert to
         env-local frame (matching the convention used by PackMotionPlanner).
      7. Wrap each grasp as a GraspResult and return the full list.

    Args:
        host:            GraspGenX server hostname (default "localhost").
        port:            GraspGenX server port (default 5556).
        gripper_name:    Gripper asset name passed to the server
                         (e.g. "franka_panda"). None uses server default.
        num_grasps:      Number of grasp candidates to sample on the server.
        topk_num_grasps: Top-K grasps returned after server-side ranking.
        num_pc_points:   Max point cloud size sent to server; None = all vertices.
        timeout_ms:      ZMQ request timeout in milliseconds.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        gripper_name: str | None = "franka_panda",
        num_grasps: int = 200,
        topk_num_grasps: int = 10,
        num_pc_points: int | None = 4096,
        timeout_ms: int = 60_000,
    ) -> None:
        self.gripper_name = gripper_name
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps
        self.num_pc_points = num_pc_points

        _ClientClass = _load_graspgenx_client_class()
        self._client = _ClientClass(host=host, port=port, timeout_ms=timeout_ms)
        self._client.connect()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GraspGenXIsaacClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def query(
        self,
        object_name: str,
        env: "ManagerBasedEnv",
        env_id: int = 0,
    ) -> list[GraspResult]:
        """Query GraspGenX for grasps on a scene object.

        Args:
            object_name: IsaacLab scene key, e.g. "object_01".
            env:         Live ManagerBasedEnv with a running USD stage.
            env_id:      Environment index (0-indexed).

        Returns:
            List of GraspResult in env-local frame, ordered by server confidence
            (highest first). Empty on extraction failure or server error.
        """
        # ---------- 1. Locate prim in USD stage ----------
        stage = env.sim.stage
        scene_entity = env.scene[object_name]
        prim_path = scene_entity.root_physx_view.prim_paths[env_id]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            logging.warning("[GraspGenXIsaacClient] Prim not found: %s", prim_path)
            return []

        # ---------- 2. Extract world-frame point cloud ----------
        xform_cache = UsdGeom.XformCache()
        pts_world = _extract_world_pointcloud(prim, xform_cache)
        if pts_world is None or len(pts_world) == 0:
            logging.warning("[GraspGenXIsaacClient] No mesh vertices under: %s", prim_path)
            return []

        # ---------- 3. Subsample ----------
        if self.num_pc_points is not None and len(pts_world) > self.num_pc_points:
            idx = np.random.choice(len(pts_world), self.num_pc_points, replace=False)
            pts_world = pts_world[idx]

        # ---------- 4. Center (GraspGenX requires zero-mean input) ----------
        center = pts_world.mean(axis=0)       # (3,) float32, world frame
        pts_centered = pts_world - center      # (N, 3)

        logging.info(
            "[GraspGenXIsaacClient] Querying server for '%s': %d mesh points (centroid=%s)",
            object_name, len(pts_centered), np.round(center, 4).tolist(),
        )

        # ---------- 5. Query server ----------
        try:
            grasps_c, confidences = self._client.infer(
                point_cloud=pts_centered,
                gripper_name=self.gripper_name,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.topk_num_grasps,
            )
        except Exception as exc:
            logging.warning("[GraspGenXIsaacClient] Server call failed for '%s': %s", object_name, exc)
            return []

        logging.info("[GraspGenXIsaacClient] Server returned %d grasps for '%s'", len(grasps_c), object_name)
        if len(grasps_c) == 0:
            return []

        # ---------- 6 & 7. Un-center, convert to env-local, wrap ----------
        env_origin = env.scene.env_origins[env_id].cpu().numpy()  # (3,) world offset

        results: list[GraspResult] = []
        for T_c, conf in zip(grasps_c, confidences):
            # T_c: (4,4) SE(3) in centroid frame, translation in column T_c[:3, 3]
            T_w = T_c.copy()
            T_w[:3, 3] += center           # → absolute world frame

            # Env-local position (matching _mock_grasp convention)
            pos_local = (T_w[:3, 3] - env_origin).astype(np.float32)

            # Rotation matrix → quaternion (wxyz)
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
