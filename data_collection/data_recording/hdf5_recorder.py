# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
HDF5EpisodeRecorder — records demonstration episodes to HDF5 for imitation learning.

Episode format (per timestep arrays, shape (T, …)):
  joint_pos          (T, 9)   — 7 arm + 2 finger joint positions
  ee_pos             (T, 3)   — end-effector world position
  ee_quat            (T, 4)   — end-effector quaternion wxyz
  obj1_pos           (T, 3)   — object_01 world position
  obj1_quat          (T, 4)
  obj2_pos           (T, 3)   — object_02 world position
  obj2_quat          (T, 4)
  obj3_pos           (T, 3)   — object_03 world position
  obj3_quat          (T, 4)
  bin_pos            (T, 3)   — packing_bin world position
  bin_quat           (T, 4)
  grasp_confidence   (T,)     — active grasp confidence (0.0 when no grasp executing)
  grasp_object_idx   (T,)     — 0 = no grasp, 1/2/3 = object index being grasped

Scalar per episode:
  success            bool
  num_objects_packed int
"""

from __future__ import annotations

import numpy as np
import h5py
import torch
from pathlib import Path


class HDF5EpisodeRecorder:
    """Accumulates per-timestep data and writes completed episodes to HDF5.

    Usage::

        recorder = HDF5EpisodeRecorder("demos.hdf5")
        recorder.start_episode()
        for each sim step:
            recorder.record_step(joint_pos, ee_pos, ee_quat,
                                 obj_poses, bin_pose,
                                 grasp_confidence, grasp_object_idx)
        recorder.close_episode(success=True, num_objects_packed=3)
        recorder.close()
    """

    def __init__(self, output_path: str | Path) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "a")
        if "episodes" not in self._file:
            self._file.create_group("episodes")

        self._episode_idx: int = len(self._file["episodes"])
        self._buf: dict[str, list[np.ndarray]] = {}
        self._recording = False

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """Begin a new episode.  Clears any previously accumulated buffer."""
        self._buf = {
            "joint_pos": [],
            "ee_pos": [],
            "ee_quat": [],
            "obj1_pos": [],
            "obj1_quat": [],
            "obj2_pos": [],
            "obj2_quat": [],
            "obj3_pos": [],
            "obj3_quat": [],
            "bin_pos": [],
            "bin_quat": [],
            "grasp_confidence": [],
            "grasp_object_idx": [],
        }
        self._recording = True

    def record_step(
        self,
        joint_pos: torch.Tensor,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        obj_poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
        bin_pose: tuple[torch.Tensor, torch.Tensor],
        grasp_confidence: float = 0.0,
        grasp_object_idx: int = 0,
    ) -> None:
        """Append one timestep to the episode buffer.

        Args:
            joint_pos:        (9,) full robot joint positions.
            ee_pos:           (3,) end-effector position.
            ee_quat:          (4,) end-effector quaternion wxyz.
            obj_poses:        Dict mapping "object_01" / "object_02" / "object_03"
                              to (pos (3,), quat (4,)) tuples.
            bin_pose:         (pos (3,), quat (4,)) for packing_bin.
            grasp_confidence: Confidence score of the active grasp (0.0 if none).
            grasp_object_idx: 1/2/3 for the object being grasped; 0 otherwise.
        """
        if not self._recording:
            raise RuntimeError("call start_episode() before record_step()")

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().float().numpy()

        self._buf["joint_pos"].append(_np(joint_pos))
        self._buf["ee_pos"].append(_np(ee_pos))
        self._buf["ee_quat"].append(_np(ee_quat))

        for label, key_pos, key_quat in [
            ("object_01", "obj1_pos", "obj1_quat"),
            ("object_02", "obj2_pos", "obj2_quat"),
            ("object_03", "obj3_pos", "obj3_quat"),
        ]:
            pos, quat = obj_poses[label]
            self._buf[key_pos].append(_np(pos))
            self._buf[key_quat].append(_np(quat))

        b_pos, b_quat = bin_pose
        self._buf["bin_pos"].append(_np(b_pos))
        self._buf["bin_quat"].append(_np(b_quat))

        self._buf["grasp_confidence"].append(np.array([grasp_confidence], dtype=np.float32))
        self._buf["grasp_object_idx"].append(np.array([grasp_object_idx], dtype=np.int32))

    def close_episode(self, success: bool, num_objects_packed: int = 0) -> None:
        """Write the accumulated buffer to HDF5 and advance the episode counter.

        Args:
            success:            Whether all three objects were packed successfully.
            num_objects_packed: How many objects ended up in the bin.
        """
        if not self._recording:
            return

        ep_key = f"episode_{self._episode_idx:04d}"
        grp = self._file["episodes"].create_group(ep_key)

        for key, frames in self._buf.items():
            if not frames:
                continue
            stacked = np.stack(frames, axis=0)  # (T, *frame_shape)
            # Scalars-per-step are stored as (1,) arrays; squeeze to (T,) for clean indexing.
            if stacked.shape[-1] == 1 and stacked.ndim == 2:
                stacked = stacked.squeeze(-1)
            grp.create_dataset(key, data=stacked, compression="gzip")

        grp.attrs["success"] = success
        grp.attrs["num_objects_packed"] = num_objects_packed
        grp.attrs["num_steps"] = len(self._buf["joint_pos"])

        self._file.flush()
        self._episode_idx += 1
        self._recording = False

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        if self._file.id.valid:
            self._file.flush()
            self._file.close()
