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

  action             (T, 8)   — optional, the raw env action tensor (7 arm + 1
                      gripper) sent to env.step() for this timestep; only present
                      when the recorder is constructed with actions enabled
                      (--dataset_format vla in collect_packing_demos.py).

  <frame_key>        (T, H, W, 3) uint8 — optional per-step RGB frame from a scene
                      camera (e.g. "front_cam_rgb"), one dataset per key in
                      frame_keys (--record_type actions_frames or --dataset_format
                      vla in collect_packing_demos.py).

Scalar per episode:
  success            bool
  num_objects_packed int

File-level attribute:
  task_description   str — language instruction for the whole file, set when the
                      recorder is constructed with task_description (VLA mode);
                      consumed by convert_to_lerobot.py as the LeRobot task.
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

    Pass ``frame_keys`` (e.g. ``["wrist_cam_rgb", "table_top_cam_rgb"]``) to also
    buffer one RGB frame per step under each key — every ``record_step`` call must
    then include a ``frames`` dict covering all of them.

    Pass ``task_description`` to stamp a language instruction onto the file
    (written once as an HDF5 file attribute) for VLA/LeRobot conversion.
    """

    def __init__(
        self,
        output_path: str | Path,
        frame_keys: list[str] | None = None,
        task_description: str | None = None,
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "a")
        if "episodes" not in self._file:
            self._file.create_group("episodes")

        self._episode_idx: int = len(self._file["episodes"])
        self._frame_keys = frame_keys or []
        self._buf: dict[str, list[np.ndarray]] = {}
        self._recording = False

        if task_description is not None:
            self._file.attrs["task_description"] = task_description

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
        for key in self._frame_keys:
            self._buf[key] = []
        if "task_description" in self._file.attrs:
            self._buf["action"] = []
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
        frames: dict[str, np.ndarray | torch.Tensor] | None = None,
        action: torch.Tensor | None = None,
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
            frames:           Dict mapping each of ``frame_keys`` to an (H, W, 3)
                              RGB frame for this step. Required (with all keys
                              present) if the recorder was constructed with
                              frame_keys.
            action:           (8,) raw env action tensor for this step. Required
                              if the recorder was constructed with
                              task_description (VLA mode).
        """
        if not self._recording:
            raise RuntimeError("call start_episode() before record_step()")

        for key in self._frame_keys:
            if frames is None or key not in frames:
                raise ValueError(
                    f"recorder was constructed with frame_keys={self._frame_keys!r}; "
                    f"record_step() requires a frame for every key, missing {key!r}."
                )
            frame = frames[key]
            frame_np = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else frame
            self._buf[key].append(frame_np.astype(np.uint8))

        if "action" in self._buf:
            if action is None:
                raise ValueError(
                    "recorder was constructed with task_description (VLA mode); "
                    "record_step() requires an action on every call."
                )
            action_np = action.detach().cpu().float().numpy() if isinstance(action, torch.Tensor) else action
            self._buf["action"].append(action_np)

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
