#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
lerobot_recorder.py — writes packing-demo episodes to disk in GR00T LeRobot
(LeRobot v2 + meta/modality.json) format:

  <output_dir>/meta/{episodes.jsonl, tasks.jsonl, info.json, modality.json, stats.json}
  <output_dir>/videos/chunk-000/observation.images.<cam>/episode_NNNNNN.mp4
  <output_dir>/data/chunk-000/episode_NNNNNN.parquet

Only episodes marked successful in close_episode() are written to disk — failed
episodes are discarded (mirrors the old ActionStateRecorderManagerCfg's
EXPORT_SUCCEEDED_ONLY behaviour). episode_index and the global frame `index` are
only assigned to WRITTEN episodes, so both stay contiguous with no gaps.

Per-camera video frames are streamed to a temp raw-rgb24 file during the episode
(not buffered in memory) and only encoded to mp4 (via an ffmpeg subprocess,
h264/yuv420p) if the episode is kept; otherwise the temp file is deleted.

Crash/resume safety: __init__ resumes episode_index/global_frame_index from an
existing meta/episodes.jsonl if output_dir already has one (so re-running against
the same --output CONTINUES a dataset instead of overwriting episode_000000
onward), and finalize() recomputes stats.json fresh from every parquet file on
disk (not from in-memory state) so it is correct regardless of process restarts.
Call finalize() after every episode, not just at the very end of a run — a killed
process (spot preemption, crash, Ctrl-C) then leaves the dataset fully loadable up
through the last episode that finished, instead of with no meta files at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Modality:
    """One named slice of the concatenated state/action array."""
    name: str
    dim: int


class LeRobotRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        camera_names: list[str],
        state_modalities: list[Modality],
        action_modalities: list[Modality],
        task_descriptions: list[str],
        fps: float,
        robot_type: str = "franka",
        image_hw: tuple[int, int] = (480, 640),
        chunk: int = 0,
    ) -> None:
        self._root = Path(output_dir)
        self._cameras = list(camera_names)
        self._state_mods = state_modalities
        self._action_mods = action_modalities
        self._task_descriptions = list(task_descriptions)
        self._fps = fps
        self._robot_type = robot_type
        self._height, self._width = image_hw
        self._chunk = chunk

        (self._root / "meta").mkdir(parents=True, exist_ok=True)
        (self._root / "data" / f"chunk-{chunk:03d}").mkdir(parents=True, exist_ok=True)
        for cam in self._cameras:
            (self._root / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{cam}").mkdir(
                parents=True, exist_ok=True
            )

        self._episode_index = 0        # next WRITTEN episode index
        self._global_frame_index = 0   # next global frame index across written episodes
        self._episodes_meta: list[dict] = []
        self._resume_from_existing_dataset()

        self._recording = False
        self._buf_state: list[np.ndarray] = []
        self._buf_action: list[np.ndarray] = []
        self._buf_timestamp: list[float] = []
        self._buf_task_idx: list[int] = []
        self._raw_video_paths: dict[str, Path] = {}
        self._raw_video_files: dict[str, BinaryIO] = {}

    @property
    def num_episodes_written(self) -> int:
        return self._episode_index

    def _resume_from_existing_dataset(self) -> None:
        """If output_dir already has a dataset (from a prior/interrupted run), resume
        episode_index/global_frame_index from it instead of starting at 0 and
        silently overwriting episode_000000 onward."""
        episodes_path = self._root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            return
        with open(episodes_path) as f:
            self._episodes_meta = [json.loads(line) for line in f if line.strip()]
        self._episode_index = len(self._episodes_meta)
        self._global_frame_index = sum(ep["length"] for ep in self._episodes_meta)

        tasks_path = self._root / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path) as f:
                existing_tasks = [json.loads(line)["task"] for line in f if line.strip()]
            if existing_tasks != self._task_descriptions:
                raise ValueError(
                    f"Resuming into {self._root}, but its meta/tasks.jsonl {existing_tasks} "
                    f"doesn't match the task_descriptions passed now {self._task_descriptions} — "
                    "existing episodes' task_index values would silently point at the wrong task."
                )

        # Guard against resuming with a CHANGED schema (e.g. state/action dims edited,
        # or a different camera set) — without this, old and new episodes would
        # silently coexist with different observation.state/action shapes or missing
        # video keys, which GR00T's loader assumes is impossible (one fixed schema
        # per dataset) and would fail on in a confusing way.
        info_path = self._root / "meta" / "info.json"
        if info_path.exists():
            existing_info = json.loads(info_path.read_text())
            expected_state_dim = sum(m.dim for m in self._state_mods)
            expected_action_dim = sum(m.dim for m in self._action_mods)
            existing_state_dim = existing_info["features"]["observation.state"]["shape"][0]
            existing_action_dim = existing_info["features"]["action"]["shape"][0]
            existing_cameras = sorted(
                k for k in existing_info["features"] if k.startswith("observation.images.")
            )
            expected_cameras = sorted(f"observation.images.{cam}" for cam in self._cameras)
            mismatches = []
            if existing_state_dim != expected_state_dim:
                mismatches.append(f"state_dim {existing_state_dim} -> {expected_state_dim}")
            if existing_action_dim != expected_action_dim:
                mismatches.append(f"action_dim {existing_action_dim} -> {expected_action_dim}")
            if existing_cameras != expected_cameras:
                mismatches.append(f"cameras {existing_cameras} -> {expected_cameras}")
            if mismatches:
                raise ValueError(
                    f"Resuming into {self._root}, but its meta/info.json schema doesn't match "
                    f"what's being recorded now: {'; '.join(mismatches)}. Existing episodes would "
                    "silently have a different observation.state/action shape or camera set than "
                    "new ones — use a fresh output directory instead."
                )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        self._buf_state = []
        self._buf_action = []
        self._buf_timestamp = []
        self._buf_task_idx = []
        self._raw_video_paths = {}
        self._raw_video_files = {}
        for cam in self._cameras:
            # mkstemp() itself opens and returns a real fd (for atomic/secure creation);
            # discarding it and calling open(path) separately — the original bug here —
            # leaks that fd for the life of the process: unlink() only removes the
            # directory entry, so every still-open fd keeps its ~1-3GB backing file
            # alive on disk. os.fdopen() reuses the SAME fd instead of opening a new one.
            fd, path = tempfile.mkstemp(suffix=f"_{cam}.rgb24")
            self._raw_video_paths[cam] = Path(path)
            self._raw_video_files[cam] = os.fdopen(fd, "wb")
        self._recording = True

    def record_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        camera_frames: dict[str, np.ndarray],
        timestamp: float,
        task_index: int = 0,
    ) -> None:
        if not self._recording:
            raise RuntimeError("call start_episode() before record_step()")

        self._buf_state.append(np.asarray(state, dtype=np.float32))
        self._buf_action.append(np.asarray(action, dtype=np.float32))
        self._buf_timestamp.append(float(timestamp))
        self._buf_task_idx.append(int(task_index))

        for cam in self._cameras:
            frame = np.asarray(camera_frames[cam])
            if frame.shape[:2] != (self._height, self._width):
                raise ValueError(
                    f"camera '{cam}' frame shape {frame.shape[:2]} != configured "
                    f"{(self._height, self._width)}"
                )
            # Isaac Lab camera rgb output is RGBA; LeRobot video is RGB.
            rgb = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
            self._raw_video_files[cam].write(rgb.tobytes())

    def close_episode(self, success: bool) -> bool:
        """Finalize the current episode. Returns True iff it was written to disk."""
        if not self._recording:
            return False
        self._recording = False
        for f in self._raw_video_files.values():
            f.close()

        if not success or not self._buf_state:
            for path in self._raw_video_paths.values():
                path.unlink(missing_ok=True)
            return False

        ep_idx = self._episode_index
        num_frames = len(self._buf_state)

        for cam in self._cameras:
            out_mp4 = (
                self._root / "videos" / f"chunk-{self._chunk:03d}"
                / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            )
            self._encode_video(self._raw_video_paths[cam], out_mp4)
            self._raw_video_paths[cam].unlink(missing_ok=True)

        states = np.stack(self._buf_state)   # (T, state_dim)
        actions = np.stack(self._buf_action)  # (T, action_dim)
        self._write_parquet(ep_idx, states, actions)

        seen_task_indices = sorted(set(self._buf_task_idx))
        self._episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [self._task_descriptions[i] for i in seen_task_indices],
            "length": num_frames,
        })

        self._episode_index += 1
        self._global_frame_index += num_frames
        return True

    def _encode_video(self, raw_path: Path, out_path: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{self._width}x{self._height}",
            "-framerate", str(self._fps),
            "-i", str(raw_path),
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)

    def _write_parquet(self, ep_idx: int, states: np.ndarray, actions: np.ndarray) -> None:
        num_frames = len(states)
        schema = pa.schema([
            pa.field("observation.state", pa.list_(pa.float32())),
            pa.field("action", pa.list_(pa.float32())),
            pa.field("timestamp", pa.float32()),
            pa.field("annotation.human.task_description", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("next.reward", pa.float32()),
            pa.field("next.done", pa.bool_()),
        ])
        task_idx = np.asarray(self._buf_task_idx, dtype=np.int64)
        table = pa.Table.from_pydict({
            "observation.state": states.tolist(),
            "action": actions.tolist(),
            "timestamp": np.asarray(self._buf_timestamp, dtype=np.float32),
            "annotation.human.task_description": task_idx,
            "task_index": task_idx,
            "episode_index": np.full(num_frames, ep_idx, dtype=np.int64),
            "index": np.arange(
                self._global_frame_index, self._global_frame_index + num_frames, dtype=np.int64
            ),
            "next.reward": np.zeros(num_frames, dtype=np.float32),
            "next.done": np.array([t == num_frames - 1 for t in range(num_frames)], dtype=bool),
        }, schema=schema)
        out_parquet = self._root / "data" / f"chunk-{self._chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        pq.write_table(table, out_parquet)

    # ------------------------------------------------------------------
    # Dataset-level meta (call once, after all episodes are collected)
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        meta_dir = self._root / "meta"

        with open(meta_dir / "tasks.jsonl", "w") as f:
            for i, task in enumerate(self._task_descriptions):
                f.write(json.dumps({"task_index": i, "task": task}) + "\n")

        with open(meta_dir / "episodes.jsonl", "w") as f:
            for ep in self._episodes_meta:
                f.write(json.dumps(ep) + "\n")

        modality = {
            "state": self._modality_ranges(self._state_mods),
            "action": self._modality_ranges(self._action_mods),
            "video": {
                f"observation.images.{cam}": {"original_key": f"observation.images.{cam}"}
                for cam in self._cameras
            },
            "annotation": {"human.task_description": {}},
        }
        with open(meta_dir / "modality.json", "w") as f:
            json.dump(modality, f, indent=2)

        with open(meta_dir / "info.json", "w") as f:
            json.dump(self._build_info(), f, indent=2)

        with open(meta_dir / "stats.json", "w") as f:
            json.dump(self._compute_stats(), f, indent=2)

    def _compute_stats(self) -> dict:
        """Read every written episode's parquet file fresh off disk (not from
        in-memory buffers) so stats.json is correct regardless of whether this
        process wrote all those episodes or resumed a prior run's dataset.

        Reads by episode_index from self._episodes_meta (the episodes.jsonl source
        of truth) rather than globbing the directory — a process killed mid-write
        can leave an orphaned truncated parquet file at an episode_index that was
        never recorded in episodes.jsonl (the crash happened before close_episode()
        finished), and glob() would pick that garbage file up and crash pyarrow.
        """
        chunk_dir = self._root / "data" / f"chunk-{self._chunk:03d}"
        parquet_paths = [chunk_dir / f"episode_{ep['episode_index']:06d}.parquet" for ep in self._episodes_meta]
        if not parquet_paths:
            return {}
        states, actions = [], []
        for p in parquet_paths:
            table = pq.read_table(p, columns=["observation.state", "action"])
            states.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))
            actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        return {
            "observation.state": self._array_stats(np.concatenate(states, axis=0)),
            "action": self._array_stats(np.concatenate(actions, axis=0)),
        }

    def _build_info(self) -> dict:
        total_episodes = len(self._episodes_meta)
        state_dim = sum(m.dim for m in self._state_mods)
        action_dim = sum(m.dim for m in self._action_mods)

        video_features = {
            f"observation.images.{cam}": {
                "dtype": "video",
                "shape": [self._height, self._width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": self._fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            for cam in self._cameras
        }
        return {
            "codebase_version": "v2.0",
            "robot_type": self._robot_type,
            "total_episodes": total_episodes,
            "total_frames": self._global_frame_index,
            "total_tasks": len(self._task_descriptions),
            "total_videos": total_episodes * len(self._cameras),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self._fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32", "shape": [state_dim],
                    "names": self._expand_names(self._state_mods),
                },
                "action": {
                    "dtype": "float32", "shape": [action_dim],
                    "names": self._expand_names(self._action_mods),
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "annotation.human.task_description": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "next.reward": {"dtype": "float32", "shape": [1], "names": None},
                "next.done": {"dtype": "bool", "shape": [1], "names": None},
                **video_features,
            },
        }

    @staticmethod
    def _modality_ranges(mods: list[Modality]) -> dict[str, dict[str, int]]:
        ranges = {}
        start = 0
        for m in mods:
            ranges[m.name] = {"start": start, "end": start + m.dim}
            start += m.dim
        return ranges

    @staticmethod
    def _expand_names(mods: list[Modality]) -> list[str]:
        names = []
        for m in mods:
            names.extend(f"{m.name}_{i}" for i in range(m.dim))
        return names

    @staticmethod
    def _array_stats(arr: np.ndarray) -> dict[str, list[float]]:
        # GR00T's stats.json validator (gr00t/data/stats.py check_stats_validity) requires
        # exactly these 6 keys per feature — q01/q99 back the percentile normalization mode.
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
