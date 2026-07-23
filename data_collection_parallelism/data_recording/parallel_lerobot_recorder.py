#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
parallel_lerobot_recorder.py — writes packing-demo episodes to disk in GR00T
LeRobot (LeRobot v2 + meta/modality.json) format, from N environments whose
episodes start/finish at different times.

This is a generalization of data_collection/data_recording/lerobot_recorder.py:
the on-disk layout, schema, and single-writer semantics are identical (see that
file's docstring for the full format description) — the only difference is that
episode buffering is keyed by a ``slot`` (one per env_id) instead of a single set
of instance attributes, because with num_envs=N, up to N episodes can be
IN PROGRESS at once (env 3 might be on its second object while env 7 just started
a fresh episode after env 7's previous one finished).

Concurrency note: despite N slots, this class is NOT thread-safe and doesn't need
to be — the collector script is single-threaded (one Python process driving one
shared batched env.step() per tick), so start_episode()/record_step()/
close_episode() calls for different slots are always serialized relative to each
other, never truly concurrent. episode_index and global_frame_index are shared
counters (dataset-wide, not per-slot) incremented atomically-by-construction:
whichever slot's close_episode() runs first gets the next index, exactly as if
every episode across every env had been collected by one sequential single-env
run — the interleaving of WHICH env's episode closes when has no effect on the
resulting dataset's correctness (still contiguous indices, same schema).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
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


@dataclass
class _SlotBuffer:
    """Per-env in-progress episode state. One of these exists per active slot."""
    buf_state: list[np.ndarray] = field(default_factory=list)
    buf_action: list[np.ndarray] = field(default_factory=list)
    buf_timestamp: list[float] = field(default_factory=list)
    buf_task_idx: list[int] = field(default_factory=list)
    raw_video_paths: dict[str, Path] = field(default_factory=dict)
    raw_video_files: dict[str, BinaryIO] = field(default_factory=dict)
    recording: bool = False


class ParallelLeRobotRecorder:
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
        collect_mode: str = "successful_only",
    ) -> None:
        """
        Args:
            collect_mode: "successful_only" (default) discards the WHOLE episode unless
                close_episode()'s `success` is True — unchanged legacy behavior. "all"
                instead keeps each task_index's frames if `object_success` (passed to
                close_episode) marks that task_index True, discarding only the failed
                objects' frames — see close_episode()'s docstring for the correctness
                requirement this depends on (clean home-pose boundaries between objects).
        """
        if collect_mode not in ("successful_only", "all"):
            raise ValueError(f"collect_mode must be 'successful_only' or 'all', got {collect_mode!r}")
        self._collect_mode = collect_mode
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

        self._episode_index = 0        # next WRITTEN episode index (shared across all slots)
        self._global_frame_index = 0   # next global frame index across written episodes
        self._episodes_meta: list[dict] = []
        self._resume_from_existing_dataset()

        self._slots: dict[int, _SlotBuffer] = {}

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
        # or a different camera set) — see lerobot_recorder.py's identical check.
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
    # Per-slot episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, slot: int) -> None:
        """Begin buffering a new episode for ``slot`` (typically the env_id)."""
        buf = _SlotBuffer()
        for cam in self._cameras:
            # mkstemp() itself opens and returns a real fd; os.fdopen() reuses that
            # SAME fd instead of opening a second one — see lerobot_recorder.py's
            # identical comment for why that matters (fd-leak history).
            fd, path = tempfile.mkstemp(suffix=f"_slot{slot}_{cam}.rgb24")
            buf.raw_video_paths[cam] = Path(path)
            buf.raw_video_files[cam] = os.fdopen(fd, "wb")
        buf.recording = True
        self._slots[slot] = buf

    def is_recording(self, slot: int) -> bool:
        buf = self._slots.get(slot)
        return buf is not None and buf.recording

    def record_step(
        self,
        slot: int,
        state: np.ndarray,
        action: np.ndarray,
        camera_frames: dict[str, np.ndarray],
        timestamp: float,
        task_index: int = 0,
    ) -> None:
        buf = self._slots.get(slot)
        if buf is None or not buf.recording:
            raise RuntimeError(f"call start_episode({slot}) before record_step({slot}, ...)")

        buf.buf_state.append(np.asarray(state, dtype=np.float32))
        buf.buf_action.append(np.asarray(action, dtype=np.float32))
        buf.buf_timestamp.append(float(timestamp))
        buf.buf_task_idx.append(int(task_index))

        for cam in self._cameras:
            frame = np.asarray(camera_frames[cam])
            if frame.shape[:2] != (self._height, self._width):
                raise ValueError(
                    f"camera '{cam}' frame shape {frame.shape[:2]} != configured "
                    f"{(self._height, self._width)}"
                )
            # Isaac Lab camera rgb output is RGBA; LeRobot video is RGB.
            rgb = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
            buf.raw_video_files[cam].write(rgb.tobytes())

    def close_episode(
        self,
        slot: int,
        success: bool,
        object_success: dict[int, bool] | None = None,
        aborted: bool = False,
    ) -> bool:
        """Finalize the episode currently buffered in ``slot``.

        Args:
            success: Whole-episode success (all objects packed, no auto-reset abort).
                Always required; drives the "successful_only" collect_mode's all-or-
                nothing decision unchanged from the original behavior.
            object_success: Per-task_index success ({task_index: packed_or_not}), only
                consulted when collect_mode == "all". Requires the caller to have
                returned the arm to the SAME home pose after every object attempt —
                success or failure alike — so that splicing out a failed object's
                frames never creates a discontinuous jump between the frame before and
                after the cut (that invariant lives in the collector script, not here).
            aborted: If the episode ended via an env auto-reset (safety timeout), the
                physical sim state was disrupted mid-motion — there is no clean
                boundary to salvage from, so the WHOLE episode is discarded regardless
                of collect_mode.

        Returns True iff anything was written to disk. Always safe to call even if
        start_episode(slot) was never called (returns False) — mirrors the original
        single-slot recorder's guard.
        """
        buf = self._slots.pop(slot, None)
        if buf is None or not buf.recording:
            return False
        buf.recording = False
        for f in buf.raw_video_files.values():
            f.close()

        def _discard() -> bool:
            for path in buf.raw_video_paths.values():
                path.unlink(missing_ok=True)
            return False

        if not buf.buf_state or aborted:
            return _discard()

        if self._collect_mode == "successful_only" or object_success is None:
            if not success:
                return _discard()
            keep_idx: list[int] | None = None  # None = keep every frame, no filtering needed
        else:  # collect_mode == "all"
            keep_task_indices = {idx for idx, ok in object_success.items() if ok}
            keep_idx = [i for i, t in enumerate(buf.buf_task_idx) if t in keep_task_indices]
            if not keep_idx:
                return _discard()

        ep_idx = self._episode_index

        if keep_idx is None:
            kept_states, kept_actions = buf.buf_state, buf.buf_action
            kept_task_idx = buf.buf_task_idx
        else:
            kept_states = [buf.buf_state[i] for i in keep_idx]
            kept_actions = [buf.buf_action[i] for i in keep_idx]
            kept_task_idx = [buf.buf_task_idx[i] for i in keep_idx]
        num_frames = len(kept_states)
        # Re-timestamp contiguously from 0 — the original per-step timestamps assumed no
        # frames would ever be removed; once a middle segment is cut, the surviving frames
        # must renumber to stay dense/monotonic (LeRobot expects one row per fps tick).
        kept_timestamps = [i / self._fps for i in range(num_frames)]

        for cam in self._cameras:
            out_mp4 = (
                self._root / "videos" / f"chunk-{self._chunk:03d}"
                / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
            )
            raw_path = buf.raw_video_paths[cam]
            if keep_idx is not None:
                raw_path = self._filter_raw_video(raw_path, keep_idx)
            self._encode_video(raw_path, out_mp4)
            raw_path.unlink(missing_ok=True)
            if keep_idx is not None:
                buf.raw_video_paths[cam].unlink(missing_ok=True)  # the original, unfiltered temp file

        states = np.stack(kept_states)   # (T, state_dim)
        actions = np.stack(kept_actions)  # (T, action_dim)
        self._write_parquet(ep_idx, states, actions, kept_timestamps, kept_task_idx)

        seen_task_indices = sorted(set(kept_task_idx))
        self._episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [self._task_descriptions[i] for i in seen_task_indices],
            "length": num_frames,
        })

        self._episode_index += 1
        self._global_frame_index += num_frames
        return True

    def _filter_raw_video(self, raw_path: Path, keep_idx: list[int]) -> Path:
        """Return a NEW raw rgb24 temp file containing only the frames at ``keep_idx``
        (in order), for splicing out a failed object's video segment. Caller is
        responsible for deleting the returned path after encoding."""
        frame_bytes = self._height * self._width * 3
        with open(raw_path, "rb") as f:
            data = f.read()
        num_frames = len(data) // frame_bytes
        arr = np.frombuffer(data, dtype=np.uint8).reshape(num_frames, self._height, self._width, 3)
        filtered = arr[keep_idx]
        fd, filtered_path = tempfile.mkstemp(suffix="_filtered.rgb24")
        with os.fdopen(fd, "wb") as f:
            f.write(filtered.tobytes())
        return Path(filtered_path)

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

    def _write_parquet(
        self,
        ep_idx: int,
        states: np.ndarray,
        actions: np.ndarray,
        buf_timestamp: list[float],
        buf_task_idx: list[int],
    ) -> None:
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
        task_idx = np.asarray(buf_task_idx, dtype=np.int64)
        table = pa.Table.from_pydict({
            "observation.state": states.tolist(),
            "action": actions.tolist(),
            "timestamp": np.asarray(buf_timestamp, dtype=np.float32),
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
    # Dataset-level meta (call after every episode close, per the crash-safety
    # convention established for the single-env recorder)
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
        """Read every written episode's parquet file fresh off disk. Uses
        self._episodes_meta (episodes.jsonl's source of truth) rather than
        globbing the directory — see lerobot_recorder.py's identical comment for
        why (orphaned truncated parquet files from a killed mid-write process)."""
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
