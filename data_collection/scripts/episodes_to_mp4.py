#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
episodes_to_mp4.py — render per-step RGB frames recorded by collect_packing_demos.py
(--record_type actions_frames) into MP4 videos, one per episode.

This is a companion to scripts/tools/hdf5_to_mp4.py, not a drop-in replacement: that
script expects the generic IsaacLab teleop/Mimic recorder layout
(``data/demo_N/obs/<camera_key>``). HDF5EpisodeRecorder (data_collection/data_recording/
hdf5_recorder.py) uses a different layout — ``episodes/episode_NNNN/<camera>_rgb`` —
so this script reads that instead.

Usage
-----
  python data_collection/scripts/episodes_to_mp4.py \
      --input_file datasets/debug_new_object_02/packing_demo_trail.hdf5 \
      --output_dir outputs/debug_new_object_02 \
      --camera front_cam

Arguments
---------
  --input_file   Path to the HDF5 file written by collect_packing_demos.py.
  --output_dir   Directory to save the output MP4 files (created if missing).
  --camera       Scene camera whose frames to render — looks up the
                 "<camera>_rgb" dataset in each episode group (default: front_cam).
  --episodes     Specific episode indices to render (default: all episodes found
                 in the file that contain the requested camera key).
  --framerate    Frames per second for the output video (default: 30).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render recorded episode RGB frames to MP4 videos.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the HDF5 file written by collect_packing_demos.py.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output MP4 files.")
    parser.add_argument("--camera", type=str, default="front_cam",
                        help="Scene camera to render — reads the '<camera>_rgb' dataset "
                             "from each episode group.")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Specific episode indices to render (default: all episodes "
                             "in the file that have the requested camera key).")
    parser.add_argument("--framerate", type=int, default=30,
                        help="Frames per second for the output video.")
    return parser.parse_args()


def write_episode_to_mp4(hdf5_file: str, episode_key: str, frame_key: str, output_dir: str, framerate: int) -> bool:
    """Write one episode's recorded frames to an MP4 file.

    Returns:
        True if a video was written, False if this episode has no frame_key data
        (e.g. an aborted episode, or --record_type actions was used for it).
    """
    with h5py.File(hdf5_file, "r") as f:
        grp = f["episodes"][episode_key]
        if frame_key not in grp:
            return False

        frames = grp[frame_key]
        num_steps, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
        if num_steps == 0:
            return False

        output_path = os.path.join(output_dir, f"{episode_key}_{frame_key}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, framerate, (width, height))

        for i in range(num_steps):
            frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

        video.release()
    return True


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    frame_key = f"{args.camera}_rgb"

    with h5py.File(args.input_file, "r") as f:
        episode_keys = sorted(f["episodes"].keys())

    if args.episodes is not None:
        wanted = {f"episode_{i:04d}" for i in args.episodes}
        episode_keys = [k for k in episode_keys if k in wanted]

    print(f"Found {len(episode_keys)} episode(s) in {args.input_file}; rendering camera '{args.camera}'.")

    written, skipped = 0, 0
    for episode_key in episode_keys:
        if write_episode_to_mp4(args.input_file, episode_key, frame_key, args.output_dir, args.framerate):
            written += 1
            print(f"  wrote {episode_key}_{frame_key}.mp4")
        else:
            skipped += 1
            print(f"  skipped {episode_key} (no '{frame_key}' data — aborted episode or recorded without frames)")

    print(f"Done: {written} video(s) written, {skipped} episode(s) skipped -> {args.output_dir}")


if __name__ == "__main__":
    main()
