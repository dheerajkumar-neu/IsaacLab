#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
convert_to_lerobot.py — convert HDF5 episodes written by collect_packing_demos.py
(--dataset_format vla) into a LeRobot dataset (parquet + mp4) with pi-0/pi-0.5
compatible feature keys.

Must run under the `vla_lerobot` conda env, which has the `lerobot` package
(with the pi0/pi0_fast/pi05 policies) installed. collect_packing_demos.py runs
under `env_isaaclab` (Isaac Sim's own Python/torch) and cannot import `lerobot`
directly, so this is a separate post-collection step.

Usage
-----
  conda run -n vla_lerobot python data_collection/scripts/convert_to_lerobot.py \
      --input "datasets/*.hdf5" \
      --output-dir dataset/lerobot \
      --repo-id local/packing_dataset

Arguments
---------
  --input        Glob pattern of HDF5 file(s) written with --dataset_format vla
                 (default: datasets/*.hdf5).
  --output-dir   Directory for the new LeRobot dataset — must not already exist,
                 LeRobotDataset.create() creates it (default: dataset/lerobot).
  --repo-id      LeRobot repo id, e.g. 'local/packing_dataset'.
  --task-desc    Language instruction override. Default: read from each HDF5
                 file's 'task_description' attribute (set by --dataset_format vla).
  --fps          Frames per second stamped on the dataset (default: 24, matching
                 the collection env's control rate: decimation=5 @ sim.dt=1/120,
                 see packing_env_cfg.py). One recorded step already equals one
                 env.step(), so no resampling is needed.
  --robot-type   Robot type string stored in metadata (default: franka_panda).

Only episode groups containing 'action', 'wrist_cam_rgb', and 'table_top_cam_rgb'
datasets are converted — episodes recorded with --dataset_format proprio (no
action/frame data) are skipped, not errored on, so a mixed-mode HDF5 file
converts cleanly.

Feature schema (matches lerobot.utils.constants: OBS_STATE, OBS_IMAGES, ACTION):
  observation.state             float32 (16,) = joint_pos(9) + ee_pos(3) + ee_quat(4)
  observation.images.wrist_cam      video (H, W, 3)
  observation.images.table_top_cam  video (H, W, 3)
  action                         float32 (8,)  = 7 commanded arm joint targets +
                                                  1 gripper command
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REQUIRED_KEYS = ("action", "wrist_cam_rgb", "table_top_cam_rgb")
STATE_DIM = 16  # joint_pos(9) + ee_pos(3) + ee_quat(4)
ACTION_DIM = 8  # 7 commanded arm joint targets + 1 gripper command


def _build_features(image_shape: tuple[int, int, int]) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {
                "axes": [f"joint_pos_{i}" for i in range(9)]
                + ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"]
            },
        },
        "observation.images.wrist_cam": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.table_top_cam": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {"axes": [f"arm_joint_{i}" for i in range(7)] + ["gripper"]},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VLA-mode packing HDF5 episodes into a LeRobot dataset."
    )
    parser.add_argument("--input", type=str, default="datasets/*.hdf5",
                        help="Glob pattern of HDF5 file(s) written with --dataset_format vla.")
    parser.add_argument("--output-dir", type=str, default="dataset/lerobot",
                        help="Directory for the new LeRobot dataset (must not already exist).")
    parser.add_argument("--repo-id", type=str, default="local/packing_dataset")
    parser.add_argument("--task-desc", type=str, default=None,
                        help="Language instruction override; default reads each file's "
                             "'task_description' attribute.")
    parser.add_argument("--fps", type=int, default=24,
                        help="Matches the collection env's control rate "
                             "(decimation=5 @ sim.dt=1/120).")
    parser.add_argument("--robot-type", type=str, default="franka_panda")
    args = parser.parse_args()

    hdf5_paths = sorted(Path(p) for p in glob.glob(args.input))
    if not hdf5_paths:
        raise ValueError(f"--input {args.input!r} matched no files.")
    logging.info("Found %d HDF5 file(s): %s", len(hdf5_paths), [str(p) for p in hdf5_paths])

    # Collect (path, episode_key, task) for every convertible episode, and validate
    # up front that every one of them has a task instruction before writing anything.
    to_convert: list[tuple[Path, str, str]] = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            file_task = f.attrs.get("task_description")
            task = args.task_desc or file_task
            episode_keys = sorted(f["episodes"].keys())
            convertible = [k for k in episode_keys if all(rk in f["episodes"][k] for rk in REQUIRED_KEYS)]
            logging.info("%s: %d/%d episodes have VLA fields (action + both cameras)",
                        path, len(convertible), len(episode_keys))
            if convertible and not task:
                raise ValueError(
                    f"{path} has no 'task_description' attribute and no --task-desc override "
                    "was given — was it collected with --dataset_format vla?"
                )
            to_convert.extend((path, ep_key, task) for ep_key in convertible)

    if not to_convert:
        raise ValueError(
            "No episodes with 'action'/'wrist_cam_rgb'/'table_top_cam_rgb' found — "
            "did you collect with --dataset_format vla?"
        )
    logging.info("Converting %d episode(s) total.", len(to_convert))

    first_path, first_ep_key, _ = to_convert[0]
    with h5py.File(first_path, "r") as f:
        image_shape = tuple(f["episodes"][first_ep_key]["wrist_cam_rgb"].shape[1:])

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=_build_features(image_shape),
        root=args.output_dir,
        robot_type=args.robot_type,
        use_videos=True,
    )

    for hdf5_path, ep_key, task in to_convert:
        with h5py.File(hdf5_path, "r") as f:
            grp = f["episodes"][ep_key]
            state = np.concatenate(
                [grp["joint_pos"][:], grp["ee_pos"][:], grp["ee_quat"][:]], axis=1
            ).astype(np.float32)
            action = grp["action"][:].astype(np.float32)
            wrist = grp["wrist_cam_rgb"][:]
            table_top = grp["table_top_cam_rgb"][:]
            num_steps = state.shape[0]

            for t in range(num_steps):
                dataset.add_frame({
                    "observation.state": state[t],
                    "observation.images.wrist_cam": wrist[t],
                    "observation.images.table_top_cam": table_top[t],
                    "action": action[t],
                    "task": task,
                })

        dataset.save_episode()
        logging.info("Saved %s:%s (%d steps, task=%r)", hdf5_path, ep_key, num_steps, task)

    dataset.finalize()
    logging.info("LeRobot dataset written to %s (%d episodes)", args.output_dir, len(to_convert))


if __name__ == "__main__":
    main()
