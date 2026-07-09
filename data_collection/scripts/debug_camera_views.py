#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
debug_camera_views.py — snapshot every packing-env camera to tune its pose.

Boots the camera-equipped packing env, pins the objects to their fixed init
poses (so framing is reproducible), captures ONE frame from every camera in the
env's ``image_obs_list``, and for each camera writes:

  <out>/<cam>_rgb.png     RGB preview (what the camera sees)
  <out>/<cam>_depth.png   depth preview (valid depth normalized to 0-255; black = no return)
  <out>/<cam>_depth.npy   raw float32 depth in metres

It also prints each camera's world position, ROS quaternion, optical axis and
intrinsics — the numbers to check when tuning ``table_side_cam_*_eye`` in
franka_pack_camera_env_cfg.py.

Workflow
--------
  1. Run this script → inspect <out>/*_rgb.png / *_depth.png.
  2. Edit the ``*_eye`` (or the shared look-at target) in the env cfg.
  3. Re-run. Orientation follows the target automatically.

Usage
-----
  ./isaaclab.sh -p data_collection/scripts/debug_camera_views.py --headless
  ./isaaclab.sh -p data_collection/scripts/debug_camera_views.py --headless \
      --out datasets/camera_debug --randomize

Run under the env_isaaclab conda env (isaaclab.sh uses $CONDA_PREFIX/bin/python).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make 'data_collection' importable regardless of CWD.
_ISAACLAB_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ISAACLAB_ROOT))

parser = argparse.ArgumentParser(description="Snapshot packing-env cameras for pose debugging")
parser.add_argument("--out", type=str, default="datasets/camera_debug",
                    help="Output directory for the RGB/depth previews.")
parser.add_argument("--settle_steps", type=int, default=20,
                    help="Physics settle steps after reset before capturing.")
parser.add_argument("--randomize", action="store_true",
                    help="Randomize object poses (default: pin them to fixed init poses).")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
args = parser.parse_args()

# --------------------------------------------------------------------------
# Bootstrap Isaac Sim (must precede any omni / isaaclab imports)
# --------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks.manager_based.isaaclab_int  # noqa: E402, F401
import isaaclab_tasks.manager_based.isaaclab_int.config.franka  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ENV_ID = "Isaac-Pack-Object-Franka-Camera-v0"


def _save_rgb(rgb: np.ndarray, path: Path) -> None:
    """Save an (H,W,3|4) uint8 RGB array as PNG (best-effort: PIL, else .npy)."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[-1] >= 3:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except Exception:
        np.save(path.with_suffix(".npy"), rgb)


def _save_depth_preview(depth: np.ndarray, path: Path) -> tuple[float, float, float, float]:
    """Normalize valid depth to 0-255 (black = invalid) and save a PNG.

    Returns (min, max, mean, valid_fraction) over valid pixels for logging.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if valid.any():
        dmin = float(depth[valid].min())
        dmax = float(depth[valid].max())
        dmean = float(depth[valid].mean())
        span = (dmax - dmin) if (dmax - dmin) > 1e-6 else 1.0
        norm = np.zeros_like(depth)
        norm[valid] = (depth[valid] - dmin) / span
        img = (norm * 255.0).astype(np.uint8)
    else:
        dmin = dmax = dmean = 0.0
        img = np.zeros(depth.shape, dtype=np.uint8)
    try:
        from PIL import Image
        Image.fromarray(img, mode="L").save(path)
    except Exception:
        np.save(path.with_suffix(".npy"), img)
    return dmin, dmax, dmean, float(valid.mean())


def main() -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)

    # Pin objects to fixed init poses unless --randomize (reproducible framing).
    if not args.randomize and getattr(env_cfg.events, "reset_objects_pose", None) is not None:
        env_cfg.events.reset_objects_pose = None

    env = gym.make(ENV_ID, cfg=env_cfg).unwrapped
    env.reset()

    # Settle physics, then force a render + sensor update so buffers are current.
    for _ in range(args.settle_steps):
        env.sim.step(render=not args.headless)
    env.sim.render()
    env.scene.update(dt=env.physics_dt)

    cam_names = list(getattr(env_cfg, "image_obs_list", []))
    if not cam_names:
        cam_names = [k for k in env.scene.sensors.keys() if "cam" in k.lower()]

    env_origin = env.scene.env_origins[0].detach().cpu().numpy()

    print("\n================ CAMERA POSE / FRAMING DEBUG ================")
    print(f"Env: {ENV_ID}  |  objects {'RANDOMIZED' if args.randomize else 'PINNED (fixed init poses)'}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"env_origin (world): {np.round(env_origin, 4).tolist()}")

    # Object positions (env-local) for reference when judging framing.
    for obj_name in ("object_01", "object_02", "object_03", "packing_bin"):
        if obj_name in env.scene.keys():
            p = (env.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy() - env_origin)
            print(f"  {obj_name:>12s} pos (env-local): {np.round(p, 4).tolist()}")

    lines = []
    for cam in cam_names:
        camera = env.scene[cam]
        out = camera.data.output

        rgb = out["rgb"][0].detach().cpu().numpy() if "rgb" in out else None
        depth = out["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy() \
            if "distance_to_image_plane" in out else None

        pos_w = camera.data.pos_w[0].detach().cpu().numpy()
        quat = camera.data.quat_w_ros[0].detach().cpu().numpy()  # wxyz
        K = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
        pos_local = pos_w - env_origin

        # Optical axis (+z of the ROS camera frame) in world coords, from the quat.
        w, x, y, z = quat
        optical_axis = np.array([
            2 * (x * z + y * w),
            2 * (y * z - x * w),
            1 - 2 * (x * x + y * y),
        ])

        if rgb is not None:
            _save_rgb(rgb, out_dir / f"{cam}_rgb.png")
        d_stats = (0.0, 0.0, 0.0, 0.0)
        if depth is not None:
            np.save(out_dir / f"{cam}_depth.npy", depth.astype(np.float32))
            d_stats = _save_depth_preview(depth, out_dir / f"{cam}_depth.png")

        print(f"\n[{cam}]")
        print(f"  pos (env-local): {np.round(pos_local, 4).tolist()}   pos (world): {np.round(pos_w, 4).tolist()}")
        print(f"  quat_ros (wxyz): {np.round(quat, 4).tolist()}")
        print(f"  optical axis   : {np.round(optical_axis, 4).tolist()}  (points from camera toward scene)")
        print(f"  intrinsics fx,fy,cx,cy: {K[0,0]:.1f}, {K[1,1]:.1f}, {K[0,2]:.1f}, {K[1,2]:.1f}")
        print(f"  depth m  min/mean/max: {d_stats[0]:.3f} / {d_stats[2]:.3f} / {d_stats[1]:.3f}"
              f"   valid pixels: {100*d_stats[3]:.1f}%")
        lines.append(
            f"{cam}\tpos_local={np.round(pos_local,4).tolist()}\tquat_wxyz={np.round(quat,4).tolist()}"
            f"\toptical_axis={np.round(optical_axis,4).tolist()}"
        )

    (out_dir / "camera_poses.txt").write_text("\n".join(lines) + "\n")
    print("\nWrote per-camera RGB/depth previews + camera_poses.txt to", out_dir.resolve())
    print("============================================================\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
