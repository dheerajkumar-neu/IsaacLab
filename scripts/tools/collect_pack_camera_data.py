# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect RGB + depth footage from the three RealSense D435-like cameras in the packing environment.

Cameras captured per environment:
  - wrist_cam      : robot wrist (panda_hand), tracks the gripper
  - table_top_cam  : top-down bird's-eye view above the workspace
  - front_cam      : tilted front overview of the whole scene

Output — HDF5 (default, recommended for large datasets):

    <output_dir>/camera_data.hdf5
      /metadata/
        cameras         : list of camera names (taken from env_cfg.image_obs_list,
                          e.g. wrist_cam, table_top_cam, front_cam,
                          table_side_cam_1, table_side_cam_2)
        num_envs        : int
        image_height    : int
        image_width     : int
        intrinsics/
          <cam>         : (3, 3) float32   # one per camera
        extrinsics/
          <cam>         : (4, 4) float32   # camera->world (env-local) transform,
                                           # for fusing clouds into one frame
      /env_0000/
        /wrist_cam/
          rgb   : (T, H, W, 3) uint8
          depth : (T, H, W)    float32  [metres]
        /table_top_cam/  ...
        /front_cam/      ...
      /env_0001/ ...

Output — images (for visual inspection; written for every camera):

    <output_dir>/
      camera_meta.json          # intrinsics + extrinsics per camera
      env_0000/
        wrist_cam/
          rgb/       frame_000000.png ...
          depth/     frame_000000.npy ...  [float32, metres — for reconstruction]
          depth_vis/ frame_000000.png ...  [normalized 0-255 preview, viewable]
        table_top_cam/ ...
        front_cam/ ...
        table_side_cam_1/ ...
        table_side_cam_2/ ...

The default --save_format is 'both': the HDF5 file AND the per-camera image files
are written in the same run.

Usage:

    # HDF5 (default)
    ./isaaclab.sh -p scripts/tools/collect_pack_camera_data.py \\
        --task Isaac-Pack-Object-Franka-Camera-v0 \\
        --num_envs 4 --num_frames 200 --enable_cameras

    # Image files (for manual inspection)
    ./isaaclab.sh -p scripts/tools/collect_pack_camera_data.py \\
        --task Isaac-Pack-Object-Franka-Camera-v0 \\
        --num_envs 2 --num_frames 50 --save_format images --enable_cameras

    # Headless
    ./isaaclab.sh -p scripts/tools/collect_pack_camera_data.py \\
        --task Isaac-Pack-Object-Franka-Camera-v0 \\
        --num_envs 4 --num_frames 200 --headless --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect RGB+depth footage from the packing env cameras.")
parser.add_argument("--task", type=str, default="Isaac-Pack-Object-Franka-Camera-v0", help="Gym task ID.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_frames", type=int, default=200, help="Frames to collect per environment.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./datasets/camera_footage",
    help="Directory to write output files.",
)
parser.add_argument(
    "--save_format",
    type=str,
    choices=["hdf5", "images", "both"],
    default="both",
    help="'hdf5' streams into a single HDF5 file (fast, compact). 'images' writes "
    "per-frame PNG (rgb) + NPY (depth) + a depth preview PNG per camera. 'both' "
    "(default) writes the HDF5 AND the per-camera image files for all cameras.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything that needs Isaac Sim starts here."""

import os

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  registers all envs
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg

# Default cameras; overridden in main() from env_cfg.image_obs_list so ALL cameras
# registered in the scene (including table_side_cam_1/2) are captured.
CAMERA_NAMES = ["wrist_cam", "table_top_cam", "front_cam"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """(num_envs, H, W, 4) GPU tensor → (num_envs, H, W, 3) uint8 numpy (drop alpha).

    Stored RAW (row 0 = top). No flip is applied so RGB, depth and the raw
    intrinsics all share one pixel grid — required for back-projection / fusion
    (matches how data_collection/grasp_client reads the cameras).
    """
    arr = tensor.cpu().numpy()[..., :3].astype(np.uint8)
    return np.ascontiguousarray(arr)


def _to_numpy_depth(tensor: torch.Tensor) -> np.ndarray:
    """(num_envs, H, W, 1) GPU tensor → (num_envs, H, W) float32 numpy (metres).

    Stored RAW (no flip) so it aligns pixel-for-pixel with the RGB image and the
    raw intrinsic matrix. (The previous vertical flip mis-aligned depth vs RGB and
    broke world-frame fusion.)
    """
    arr = tensor.cpu().numpy()[..., 0].astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.ascontiguousarray(arr)


def _get_intrinsics(camera: Camera) -> np.ndarray:
    """Return the (3, 3) intrinsic matrix for env 0 as numpy array."""
    return camera.data.intrinsic_matrices[0].cpu().numpy().astype(np.float32)


def _quat_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w, x, y, z) unit quaternion → (3, 3) rotation matrix."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _get_extrinsics(camera: Camera, env_origin: np.ndarray | None = None) -> np.ndarray:
    """Return the (4, 4) camera→world transform for env 0.

    Built from the camera's world position and ROS-optical-frame quaternion, so
    ``world_xyz = xyz_cam @ T[:3,:3].T + T[:3,3]`` back-projects a depth pixel
    (via the raw intrinsics) into the world/env-local frame — the same convention
    grasp_client and the 3D viewer use. If ``env_origin`` is given the translation
    is expressed env-local (world − origin); for a single env the origin is 0.
    """
    pos = camera.data.pos_w[0].cpu().numpy().astype(np.float64)
    quat = camera.data.quat_w_ros[0].cpu().numpy().astype(np.float64)  # wxyz
    if env_origin is not None:
        pos = pos - np.asarray(env_origin, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_R(quat)
    T[:3, 3] = pos
    return T


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


def _create_hdf5(path: str, num_envs: int, num_frames: int, H: int, W: int, cameras: list[Camera]) -> "h5py.File":
    import h5py

    f = h5py.File(path, "w")

    # Metadata
    meta = f.create_group("metadata")
    meta.attrs["num_envs"] = num_envs
    meta.attrs["num_frames"] = num_frames
    meta.attrs["image_height"] = H
    meta.attrs["image_width"] = W
    meta.attrs["cameras"] = CAMERA_NAMES
    intr = meta.create_group("intrinsics")
    for name, cam in zip(CAMERA_NAMES, cameras):
        intr.create_dataset(name, data=_get_intrinsics(cam))

    # Pre-allocate per-env, per-camera datasets for fast streaming writes
    for env_idx in range(num_envs):
        env_grp = f.create_group(f"env_{env_idx:04d}")
        for name in CAMERA_NAMES:
            cam_grp = env_grp.create_group(name)
            cam_grp.create_dataset(
                "rgb",
                shape=(num_frames, H, W, 3),
                dtype=np.uint8,
                chunks=(1, H, W, 3),  # one frame per chunk for fast random access
            )
            cam_grp.create_dataset(
                "depth",
                shape=(num_frames, H, W),
                dtype=np.float32,
                chunks=(1, H, W),
            )
    return f


def _write_hdf5_frame(hdf5_file, frame_idx: int, cam_name: str, rgb_np: np.ndarray, depth_np: np.ndarray):
    """Write one frame (all envs) for one camera into the pre-allocated HDF5 datasets."""
    num_envs = rgb_np.shape[0]
    for env_idx in range(num_envs):
        hdf5_file[f"env_{env_idx:04d}/{cam_name}/rgb"][frame_idx] = rgb_np[env_idx]
        hdf5_file[f"env_{env_idx:04d}/{cam_name}/depth"][frame_idx] = depth_np[env_idx]


# ---------------------------------------------------------------------------
# Image-file writer
# ---------------------------------------------------------------------------


def _setup_image_dirs(output_dir: str, num_envs: int):
    """Create the per-env / per-camera / rgb+depth+depth_vis subdirectory tree."""
    from pathlib import Path

    for env_idx in range(num_envs):
        for cam_name in CAMERA_NAMES:
            for sub in ("rgb", "depth", "depth_vis"):
                Path(output_dir, f"env_{env_idx:04d}", cam_name, sub).mkdir(parents=True, exist_ok=True)


def _depth_to_preview(depth: np.ndarray) -> np.ndarray:
    """Normalize a (H,W) float depth map (metres) to a (H,W) uint8 image for viewing.
    Valid depth spans 0-255; invalid (<=0 / non-finite) pixels are black."""
    valid = np.isfinite(depth) & (depth > 0.0)
    out = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        dmin = float(depth[valid].min())
        dmax = float(depth[valid].max())
        span = (dmax - dmin) if (dmax - dmin) > 1e-6 else 1.0
        out[valid] = ((depth[valid] - dmin) / span * 255.0).astype(np.uint8)
    return out


def _write_image_frame(output_dir: str, frame_idx: int, cam_name: str, rgb_np: np.ndarray, depth_np: np.ndarray):
    """Save per-env RGB PNG + raw depth NPY + depth preview PNG for one camera, one frame."""
    from pathlib import Path

    try:
        from PIL import Image as PILImage
        _use_pil = True
    except ImportError:
        import cv2
        _use_pil = False

    num_envs = rgb_np.shape[0]
    for env_idx in range(num_envs):
        base = Path(output_dir, f"env_{env_idx:04d}", cam_name)
        # RGB
        rgb_path = str(base / "rgb" / f"frame_{frame_idx:06d}.png")
        # Depth preview (viewable), normalized per-frame
        depth_vis = _depth_to_preview(depth_np[env_idx])
        vis_path = str(base / "depth_vis" / f"frame_{frame_idx:06d}.png")
        if _use_pil:
            PILImage.fromarray(rgb_np[env_idx]).save(rgb_path)
            PILImage.fromarray(depth_vis, mode="L").save(vis_path)
        else:
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_np[env_idx], cv2.COLOR_RGB2BGR))
            cv2.imwrite(vis_path, depth_vis)
        # Depth (float32 metres — .npy so precision is preserved for reconstruction)
        depth_path = str(base / "depth" / f"frame_{frame_idx:06d}.npy")
        np.save(depth_path, depth_np[env_idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ---- Environment setup -----------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Capture every camera the env exposes (image_obs_list now includes the two
    # table_side cameras), not just the original three.
    global CAMERA_NAMES
    img_list = getattr(env_cfg, "image_obs_list", None)
    if img_list:
        CAMERA_NAMES = list(img_list)

    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs

    # Retrieve camera sensor objects from the scene
    scene = env.unwrapped.scene
    cameras: list[Camera] = [scene[name] for name in CAMERA_NAMES]

    # Infer resolution from first camera config
    H = cameras[0].cfg.height
    W = cameras[0].cfg.width

    print(f"[INFO] Collecting {args_cli.num_frames} frames × {num_envs} envs × {len(CAMERA_NAMES)} cameras")
    print(f"[INFO] Resolution : {H} × {W}   format: {args_cli.save_format}")
    print(f"[INFO] Output dir : {os.path.abspath(args_cli.output_dir)}")

    # ---- Output setup ----------------------------------------------------
    os.makedirs(args_cli.output_dir, exist_ok=True)
    use_hdf5 = args_cli.save_format in ("hdf5", "both")
    use_images = args_cli.save_format in ("images", "both")
    hdf5_file = None

    if use_hdf5:
        import h5py  # noqa: F401 — just verify it is installed before starting sim
        hdf5_path = os.path.join(args_cli.output_dir, "camera_data.hdf5")
        hdf5_file = _create_hdf5(hdf5_path, num_envs, args_cli.num_frames, H, W, cameras)
        print(f"[INFO] HDF5 file  : {hdf5_path}")
    if use_images:
        _setup_image_dirs(args_cli.output_dir, num_envs)
        print(f"[INFO] Image dirs : {args_cli.output_dir}/env_XXXX/<cam>/{{rgb,depth,depth_vis}}")

    # ---- Camera extrinsics (world poses) --------------------------------
    # Capture after reset + a render so pos_w/quat are populated. These let the
    # 3D viewer fuse the per-camera clouds into one world/env-local frame.
    env.reset()
    env.unwrapped.sim.render()
    env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
    env_origin0 = env.unwrapped.scene.env_origins[0].cpu().numpy()
    extrinsics = {name: _get_extrinsics(cam, env_origin0) for name, cam in zip(CAMERA_NAMES, cameras)}
    if use_hdf5:
        extr_grp = hdf5_file["metadata"].create_group("extrinsics")
        for name in CAMERA_NAMES:
            extr_grp.create_dataset(name, data=extrinsics[name].astype(np.float32))
        print("[INFO] Wrote camera extrinsics to /metadata/extrinsics")
    if use_images:
        import json

        meta_json = {
            "cameras": CAMERA_NAMES,
            "image_height": H,
            "image_width": W,
            "intrinsics": {n: _get_intrinsics(c).tolist() for n, c in zip(CAMERA_NAMES, cameras)},
            "extrinsics": {n: extrinsics[n].tolist() for n in CAMERA_NAMES},
        }
        with open(os.path.join(args_cli.output_dir, "camera_meta.json"), "w") as jf:
            json.dump(meta_json, jf, indent=2)
        print(f"[INFO] Wrote camera_meta.json (intrinsics + extrinsics) to {args_cli.output_dir}")

    # ---- Simulation loop ------------------------------------------------
    frame_idx = 0
    frames_collected = 0

    while simulation_app.is_running() and frames_collected < args_cli.num_frames:
        with torch.inference_mode():
            # Random actions (replace with your policy here)
            actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            obs, _, terminated, truncated, _ = env.step(actions)

        # Skip the very first frame — cameras may not have rendered yet
        if frame_idx == 0:
            frame_idx += 1
            continue

        # -- Read and save each camera -----------------------------------
        for cam_name, camera in zip(CAMERA_NAMES, cameras):
            cam_out = camera.data.output

            # Safety check: skip if data not yet populated
            if "rgb" not in cam_out or cam_out["rgb"] is None:
                continue

            rgb_np   = _to_numpy_rgb(cam_out["rgb"])            # (N, H, W, 3) uint8
            depth_np = _to_numpy_depth(cam_out["distance_to_image_plane"])  # (N, H, W) float32

            if use_hdf5:
                _write_hdf5_frame(hdf5_file, frames_collected, cam_name, rgb_np, depth_np)
            if use_images:
                _write_image_frame(args_cli.output_dir, frames_collected, cam_name, rgb_np, depth_np)

        frames_collected += 1
        frame_idx += 1

        if frames_collected % 10 == 0:
            print(f"[INFO] Collected {frames_collected}/{args_cli.num_frames} frames …")

    # ---- Teardown --------------------------------------------------------
    print(f"[INFO] Done — {frames_collected} frames saved.")
    if hdf5_file is not None:
        # Trim datasets if we collected fewer frames than requested
        if frames_collected < args_cli.num_frames:
            print(f"[WARN] Trimming HDF5 datasets from {args_cli.num_frames} → {frames_collected} frames")
            for env_idx in range(num_envs):
                for cam_name in CAMERA_NAMES:
                    for key in ("rgb", "depth"):
                        ds = hdf5_file[f"env_{env_idx:04d}/{cam_name}/{key}"]
                        data = ds[:frames_collected]
                        del hdf5_file[f"env_{env_idx:04d}/{cam_name}/{key}"]
                        hdf5_file[f"env_{env_idx:04d}/{cam_name}/{key}"] = data
        hdf5_file.close()
        print(f"[INFO] HDF5 closed: {os.path.join(args_cli.output_dir, 'camera_data.hdf5')}")

    env.close()


# ---------------------------------------------------------------------------
# Quick dataset reader (importable utility)
# ---------------------------------------------------------------------------


def load_hdf5_footage(hdf5_path: str, env_idx: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Load camera footage for one environment from an HDF5 file.

    Returns a dict:
        {
          "wrist_cam":      {"rgb": (T, H, W, 3) uint8, "depth": (T, H, W) float32},
          "table_top_cam":  {...},
          "front_cam":      {...},
        }

    Example::

        data = load_hdf5_footage("datasets/camera_footage/camera_data.hdf5", env_idx=0)
        rgb_frame_0 = data["wrist_cam"]["rgb"][0]   # first frame, wrist camera
    """
    import h5py

    result = {}
    with h5py.File(hdf5_path, "r") as f:
        env_key = f"env_{env_idx:04d}"
        for cam_name in f[env_key].keys():
            result[cam_name] = {
                "rgb":   f[env_key][cam_name]["rgb"][:],
                "depth": f[env_key][cam_name]["depth"][:],
            }
    return result


if __name__ == "__main__":
    main()
    simulation_app.close()
