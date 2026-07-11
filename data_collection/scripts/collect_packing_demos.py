#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
collect_packing_demos.py — autonomous data collection for the packing task.

Pipeline per episode
--------------------
For each object, strictly one at a time (object_01 → object_02 → object_03):
  1. Refresh cameras, then query GraspGenX ZMQ server for grasp candidates.
  2. Filter grasps by confidence and select the best one.
  3. Plan pick motion with CuRobo (approach to grasp pose, fingers open).
  4. Execute waypoints (one per env step — plans are interpolated at env.step_dt,
     so planned timing is reproduced), settle, close gripper. No grasp
     verification — the flow proceeds straight to placing.
  5. Plan place motion with CuRobo (carry to above bin, fingers closed).
  6. Execute waypoints, open gripper, record data.
  7. Plan a joint-space motion back to the home configuration and execute it,
     so the next object is approached from a known start pose.

Every waypoint execution step is logged at DEBUG level to the .log file beside the
HDF5 output: commanded joint targets, raw env action, measured joints, tracking
error, commanded EE position (cuRobo FK) and measured EE position. Plan-level
stats (waypoint count, duration, implied max joint velocity) are logged at INFO.
If the env auto-resets mid-episode (safety time_out), the episode is aborted and
marked unsuccessful.

Usage
-----
Run via isaaclab.sh so that the correct Python and Isaac Sim paths are on PATH:

  ./isaaclab.sh -p data_collection/scripts/collect_packing_demos.py \
      --num_episodes 50 \
      --output datasets/packing_demos.hdf5 \
      --headless

Arguments
---------
  --num_episodes     Number of episodes to collect (default: 10).
  --output           HDF5 output path (default: datasets/packing_demos.hdf5).
  --headless         Run without GUI (recommended for large-scale collection).
  --grasp_threshold  Minimum GraspGenX confidence to accept (default: 0.5).
  --settle_steps     Physics settle steps after reset (default: 20).
  --gripper_steps    Steps to hold gripper open/close command (default: 15).
  --grasp_host       GraspGenX ZMQ server host (default: localhost).
  --grasp_port       GraspGenX ZMQ server port (default: 5556).
  --grasp_topk       Top-K grasps to request from server (default: 10).
  --object_placement 'random' (default): object x/y positions are randomized on
                     the table each reset; 'fixed': objects spawn at the env cfg
                     init_state poses every reset. In BOTH modes objects spawn
                     upright/standing (easiest geometry for GraspGenX).
  --debug            Deprecated alias for --object_placement fixed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

# Force line-buffered stdout so output appears even through pipes / conda run.
# reconfigure() is safe — it does NOT replace the stream object, so Isaac Sim's
# internal print() calls still work on the same stream.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Make 'data_collection' importable regardless of working directory.
# Script lives at <isaaclab_root>/data_collection/scripts/collect_packing_demos.py
# so going up three levels (.../scripts/ -> .../data_collection/ -> <root>) gives us the root.
_ISAACLAB_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ISAACLAB_ROOT))


# Handler references kept so we can re-attach them after Isaac Sim boots —
# omni.log installs its own Python logging bridge and can reconfigure the root logger.
_FILE_HANDLER: logging.FileHandler | None = None
_CONSOLE_HANDLER: logging.StreamHandler | None = None


def _tee_console_to_file(log_path: Path) -> None:
    """Mirror EVERYTHING written to stdout/stderr into <output>.console.log.

    Python logging handlers only see logging.* calls. Isaac Sim's simulator output
    (carb/omni logs) is written by C++ code directly to file descriptors 1/2 and
    bypasses Python entirely — same for raw print() calls from third-party code.
    This duplicates the fds through a pipe so every byte that reaches the terminal
    is also appended to the console log.
    """
    console_path = log_path.with_suffix(".console.log")
    log_file = open(console_path, "wb", buffering=0)

    def _tee_fd(fd: int) -> None:
        orig_fd = os.dup(fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)

        def _pump() -> None:
            while True:
                try:
                    data = os.read(read_fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                os.write(orig_fd, data)
                log_file.write(data)

        threading.Thread(target=_pump, daemon=True, name=f"console-tee-fd{fd}").start()

    _tee_fd(1)
    _tee_fd(2)


def _setup_logging(log_path: Path) -> None:
    """Write INFO+ logs to console, DEBUG+ to a .log file, and mirror the raw
    console (including Isaac Sim's own output) to a .console.log file."""
    global _FILE_HANDLER, _CONSOLE_HANDLER

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — captures everything including tracebacks
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    _FILE_HANDLER = fh

    # Console handler — mirrors to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    _CONSOLE_HANDLER = ch

    # Raw console mirror (Isaac Sim / carb / print() output included)
    _tee_console_to_file(log_path)

    # Redirect uncaught exceptions to the log file
    def _exc_handler(exc_type, exc_value, exc_tb):
        try:
            logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        except Exception:
            pass  # logging may already be torn down during Isaac Sim shutdown
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _exc_handler


def _reassert_logging() -> None:
    """Ensure our handlers survived Isaac Sim's boot.

    Isaac Sim bridges Python logging into omni.log at app startup and can drop or
    reconfigure root handlers/levels — that is how per-step DEBUG lines silently
    stop reaching the .log file. Call this once right after AppLauncher returns.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if _FILE_HANDLER is not None and _FILE_HANDLER not in root.handlers:
        root.addHandler(_FILE_HANDLER)
        logging.info("Re-attached file log handler (Isaac Sim boot had removed it).")
    if _CONSOLE_HANDLER is not None and _CONSOLE_HANDLER not in root.handlers:
        root.addHandler(_CONSOLE_HANDLER)

# --------------------------------------------------------------------------
# Parse args BEFORE launching IsaacSim so AppLauncher can consume --headless
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Packing demonstration collector")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--output", type=str, default="datasets/packing_demos.hdf5")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--grasp_threshold", type=float, default=0.5)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--gripper_steps", type=int, default=15)
parser.add_argument("--grasp_host", type=str, default="localhost",
                    help="GraspGenX ZMQ server hostname")
parser.add_argument("--grasp_port", type=int, default=5556,
                    help="GraspGenX ZMQ server port")
parser.add_argument("--grasp_topk", type=int, default=10,
                    help="Top-K grasps to request from GraspGenX server")
parser.add_argument("--cameras", type=str,
                    default="table_top_cam,table_side_cam_1,table_side_cam_2",
                    help="Comma-separated scene cameras whose depth clouds are fused "
                         "into one 3D point cloud for GraspGenX. Default is the three "
                         "table-focused cameras (top + both lateral views), which give "
                         "GraspGenX complete object geometry instead of just the top "
                         "surface. Available: wrist_cam, table_top_cam, front_cam, "
                         "table_side_cam_1, table_side_cam_2.")
parser.add_argument("--filter_radius", type=float, default=0.15,
                    help="Radius (m) around each object centre used to segment "
                         "the depth point cloud before sending to GraspGenX. "
                         "Keep this just larger than the biggest object (~0.12-0.20); "
                         "large values pull in the table and neighbouring objects, and "
                         "GraspGenX will then propose grasps on those instead.")
parser.add_argument("--table_margin", type=float, default=0.015,
                    help="Height band (m) above the lowest point of each object crop that "
                         "is removed as table surface before querying GraspGenX. The table "
                         "disc otherwise attracts high-confidence horizontal rim grasps "
                         "around the object that always fail cuRobo IK. <= 0 disables.")
parser.add_argument("--object_placement", type=str, default="random",
                    choices=["random", "fixed"],
                    help="Where objects spawn at every reset. 'random': x/y position is "
                         "randomized on the table surface (orientation stays upright — the "
                         "reset event samples roll/pitch/yaw of zero, so objects always "
                         "STAND, which gives GraspGenX the easiest grasp geometry). "
                         "'fixed': objects spawn at the upright init_state poses defined "
                         "in the env cfg, identical every episode.")
parser.add_argument("--debug", action="store_true",
                    help="Deprecated alias for --object_placement fixed.")
args = parser.parse_args()
if args.debug:
    args.object_placement = "fixed"

# Set up logging as early as possible — before Isaac Sim boots so we capture
# any import-time warnings too.
_log_path = Path(args.output).with_suffix(".log")
_setup_logging(_log_path)
logging.info("collect_packing_demos starting — log: %s", _log_path)
logging.info("Args: %s", vars(args))

# --------------------------------------------------------------------------
# Bootstrap IsaacSim — must happen before any omni / isaaclab imports
# --------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

# Isaac Sim's omni.log bridge may have reconfigured Python logging during boot —
# restore our file/console handlers so DEBUG traces keep flowing to the .log file.
_reassert_logging()

# --------------------------------------------------------------------------
# Normal imports (IsaacSim is now running)
# --------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.envs.manager_based_env import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg  # noqa: E402
from isaaclab.sensors import FrameTransformer  # noqa: E402

# Register the packing env and its config
import isaaclab_tasks.manager_based.isaaclab_int  # noqa: E402, F401
import isaaclab_tasks.manager_based.isaaclab_int.config.franka  # noqa: E402, F401
from isaaclab_tasks.manager_based.isaaclab_int.mdp import franka_pack_events  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from data_collection.grasp_client import GraspResult, GraspGenXDepthClient, GraspGenXIsaacClient, filter_grasps  # noqa: E402
from data_collection.motion_planning import PackMotionPlanner  # noqa: E402
from data_collection.data_recording import HDF5EpisodeRecorder  # noqa: E402


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ENV_ID = "Isaac-Pack-Object-Franka-Camera-v0"
GRIPPER_OPEN_CMD: float = 1.0
GRIPPER_CLOSE_CMD: float = -1.0

# Joint names that are arm joints (not finger joints)
ARM_JOINT_PATTERN = "panda_joint"
FINGER_JOINT_PATTERN = "panda_finger"

# Object sequence — process in this order every episode
OBJECT_NAMES = ["object_01", "object_02", "object_03"]


# --------------------------------------------------------------------------
# GraspGenX client (created once in main(), shared across all episodes)
# --------------------------------------------------------------------------
# grasp_client is set in main() to a GraspGenXDepthClient instance.
grasp_client: GraspGenXDepthClient | None = None


# --------------------------------------------------------------------------
# Action helpers
# --------------------------------------------------------------------------

# Offset/scale ACTUALLY used by the arm JointPositionAction term, read from the
# action manager in main(). The term caches its offset (default_joint_pos.clone())
# at construction, but the env's reset event `set_default_joint_pose` REPLACES
# robot.data.default_joint_pos afterwards — so inverting with the live default
# shifts every command by (cached - event) per joint. On this env panda_joint6 is
# off by 3.037 - 2.3775 = 0.6595 rad: the constant "arm did not converge (0.6595
# rad)" error — the arm tracked fine, it was simply sent the wrong target.
_ARM_OFFSET: torch.Tensor | None = None  # (num_arm_joints,) on env device
_ARM_SCALE: float = 0.5


def _read_arm_action_transform(env: ManagerBasedEnv, arm_joint_ids: list[int]) -> None:
    """Read the offset/scale the arm action term really uses and log any mismatch
    with the live default joint pose (which the reset event rewrites)."""
    global _ARM_OFFSET, _ARM_SCALE

    term = env.action_manager.get_term("arm_action")
    offset = term._offset
    if isinstance(offset, torch.Tensor):
        _ARM_OFFSET = offset[0].detach().clone().to(env.device)
    else:  # plain float offset
        _ARM_OFFSET = torch.full((len(arm_joint_ids),), float(offset), device=env.device)

    scale = term._scale
    _ARM_SCALE = float(scale) if not isinstance(scale, torch.Tensor) else float(scale.reshape(-1)[0])

    robot: Articulation = env.scene["robot"]
    live_default = robot.data.default_joint_pos[0, arm_joint_ids].to(env.device)
    mismatch = (_ARM_OFFSET - live_default).abs()
    logging.info(
        "Arm action transform: scale=%.3f | term offset=%s | live default_joint_pos=%s",
        _ARM_SCALE, _fmt(_ARM_OFFSET.cpu()), _fmt(live_default.cpu()),
    )
    if mismatch.max().item() > 1e-3:
        logging.info(
            "Action offset differs from live default by up to %.4f rad (reset event rewrote "
            "default_joint_pos) — inverting with the term's cached offset, worst joint: %s",
            mismatch.max().item(),
            robot.data.joint_names[arm_joint_ids[int(mismatch.argmax())]],
        )


def _make_action(
    robot: Articulation,
    arm_target_q: torch.Tensor,
    gripper_cmd: float,
    arm_joint_ids: list[int],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert absolute arm joint positions + gripper binary command to env action.

    The packing env uses JointPositionActionCfg(scale=0.5, use_default_offset=True),
    which applies: target_q = offset + scale * action, where offset was cloned from
    default_joint_pos at action-term construction. We must invert with that SAME
    cached offset (read in _read_arm_action_transform) — NOT the live
    default_joint_pos, which the env's reset event overwrites (see _ARM_OFFSET note).

    Args:
        robot:          Robot articulation (fallback for the offset).
        arm_target_q:   (num_arm_joints,) desired arm joint positions.
        gripper_cmd:    GRIPPER_OPEN_CMD or GRIPPER_CLOSE_CMD.
        arm_joint_ids:  Indices of arm joints within robot.data.joint_pos.
        device:         Target device for the action tensor.

    Returns:
        (1, num_arm_joints + 1) action tensor ready for env.step().
    """
    if _ARM_OFFSET is not None:
        offset = _ARM_OFFSET.to(device=device)
        scale = _ARM_SCALE
    else:  # fallback (main() not initialized yet) — original, possibly-stale inversion
        offset = robot.data.default_joint_pos[0, arm_joint_ids].to(device=device)
        scale = 0.5
    # CuRobo's get_full_js() returns all joints (arm + fingers); slice to arm only.
    arm_target_on_dev = arm_target_q[:len(arm_joint_ids)].to(device=device, dtype=torch.float32)
    action_arm = (arm_target_on_dev - offset) / scale
    action_gripper = torch.tensor([gripper_cmd], device=device, dtype=torch.float32)
    return torch.cat([action_arm, action_gripper]).unsqueeze(0)  # (1, 8)


def _get_joint_ids(robot: Articulation) -> tuple[list[int], list[int]]:
    """Return (arm_joint_ids, finger_joint_ids) as index lists."""
    arm_ids = [
        i for i, name in enumerate(robot.data.joint_names)
        if ARM_JOINT_PATTERN in name and FINGER_JOINT_PATTERN not in name
    ]
    finger_ids = [
        i for i, name in enumerate(robot.data.joint_names)
        if FINGER_JOINT_PATTERN in name
    ]
    return arm_ids, finger_ids


# --------------------------------------------------------------------------
# Data snapshot helper
# --------------------------------------------------------------------------

def _snapshot(
    robot: Articulation,
    ee_frame: FrameTransformer,
    env: ManagerBasedEnv,
    env_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, tuple]:
    """
    Read current sim state for one timestep.

    Returns:
        joint_pos  (9,)
        ee_pos     (3,)
        ee_quat    (4,)
        obj_poses  dict: object_name -> (pos (3,), quat (4,))
        bin_pose   (pos (3,), quat (4,))
    """
    joint_pos = robot.data.joint_pos[env_id].detach().cpu()

    ee_pos = (ee_frame.data.target_pos_w[env_id, 0, :] - env.scene.env_origins[env_id]).detach().cpu()
    ee_quat = ee_frame.data.target_quat_w[env_id, 0, :].detach().cpu()

    obj_poses: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for obj_name in OBJECT_NAMES:
        obj = env.scene[obj_name]
        pos = (obj.data.root_pos_w[env_id] - env.scene.env_origins[env_id]).detach().cpu()
        quat = obj.data.root_quat_w[env_id].detach().cpu()
        obj_poses[obj_name] = (pos, quat)

    bin_obj = env.scene["packing_bin"]
    bin_pos = (bin_obj.data.root_pos_w[env_id] - env.scene.env_origins[env_id]).detach().cpu()
    bin_quat = bin_obj.data.root_quat_w[env_id].detach().cpu()

    return joint_pos, ee_pos, ee_quat, obj_poses, (bin_pos, bin_quat)


# --------------------------------------------------------------------------
# Sim stepping helpers
# --------------------------------------------------------------------------

def _fmt(t: torch.Tensor) -> str:
    """Compact fixed-precision string for a 1-D tensor, for log lines."""
    return "[" + ", ".join(f"{v:+.4f}" for v in t.reshape(-1).tolist()) + "]"


def _step_env(env: ManagerBasedEnv, action: torch.Tensor) -> bool:
    """Step the environment and report whether it auto-reset.

    ManagerBasedRLEnv resets a terminated/truncated env *inside* step(). If that
    happens mid-execution the robot teleports to its default pose and the current
    plan (and episode recording) is void — the caller must abort.

    Returns:
        True if the env auto-reset during this step.
    """
    _, _, terminated, truncated, _ = env.step(action)
    return bool(terminated[0].item()) or bool(truncated[0].item())


def _refresh_cameras(env: ManagerBasedEnv) -> None:
    """Force a render + sensor update so depth queries see the current scene.

    In headless mode the physics-settle loop runs with render=False, leaving the
    camera buffers stale (from the reset re-renders). Call this right before every
    GraspGenX query.
    """
    env.sim.render()
    env.scene.update(dt=env.physics_dt)


def _log_plan_stats(label: str, waypoints: list[torch.Tensor], step_dt: float) -> None:
    """Log waypoint count, duration, and implied joint velocity of a plan.

    The executor consumes one waypoint per env step (period ``step_dt``), so
    max |dq| / step_dt is the joint velocity the PD controller is asked to track.
    Franka joint velocity limits are ~2.2-2.6 rad/s — implied velocities well above
    that mean the plan timing and execution rate are mismatched.
    """
    if len(waypoints) < 2:
        logging.info("    [%s] plan has %d waypoint(s)", label, len(waypoints))
        return
    wps = torch.stack([wp.detach().cpu() for wp in waypoints])
    deltas = (wps[1:] - wps[:-1]).abs()
    max_delta = deltas.max().item()
    logging.info(
        "    [%s] %d waypoints | duration %.2f s @ %.1f Hz | max |dq| %.4f rad/step -> implied max vel %.2f rad/s",
        label, len(wps), len(wps) * step_dt, 1.0 / step_dt, max_delta, max_delta / step_dt,
    )


# --------------------------------------------------------------------------
# Waypoint execution
# --------------------------------------------------------------------------

def _execute_waypoints(
    env: ManagerBasedEnv,
    robot: Articulation,
    ee_frame: FrameTransformer,
    waypoints: list[torch.Tensor],
    gripper_cmd: float,
    arm_joint_ids: list[int],
    recorder: HDF5EpisodeRecorder,
    grasp_confidence: float,
    grasp_object_idx: int,
    label: str = "",
    wp_ee_positions: list[torch.Tensor] | None = None,
    settle_tol: float = 0.01,
    max_settle_steps: int = 24,
) -> bool:
    """
    Step through CuRobo joint-position waypoints and record each step.

    One waypoint is consumed per env step; because the plan is interpolated at
    exactly ``env.step_dt``, this reproduces cuRobo's planned velocity profile.
    After the last waypoint the final target is held until the arm converges
    (max joint error < ``settle_tol``) or ``max_settle_steps`` is exhausted, so
    the EE has actually arrived before the gripper is toggled.

    Args:
        waypoints:        List of (num_arm_joints,) tensors from PackMotionPlanner.
        gripper_cmd:      GRIPPER_OPEN_CMD or GRIPPER_CLOSE_CMD to hold during execution.
        grasp_confidence: Meta value to stamp on each recorded step.
        grasp_object_idx: Meta value to stamp on each recorded step.
        label:            Tag for log lines (e.g. "pick:object_01").
        wp_ee_positions:  Optional per-waypoint commanded EE positions (from FK)
                          logged alongside the measured EE position.

    Returns:
        True if all waypoints executed; False if the env auto-reset mid-execution
        (episode must be aborted).
    """
    if not waypoints:
        logging.warning("    [%s] empty waypoint list — nothing to execute.", label)
        return True

    num_arm = len(arm_joint_ids)

    # The plan must start where the robot currently is; a large gap here means the
    # plan was computed from a stale/wrong start state and execution will jump.
    q_now = robot.data.joint_pos[0, arm_joint_ids].detach().cpu()
    start_gap = (waypoints[0].detach().cpu()[:num_arm] - q_now).abs().max().item()
    logging.debug("[%s] start check: q_now=%s | wp[0]=%s | gap=%.4f rad", label, _fmt(q_now),
                  _fmt(waypoints[0].detach().cpu()[:num_arm]), start_gap)
    if start_gap > 0.05:
        logging.warning(
            "    [%s] plan starts %.4f rad away from current joint state — "
            "plan start state is stale or wrong.", label, start_gap,
        )

    diverged = False

    def _record() -> None:
        joint_pos, ee_pos, ee_quat, obj_poses, bin_pose = _snapshot(robot, ee_frame, env)
        recorder.record_step(
            joint_pos=joint_pos,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            obj_poses=obj_poses,
            bin_pose=bin_pose,
            grasp_confidence=grasp_confidence,
            grasp_object_idx=grasp_object_idx,
        )

    for i, wp in enumerate(waypoints):
        action = _make_action(
            robot=robot,
            arm_target_q=wp,
            gripper_cmd=gripper_cmd,
            arm_joint_ids=arm_joint_ids,
            device=env.device,
        )
        env_reset = _step_env(env, action)
        _record()

        # Per-step debug log: commanded joints, raw action, measured joints,
        # tracking error, commanded EE (FK) and measured EE.
        q_cmd = wp.detach().cpu()[:num_arm]
        q_meas = robot.data.joint_pos[0, arm_joint_ids].detach().cpu()
        track_err = (q_cmd - q_meas).abs().max().item()
        ee_meas = (ee_frame.data.target_pos_w[0, 0, :] - env.scene.env_origins[0]).detach().cpu()
        ee_cmd = wp_ee_positions[i] if wp_ee_positions is not None and i < len(wp_ee_positions) else None
        logging.debug(
            "[%s] wp %03d/%03d | q_cmd=%s | action=%s | q_meas=%s | max_err=%.4f rad | ee_cmd=%s | ee_meas=%s",
            label, i + 1, len(waypoints),
            _fmt(q_cmd), _fmt(action[0].detach().cpu()), _fmt(q_meas), track_err,
            _fmt(ee_cmd) if ee_cmd is not None else "n/a", _fmt(ee_meas),
        )

        # Flag divergence as soon as it happens (once), not just at the end —
        # the waypoint index tells us where the arm stopped following the plan
        # (collision, joint limit, or unreachable command).
        if track_err > 0.3 and not diverged:
            diverged = True
            logging.warning(
                "    [%s] tracking error %.3f rad at waypoint %d/%d | worst joint: %s | q_cmd=%s | q_meas=%s",
                label, track_err, i + 1, len(waypoints),
                robot.data.joint_names[arm_joint_ids[int((q_cmd - q_meas).abs().argmax())]],
                _fmt(q_cmd), _fmt(q_meas),
            )

        if env_reset:
            logging.warning(
                "    [%s] env auto-reset at waypoint %d/%d — aborting episode.", label, i + 1, len(waypoints)
            )
            return False

    # Settle on the final waypoint: hold the target until the arm converges.
    final_wp = waypoints[-1]
    final_target = final_wp.detach().cpu()[:num_arm]
    settle_steps = 0
    for _ in range(max_settle_steps):
        q_meas = robot.data.joint_pos[0, arm_joint_ids].detach().cpu()
        if (final_target - q_meas).abs().max().item() < settle_tol:
            break
        action = _make_action(
            robot=robot,
            arm_target_q=final_wp,
            gripper_cmd=gripper_cmd,
            arm_joint_ids=arm_joint_ids,
            device=env.device,
        )
        env_reset = _step_env(env, action)
        _record()
        settle_steps += 1
        if env_reset:
            logging.warning("    [%s] env auto-reset while settling — aborting episode.", label)
            return False

    q_meas = robot.data.joint_pos[0, arm_joint_ids].detach().cpu()
    final_err = (final_target - q_meas).abs().max().item()
    ee_meas = (ee_frame.data.target_pos_w[0, 0, :] - env.scene.env_origins[0]).detach().cpu()
    logging.info(
        "    [%s] done: %d waypoints + %d settle steps | final joint err %.4f rad | ee_meas=%s",
        label, len(waypoints), settle_steps, final_err, _fmt(ee_meas),
    )
    if final_err > settle_tol:
        logging.warning(
            "    [%s] arm did not converge (%.4f rad > %.4f) — PD tracking is lagging.",
            label, final_err, settle_tol,
        )
    return True


def _hold_gripper(
    env: ManagerBasedEnv,
    robot: Articulation,
    ee_frame: FrameTransformer,
    arm_joint_ids: list[int],
    gripper_cmd: float,
    steps: int,
    recorder: HDF5EpisodeRecorder,
    grasp_confidence: float,
    grasp_object_idx: int,
    finger_joint_ids: list[int] | None = None,
    label: str = "gripper",
) -> bool:
    """Hold the current arm pose while toggling the gripper.

    Returns:
        True if all steps executed; False if the env auto-reset mid-hold.
    """
    current_arm_q = robot.data.joint_pos[0, arm_joint_ids].detach().clone()
    for k in range(steps):
        action = _make_action(
            robot=robot,
            arm_target_q=current_arm_q,
            gripper_cmd=gripper_cmd,
            arm_joint_ids=arm_joint_ids,
            device=env.device,
        )
        env_reset = _step_env(env, action)

        joint_pos, ee_pos, ee_quat, obj_poses, bin_pose = _snapshot(robot, ee_frame, env)
        recorder.record_step(
            joint_pos=joint_pos,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            obj_poses=obj_poses,
            bin_pose=bin_pose,
            grasp_confidence=grasp_confidence,
            grasp_object_idx=grasp_object_idx,
        )

        if finger_joint_ids is not None:
            fingers = robot.data.joint_pos[0, finger_joint_ids].detach().cpu()
            logging.debug("[%s] step %02d/%02d | cmd=%+.1f | fingers=%s", label, k + 1, steps, gripper_cmd, _fmt(fingers))

        if env_reset:
            logging.warning("    [%s] env auto-reset while toggling gripper — aborting episode.", label)
            return False

    if finger_joint_ids is not None:
        fingers = robot.data.joint_pos[0, finger_joint_ids].detach().cpu()
        logging.info("    [%s] final finger positions: %s", label, _fmt(fingers))
    return True


# --------------------------------------------------------------------------
# Tracking configuration enforcement
# --------------------------------------------------------------------------

def _enforce_tracking_gains(robot: Articulation, arm_joint_ids: list[int]) -> None:
    """Verify the PD gains and gravity flags that actually reached PhysX, and
    enforce motion-planner-grade tracking if the env cfg delivered soft values.

    Guards against the failure seen in the 2026-07-02 run: soft FRANKA_PANDA_CFG
    gains (kp=80, kd=4) with gravity enabled produce a steady-state elbow sag of
    tau_gravity / kp ~= 0.65 rad, so the arm can never reach cuRobo's waypoints no
    matter how good the plan is (constant "arm did not converge (0.6595 rad)").
    FRANKA_PANDA_HIGH_PD_CFG is supposed to deliver kp=400/kd=80 with gravity
    disabled at spawn; this function makes that guaranteed at runtime and logs
    what the env cfg actually produced.
    """
    view = robot.root_physx_view
    kp = view.get_dof_stiffnesses()[0]
    kd = view.get_dof_dampings()[0]
    logging.info(
        "PhysX arm gains at startup: kp=%s | kd=%s",
        [round(float(kp[i]), 1) for i in arm_joint_ids],
        [round(float(kd[i]), 1) for i in arm_joint_ids],
    )

    arm_kp_min = min(float(kp[i]) for i in arm_joint_ids)
    if arm_kp_min < 300.0:
        logging.warning(
            "Soft arm gains reached the sim (min kp=%.0f) — the HIGH_PD env cfg did not "
            "apply; writing kp=400, kd=80 to PhysX for waypoint tracking.",
            arm_kp_min,
        )
        robot.write_joint_stiffness_to_sim(400.0, joint_ids=arm_joint_ids)
        robot.write_joint_damping_to_sim(80.0, joint_ids=arm_joint_ids)

    # Stiff PD tracking assumes no gravity load on the links (same choice
    # FRANKA_PANDA_HIGH_PD_CFG makes at spawn time via disable_gravity=True).
    grav = view.get_disable_gravities()  # (num_articulations, max_links), 1 = gravity off
    grav_t = torch.as_tensor(grav).reshape(view.count, -1)
    if not bool(grav_t.all()):
        num_affected = int((grav_t == 0).sum().item())
        logging.warning(
            "Gravity is enabled on %d robot links — disabling it for tracking "
            "(matches FRANKA_PANDA_HIGH_PD_CFG).",
            num_affected,
        )
        all_off = torch.ones_like(grav_t)
        indices = torch.arange(view.count, dtype=torch.int32, device=all_off.device)
        view.set_disable_gravities(all_off, indices)

    # Re-read so the log records the effective values used for the run.
    kp_after = view.get_dof_stiffnesses()[0]
    logging.info(
        "Effective arm gains for this run: kp=%s | gravity disabled on all robot links: %s",
        [round(float(kp_after[i]), 1) for i in arm_joint_ids],
        bool(torch.as_tensor(view.get_disable_gravities()).all()),
    )


# --------------------------------------------------------------------------
# Home return
# --------------------------------------------------------------------------

def _return_home(
    env: ManagerBasedEnv,
    robot: Articulation,
    ee_frame: FrameTransformer,
    planner,
    arm_joint_ids: list[int],
    arm_joint_names: list[str],
    recorder: HDF5EpisodeRecorder,
) -> bool:
    """Plan and execute a joint-space motion back to the robot's home configuration.

    Called after every place (and after failed grasps) so each pick starts from the
    same known configuration and the arm is clear of the cameras' workspace view.

    Returns:
        True if the robot reached home (or planning failed benignly and we continue
        from the current pose); False if the env auto-reset during execution.
    """
    home_q = robot.data.default_joint_pos[0].detach().clone()
    if not planner.plan_home(home_q, list(robot.data.joint_names)):
        logging.warning("    CuRobo home planning failed — continuing from current pose.")
        return True
    waypoints = planner.get_arm_waypoints(arm_joint_names)
    _log_plan_stats("home", waypoints, env.step_dt)
    return _execute_waypoints(
        env=env,
        robot=robot,
        ee_frame=ee_frame,
        waypoints=waypoints,
        gripper_cmd=GRIPPER_OPEN_CMD,
        arm_joint_ids=arm_joint_ids,
        recorder=recorder,
        grasp_confidence=0.0,
        grasp_object_idx=0,
        label="home",
        wp_ee_positions=planner.get_waypoint_ee_positions(),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    global grasp_client

    # ---- environment ----
    env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=1)

    # The script owns the episode lifecycle (env.reset() at the top of each episode).
    # Disable every termination that would make ManagerBasedRLEnv auto-reset mid-episode
    # (teleporting the robot while waypoints are still being executed), and give the
    # safety time_out enough headroom for 3x (pick + place + home) at 24 Hz.
    env_cfg.episode_length_s = 120.0
    env_cfg.terminations.object_1_dropping = None
    env_cfg.terminations.object_2_dropping = None
    env_cfg.terminations.object_3_dropping = None
    env_cfg.terminations.success = None

    # --object_placement fixed: pin objects to their init_state poses for reproducible
    # runs. The env's reset_objects_pose event (see franka_pack_joint_pos_env_cfg.
    # EventCfg) otherwise samples a fresh x/y position on the table for
    # object_01/02/03 at every reset — orientation is always upright/standing in BOTH
    # modes (the event samples roll/pitch/yaw of zero; the init_state rots are
    # identity).
    #
    # IMPORTANT: this event is the ONLY thing that ever calls write_root_pose_to_sim()
    # on these objects. RigidObject.reset() (invoked by scene.reset() inside every
    # env.reset()) only clears external-wrench buffers — it never rewrites root pose.
    # So simply deleting the event (env_cfg.events.reset_objects_pose = None) leaves
    # objects wherever physics put them at the end of the previous episode (e.g. still
    # sitting in the bin) — they'd never "reset" again after the very first episode.
    # Swap in reset_objects_to_default instead, which explicitly teleports each object
    # back to its own init_state pose every reset.
    if args.object_placement == "fixed":
        if getattr(env_cfg.events, "reset_objects_pose", None) is not None:
            env_cfg.events.reset_objects_pose = EventTerm(
                func=franka_pack_events.reset_objects_to_default,
                mode="reset",
                params={
                    "asset_cfgs": [
                        SceneEntityCfg("object_01"),
                        SceneEntityCfg("object_02"),
                        SceneEntityCfg("object_03"),
                    ],
                },
            )
            logging.info(
                "Object placement FIXED — objects reset to their init_state poses "
                "every episode."
            )
        else:
            logging.warning(
                "--object_placement fixed given but no 'reset_objects_pose' event was "
                "found on the env cfg; objects use whatever placement the env already defines."
            )
    else:
        logging.info(
            "Object placement RANDOM — x/y positions are randomized on the table each "
            "reset, orientation always upright/standing (pass --object_placement fixed "
            "to pin objects to their init_state poses)."
        )

    env: ManagerBasedEnv = gym.make(ENV_ID, cfg=env_cfg).unwrapped
    env.reset()

    # Env creation reconfigures Python logging again (omni bridge) — restore handlers
    # so INFO/DEBUG lines keep reaching the .log file.
    _reassert_logging()

    logging.info(
        "Timing: physics %.1f Hz | control (env.step) %.1f Hz | plan waypoint dt %.4f s",
        1.0 / env.physics_dt, 1.0 / env.step_dt, env.step_dt,
    )
    if args.filter_radius > 0.30:
        logging.warning(
            "filter_radius=%.2f m is large: the cropped cloud will include table and "
            "neighbouring-object points, and GraspGenX will propose grasps on those. "
            "Recommended: ~0.12-0.20 m (just larger than the biggest object).",
            args.filter_radius,
        )

    robot: Articulation = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    arm_joint_ids, finger_joint_ids = _get_joint_ids(robot)
    arm_joint_names = [robot.data.joint_names[i] for i in arm_joint_ids]

    # Verify (and if needed fix) the PD gains + gravity flags that actually reached
    # PhysX — soft gains sag under gravity and can never track cuRobo waypoints.
    _enforce_tracking_gains(robot, arm_joint_ids)

    # Read the offset/scale the arm action term actually applies. Inverting with the
    # live default_joint_pos is WRONG on this env: the reset event rewrites it after
    # the action term cached its offset (0.6595 rad shift on panda_joint6).
    _read_arm_action_transform(env, arm_joint_ids)

    # ---- GraspGenX depth client (primary) ----
    # Fuses the depth clouds of all requested cameras (default: top + two lateral
    # table cameras) into one world-frame cloud per grasp query, so GraspGenX sees
    # full 3D object geometry rather than a single viewpoint.
    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    missing_cams = [c for c in camera_names if c not in env.scene.keys()]
    if missing_cams:
        raise ValueError(
            f"--cameras contains unknown scene camera(s): {missing_cams}. "
            f"Check the env cfg (Isaac-Pack-Object-Franka-Camera-v0) camera names."
        )
    grasp_client = GraspGenXDepthClient(
        camera_names=camera_names,
        host=args.grasp_host,
        port=args.grasp_port,
        gripper_name="franka_panda",
        num_grasps=200,
        topk_num_grasps=args.grasp_topk,
        filter_radius=args.filter_radius,
        table_margin=args.table_margin,
        num_pc_points=4096,
    )
    # ---- GraspGenX USD mesh client (fallback for transparent/glass objects) ----
    isaac_client = GraspGenXIsaacClient(
        host=args.grasp_host,
        port=args.grasp_port,
        gripper_name="franka_panda",
        num_grasps=200,
        topk_num_grasps=args.grasp_topk,
    )
    logging.info(
        "Connected to GraspGenX server at %s:%s (fused cameras=%s, filter_radius=%.3f m)",
        args.grasp_host, args.grasp_port, ",".join(camera_names), args.filter_radius,
    )

    # ---- motion planner ----
    planner = PackMotionPlanner(env=env, robot=robot, env_id=0)

    # Surface cuRobo's planning-failure reasons (result.status, per-phase failures):
    # CuroboPlanner logs them at DEBUG on its own logger, which defaults to INFO.
    logging.getLogger("CuroboPlanner_0").setLevel(logging.DEBUG)

    # ---- data recorder ----
    output_path = Path(args.output)
    recorder = HDF5EpisodeRecorder(output_path)

    for ep in range(args.num_episodes):
        logging.info("=== Episode %d/%d ===", ep + 1, args.num_episodes)

        obs, _ = env.reset()

        # Let physics settle after reset
        for _ in range(args.settle_steps):
            env.sim.step(render=not args.headless)
        env.scene.update(env.step_dt)

        recorder.start_episode()
        num_packed = 0
        episode_aborted = False

        # One object at a time: query grasp -> pick -> place -> return home,
        # then move on to the next object from the same home configuration.
        for obj_idx, obj_name in enumerate(OBJECT_NAMES, start=1):
            logging.info("  Object %d/3: %s", obj_idx, obj_name)
            _reassert_logging()  # Isaac Sim strips our handlers during env creation/reset

            obj_pos = (env.scene[obj_name].data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu()

            # Objects that fell off the table (drop terminations are disabled) are
            # unreachable — don't waste a grasp query and a failed plan on them.
            if obj_pos[2] < 0.0:
                logging.warning("    %s has fallen off the table (z=%.3f) — skipping.", obj_name, obj_pos[2])
                continue

            # -- 1. Query GraspGenX server (depth first, USD mesh fallback for glass/transparent) --
            # Fresh render first: in headless mode the camera buffers are otherwise stale.
            _refresh_cameras(env)
            raw_grasps = grasp_client.query(obj_name, env, env_id=0)
            if not raw_grasps:
                logging.info("    Depth client got 0 points for %s — falling back to USD mesh client.", obj_name)
                raw_grasps = isaac_client.query(obj_name, env, env_id=0)

            # Log where the candidates sit relative to the target object and how
            # top-down their approach directions are — filter_grasps() below uses
            # these same two metrics (downwardness, TCP-to-centroid distance) to
            # rank candidates, so this line shows the spread it's choosing from.
            if raw_grasps:
                dists = [float(torch.linalg.vector_norm(g.tcp_position() - obj_pos)) for g in raw_grasps]
                downs = [g.downwardness() for g in raw_grasps]
                logging.info(
                    "    Candidates for %s: %d raw | TCP dist to object root %s: min %.3f / mean %.3f m "
                    "| downwardness max %.2f / mean %.2f",
                    obj_name, len(raw_grasps), _fmt(obj_pos), min(dists), sum(dists) / len(dists),
                    max(downs), sum(downs) / len(downs),
                )

            best_grasps = filter_grasps(
                raw_grasps,
                confidence_threshold=args.grasp_threshold,
                top_k=1,
                object_pos=obj_pos,
            )

            if not best_grasps:
                logging.warning("    No grasp above threshold for %s (got %d raw), skipping.", obj_name, len(raw_grasps))
                continue

            best_grasp = best_grasps[0]
            logging.info(
                "    Grasp: conf %.3f | TCP dist to object %.3f m | downwardness %.2f | hand pos=%s tcp=%s quat=%s",
                best_grasp.confidence,
                float(torch.linalg.vector_norm(best_grasp.tcp_position() - obj_pos)),
                best_grasp.downwardness(),
                _fmt(best_grasp.position), _fmt(best_grasp.tcp_position()), _fmt(best_grasp.quaternion),
            )

            # Height audit (env-local == robot-base frame): compare the z of the
            # commanded hand-base pose, the fingertip TCP (hand base + 0.1034 m along
            # the grasp +z approach axis), and the object root. For a correct top-down
            # grasp the hand base sits ~0.10 m ABOVE the object while the TCP z lands
            # ON the object (tcp_gap ~ 0). A large |tcp_gap| means the grasp is placed
            # off the object vertically (height problem), not merely the expected
            # hand-base standoff. All three are in the same frame, so gaps are exact.
            hand_z = float(best_grasp.position[2])
            tcp_z = float(best_grasp.tcp_position()[2])
            obj_z = float(obj_pos[2])
            logging.debug(
                "    [height] %s | hand_base_z=%+.4f | tcp_z=%+.4f | object_root_z=%+.4f "
                "| hand_gap(hand-obj)=%+.4f m | tcp_gap(tcp-obj)=%+.4f m",
                obj_name, hand_z, tcp_z, obj_z, hand_z - obj_z, tcp_z - obj_z,
            )

            # -- 2. Plan pick --
            pick_ok = planner.plan_pick(best_grasp)
            if not pick_ok:
                logging.warning("    CuRobo pick planning failed for %s.", obj_name)
                continue

            pick_waypoints = planner.get_arm_waypoints(arm_joint_names)
            _log_plan_stats(f"pick:{obj_name}", pick_waypoints, env.step_dt)

            # -- 3. Execute pick (fingers open) --
            if not _execute_waypoints(
                env=env,
                robot=robot,
                ee_frame=ee_frame,
                waypoints=pick_waypoints,
                gripper_cmd=GRIPPER_OPEN_CMD,
                arm_joint_ids=arm_joint_ids,
                recorder=recorder,
                grasp_confidence=best_grasp.confidence,
                grasp_object_idx=obj_idx,
                label=f"pick:{obj_name}",
                wp_ee_positions=planner.get_waypoint_ee_positions(),
            ):
                episode_aborted = True
                break

            # Reach accuracy: measured panda_hand position vs the grasp target
            # (same frame — ee_frame target 0 is panda_hand with zero offset).
            ee_meas = (ee_frame.data.target_pos_w[0, 0, :] - env.scene.env_origins[0]).detach().cpu()
            reach_err = torch.linalg.vector_norm(ee_meas - best_grasp.position).item()
            logging.info("    Pick reach error: %.4f m (EE vs grasp target)", reach_err)

            # -- 4. Close gripper --
            if not _hold_gripper(
                env=env,
                robot=robot,
                ee_frame=ee_frame,
                arm_joint_ids=arm_joint_ids,
                gripper_cmd=GRIPPER_CLOSE_CMD,
                steps=args.gripper_steps,
                recorder=recorder,
                grasp_confidence=best_grasp.confidence,
                grasp_object_idx=obj_idx,
                finger_joint_ids=finger_joint_ids,
                label=f"close:{obj_name}",
            ):
                episode_aborted = True
                break

            # -- 4b. No grasp verification: the gripper closed on the planned
            # grasp pose, and we proceed straight to the place motion. Whether
            # the object actually travelled shows up in the recorded poses and
            # the release log below.
            fingers = robot.data.joint_pos[0, finger_joint_ids].detach().cpu()
            logging.info("    Gripper closed on %s (fingers=%s) — proceeding to place.", obj_name, _fmt(fingers))

            # -- 5. Plan place --
            place_ok = planner.plan_place(obj_name, slot_index=obj_idx - 1)
            if not place_ok:
                logging.warning("    CuRobo place planning failed for %s.", obj_name)
                # Open gripper, go home, continue to next object
                if not _hold_gripper(
                    env=env, robot=robot, ee_frame=ee_frame,
                    arm_joint_ids=arm_joint_ids,
                    gripper_cmd=GRIPPER_OPEN_CMD,
                    steps=args.gripper_steps,
                    recorder=recorder,
                    grasp_confidence=0.0, grasp_object_idx=0,
                    finger_joint_ids=finger_joint_ids,
                    label=f"open:{obj_name}",
                ) or not _return_home(env, robot, ee_frame, planner, arm_joint_ids, arm_joint_names, recorder):
                    episode_aborted = True
                    break
                continue

            place_waypoints = planner.get_arm_waypoints(arm_joint_names)
            _log_plan_stats(f"place:{obj_name}", place_waypoints, env.step_dt)

            # -- 6. Execute place (fingers closed, object attached) --
            if not _execute_waypoints(
                env=env,
                robot=robot,
                ee_frame=ee_frame,
                waypoints=place_waypoints,
                gripper_cmd=GRIPPER_CLOSE_CMD,
                arm_joint_ids=arm_joint_ids,
                recorder=recorder,
                grasp_confidence=best_grasp.confidence,
                grasp_object_idx=obj_idx,
                label=f"place:{obj_name}",
                wp_ee_positions=planner.get_waypoint_ee_positions(),
            ):
                episode_aborted = True
                break

            # -- 7. Open gripper (release into bin) --
            if not _hold_gripper(
                env=env,
                robot=robot,
                ee_frame=ee_frame,
                arm_joint_ids=arm_joint_ids,
                gripper_cmd=GRIPPER_OPEN_CMD,
                steps=args.gripper_steps,
                recorder=recorder,
                grasp_confidence=0.0,
                grasp_object_idx=0,
                finger_joint_ids=finger_joint_ids,
                label=f"release:{obj_name}",
            ):
                episode_aborted = True
                break

            # Where did the object land relative to the bin?
            obj_final = (env.scene[obj_name].data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu()
            bin_final = (env.scene["packing_bin"].data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu()
            logging.info(
                "    Released %s at %s (bin at %s, horizontal offset %.4f m)",
                obj_name, _fmt(obj_final), _fmt(bin_final),
                torch.linalg.vector_norm(obj_final[:2] - bin_final[:2]).item(),
            )

            num_packed += 1
            logging.info("    Packed %s successfully.", obj_name)

            # -- 8. Return to home before planning the next object --
            if not _return_home(env, robot, ee_frame, planner, arm_joint_ids, arm_joint_names, recorder):
                episode_aborted = True
                break

        episode_success = (num_packed == len(OBJECT_NAMES)) and not episode_aborted
        recorder.close_episode(success=episode_success, num_objects_packed=num_packed)
        logging.info(
            "  Episode done — packed %d/3, success=%s%s",
            num_packed, episode_success, " (aborted: env auto-reset)" if episode_aborted else "",
        )

    grasp_client.close()
    isaac_client.close()
    recorder.close()
    env.close()
    simulation_app.close()
    logging.info("Data saved to %s", output_path.resolve())


if __name__ == "__main__":
    main()
