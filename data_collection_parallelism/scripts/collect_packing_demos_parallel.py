#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
collect_packing_demos_parallel.py — multi-environment autonomous data collection
for the packing task. One Isaac Sim process, num_envs=N, one shared GraspGenX
server connection, N independent CuRobo motion planners.

Design (see conversation / project memory for the full scoping discussion):
--------------------------------------------------------------------------
IsaacLab's vectorized env takes ONE action tensor for all N envs per env.step()
call — but each env can be at a different point of its own pick-place sequence
(env 3 might be mid-plan while env 7 just started a fresh episode). This script
resolves that by giving every env its own Python generator ("coroutine") that
mirrors data_collection/scripts/collect_packing_demos.py's per-object/per-attempt
control flow almost line-for-line — the only difference is every point that used
to call env.step() directly now does `env_reset = yield action` instead. A single
driver loop in main() collects the next action from all N generators, stacks them
into one (N, action_dim) tensor, calls env.step() ONCE for the whole batch, then
feeds each generator back its own env's reset flag. This preserves each env's
control flow exactly while sharing one physics/rendering/GraspGenX-connection
step.

Scoping note: motion planning here is N independent CuroboPlanner instances (one
per env_id, see ParallelPackMotionPlanner), NOT cuRobo's fused batched-planning API
(MotionGen.plan_batch_env, one GPU call across all N envs' collision worlds at
once). That deeper rewrite touches the shared isaaclab_mimic CuroboPlanner's world
construction and couldn't be safely verified without many live Isaac Sim
iterations. What's here already gives real parallelism: one Kit boot, one
GraspGenX server/model load, one process, N robots packing concurrently in sim
time — cuRobo's own per-instance planning cost is unchanged (see
parallel_pack_motion_planner.py's docstring for the full tradeoff).

Output format is identical to the single-env pipeline (see
data_collection/data_recording/lerobot_recorder.py) — GR00T LeRobot (LeRobot v2 +
meta/modality.json). episode_index/global frame index are shared, contiguous
counters across ALL envs' written episodes (see ParallelLeRobotRecorder), so the
resulting dataset is indistinguishable in format from one collected by N
sequential single-env runs — just faster to collect.

Usage
-----
  ./isaaclab.sh -p data_collection_parallelism/scripts/collect_packing_demos_parallel.py \\
      --num_envs 4 \\
      --num_episodes 40 \\
      --output datasets/packing_demos_parallel \\
      --headless

--num_episodes is a TOTAL across all envs (e.g. --num_envs 4 --num_episodes 40
means each env launches roughly 10 episodes, first-come-first-served — whichever
env finishes its current episode first claims the next slot).  --device controls
PhysX only (default "cpu" — see --device's help text for the measured VRAM
tradeoff vs "cuda:0"; rendering and CuRobo are unaffected either way).  All other
arguments match collect_packing_demos.py exactly (see that script's docstring for
--grasp_host/--grasp_port/--cameras/--filter_radius/--table_margin/
--object_placement/--record_cameras).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

# Force line-buffered stdout so output appears even through pipes / conda run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Make 'data_collection_parallelism' importable regardless of working directory.
# Script lives at <isaaclab_root>/data_collection_parallelism/scripts/collect_packing_demos_parallel.py
_ISAACLAB_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ISAACLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_ISAACLAB_ROOT))


_FILE_HANDLER: logging.FileHandler | None = None
_CONSOLE_HANDLER: logging.StreamHandler | None = None


def _tee_console_to_file(log_path: Path) -> None:
    """Mirror EVERYTHING written to stdout/stderr into <output>.console.log.

    See collect_packing_demos.py's identical helper for the full rationale
    (Isaac Sim's C++ output bypasses Python logging entirely).
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
    global _FILE_HANDLER, _CONSOLE_HANDLER

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    _FILE_HANDLER = fh

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    _CONSOLE_HANDLER = ch

    _tee_console_to_file(log_path)

    def _exc_handler(exc_type, exc_value, exc_tb):
        try:
            logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _exc_handler


def _reassert_logging() -> None:
    """Ensure our handlers survived Isaac Sim's boot / env creation (omni.log
    bridge can drop or reconfigure root handlers)."""
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
parser = argparse.ArgumentParser(description="Parallel (multi-env) packing demonstration collector")
parser.add_argument("--num_envs", type=int, default=4,
                     help="Number of parallel environments in ONE Isaac Sim process. Each env gets "
                          "its own CuRobo motion planner instance (own warmup/collision world), so "
                          "GPU memory scales roughly linearly with this — start small (2-4) and check "
                          "nvidia-smi headroom before scaling up on a single GPU.")
parser.add_argument("--num_episodes", type=int, default=10,
                     help="TOTAL episodes to collect across all --num_envs environments combined "
                          "(first-come-first-served — not per-env).")
parser.add_argument("--device", type=str, default="cpu",
                     help="Physics simulation device ('cpu' or e.g. 'cuda:0'). Does NOT affect "
                          "camera rendering (always GPU/RTX) or CuRobo (each planner hard-forces "
                          "CUDA regardless of this flag) — only PhysX. Measured on this box: CPU "
                          "physics saves ~0.9GB VRAM per env over GPU physics (~1.6GB/env vs "
                          "~2.6GB/env for physics+rendering combined), with no correctness issues "
                          "observed for num_envs in the 2-4 range. cuRobo (~3.3GB/env) still "
                          "dominates either way — this just frees a bit more headroom per env.")
parser.add_argument("--curobo_seeds", type=int, default=None,
                     help="Override CuroboPlannerCfg's num_trajopt_seeds/num_graph_seeds "
                          "(factory default: 12) for every env's motion planner. Measured "
                          "2026-07-23 on this task: 12 seeds -> ~3.6GB VRAM per planner instance, "
                          "4 seeds -> ~1.6GB (~54% less) — the single biggest lever found for "
                          "cuRobo's per-env footprint. NOT a free win: fewer seeds means fewer "
                          "parallel candidate starting points per planning call, which can lower "
                          "the pick/place planning success rate (this task's grasp poses are "
                          "often marginal — that's why max_planning_attempts=4 exists). Leave "
                          "unset to keep the factory default (12, unchanged behavior); only lower "
                          "this after comparing success rates at your target value against 12.")
parser.add_argument("--output", type=str, default="datasets/packing_demos_parallel")
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
                    default="table_top_cam,table_side_cam_1,table_side_cam_2,"
                    "table_side_cam_3,table_side_cam_4",
                    help="Comma-separated scene cameras whose depth clouds are fused into one "
                         "3D point cloud for GraspGenX (same semantics as collect_packing_demos.py).")
parser.add_argument("--filter_radius", type=float, default=0.15,
                    help="Radius (m) around each object centre used to segment the depth point "
                         "cloud before sending to GraspGenX.")
parser.add_argument("--table_margin", type=float, default=0.015,
                    help="Height band (m) above the lowest point of each object crop removed as "
                         "table surface before querying GraspGenX. <= 0 disables.")
parser.add_argument("--object_placement", type=str, default="random",
                    choices=["random", "fixed"],
                    help="'random': x/y position randomized per env per reset. 'fixed': objects "
                         "spawn at the env cfg init_state poses every reset.")
parser.add_argument("--debug", action="store_true",
                    help="Deprecated alias for --object_placement fixed.")
parser.add_argument("--record_cameras", type=str, default="wrist_cam,env_top_cam",
                    help="Comma-separated scene cameras to record RGB video for.")
parser.add_argument("--collect_mode", type=str, default="successful_only",
                     choices=["successful_only", "all"],
                     help="'successful_only' (default): discard the WHOLE episode unless all 3 "
                          "objects were packed (unchanged legacy behavior). 'all': keep each "
                          "object's pick/place/home segment if THAT object individually "
                          "succeeded, splicing out only the failed object(s) — an episode with "
                          "e.g. 2/3 packed still gets written (with only those 2 objects' data "
                          "and task list) instead of being discarded outright. An episode ended "
                          "by an env auto-reset (safety timeout) is always discarded fully in "
                          "both modes — there is no clean boundary to salvage from a mid-motion "
                          "interruption.")
parser.add_argument("--disable_camera_trim", action="store_true",
                     help="Keep every scene camera at its full data_types set (rgb+depth+semantic) "
                          "and leave the env cfg's 'rgb_camera' policy-observation group active. "
                          "By default (trim ENABLED) this script drops data_types nothing in the "
                          "pipeline reads (rgb from the 5 grasp cameras — GraspGenX only reads "
                          "depth+semantic; depth+semantic from the 2 recorded cameras — the "
                          "recorder only reads rgb) and removes any scene camera that's in "
                          "neither --cameras nor --record_cameras (e.g. front_cam by default). "
                          "Measured 2026-07-23: ~37-40% faster per tick, no correctness downside "
                          "since nothing in this pipeline ever reads the dropped outputs. Pass "
                          "this flag only if you need the full untrimmed camera set for debugging.")
args = parser.parse_args()
if args.debug:
    args.object_placement = "fixed"

_log_path = Path(args.output).with_suffix(".log")
_setup_logging(_log_path)
logging.info("collect_packing_demos_parallel starting — log: %s", _log_path)
logging.info("Args: %s", vars(args))

# --------------------------------------------------------------------------
# Bootstrap IsaacSim — must happen before any omni / isaaclab imports
# --------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

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

import isaaclab_tasks.manager_based.isaaclab_int  # noqa: E402, F401
import isaaclab_tasks.manager_based.isaaclab_int.config.franka  # noqa: E402, F401
from isaaclab_tasks.manager_based.isaaclab_int.mdp import franka_pack_events  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from data_collection_parallelism.grasp_client import (  # noqa: E402
    GraspGenXDepthClient, GraspGenXIsaacClient, filter_grasps,
)
from data_collection_parallelism.motion_planning import ParallelPackMotionPlanner  # noqa: E402
from data_collection_parallelism.motion_planning.pack_motion_planner import PackMotionPlanner  # noqa: E402, F401
from data_collection_parallelism.data_recording import Modality, ParallelLeRobotRecorder  # noqa: E402


# --------------------------------------------------------------------------
# Constants (identical to collect_packing_demos.py)
# --------------------------------------------------------------------------
ENV_ID = "Isaac-Pack-Object-Franka-Camera-v0"

GRIPPER_OPEN_CMD: float = 1.0
GRIPPER_CLOSE_CMD: float = -1.0

ARM_JOINT_PATTERN = "panda_joint"
FINGER_JOINT_PATTERN = "panda_finger"

OBJECT_NAMES = ["object_01", "object_02", "object_03"]

OBJECT_TASK_DESCRIPTIONS = [
    "pick up the bowl and place it in the packing bin",
    "pick up the packer bottle and place it in the packing bin",
    "pick up the coffee mug and place it in the packing bin",
]

MAX_PICK_ATTEMPTS = 5

MIN_GRASP_FINGER_POS = 0.001

# Offset/scale the arm JointPositionAction term actually applies — env-invariant
# (all N envs share an identical action-term config), read once in main() from
# env 0 and used for every env's action construction. See collect_packing_demos.py's
# identical note for why this must be the term's CACHED offset, not the live
# default_joint_pos (which the reset event rewrites after the term caches it).
_ARM_OFFSET: torch.Tensor | None = None
_ARM_SCALE: float = 0.5


def _read_arm_action_transform(env: ManagerBasedEnv, arm_joint_ids: list[int]) -> None:
    global _ARM_OFFSET, _ARM_SCALE

    term = env.action_manager.get_term("arm_action")
    offset = term._offset
    if isinstance(offset, torch.Tensor):
        _ARM_OFFSET = offset[0].detach().clone().to(env.device)
    else:
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
            "Action offset differs from live default by up to %.4f rad — inverting with the "
            "term's cached offset, worst joint: %s",
            mismatch.max().item(),
            robot.data.joint_names[arm_joint_ids[int(mismatch.argmax())]],
        )


def _make_action(
    robot: Articulation,
    arm_target_q: torch.Tensor,
    gripper_cmd: float,
    arm_joint_ids: list[int],
    env_id: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Convert absolute arm joint positions + gripper binary command to a single
    env's action tensor (num_arm_joints + 1,) — NOT batched; the driver stacks
    one of these per env into the (num_envs, action_dim) tensor env.step() wants."""
    if _ARM_OFFSET is not None:
        offset = _ARM_OFFSET.to(device=device)
        scale = _ARM_SCALE
    else:  # fallback (main() not initialized yet)
        offset = robot.data.default_joint_pos[env_id, arm_joint_ids].to(device=device)
        scale = 0.5
    arm_target_on_dev = arm_target_q[:len(arm_joint_ids)].to(device=device, dtype=torch.float32)
    action_arm = (arm_target_on_dev - offset) / scale
    action_gripper = torch.tensor([gripper_cmd], device=device, dtype=torch.float32)
    return torch.cat([action_arm, action_gripper])  # (num_arm_joints + 1,)


def _get_joint_ids(robot: Articulation) -> tuple[list[int], list[int]]:
    arm_ids = [
        i for i, name in enumerate(robot.data.joint_names)
        if ARM_JOINT_PATTERN in name and FINGER_JOINT_PATTERN not in name
    ]
    finger_ids = [
        i for i, name in enumerate(robot.data.joint_names)
        if FINGER_JOINT_PATTERN in name
    ]
    return arm_ids, finger_ids


def _fmt(t: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in t.reshape(-1).tolist()) + "]"


def _refresh_cameras(env: ManagerBasedEnv) -> None:
    """Force a render + sensor update so depth queries see the current scene.

    This refreshes ALL envs' cameras at once (rendering is not per-env-selectable)
    — harmless when called because one particular env needs a fresh grasp query;
    the other envs' cameras simply get refreshed a little more often than strictly
    necessary.
    """
    env.sim.render()
    env.scene.update(dt=env.physics_dt)


def _log_plan_stats(label: str, waypoints: list[torch.Tensor], step_dt: float) -> None:
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


def _grasp_gripped_object(robot: Articulation, env_id: int, finger_joint_ids: list[int]) -> bool:
    fingers = robot.data.joint_pos[env_id, finger_joint_ids].detach().cpu()
    return bool((fingers > MIN_GRASP_FINGER_POS).all().item())


def _enforce_tracking_gains(robot: Articulation, arm_joint_ids: list[int]) -> None:
    """Verify/enforce PD gains + gravity flags across ALL envs (view.count spans
    every env's robot instance, so this is already batch-correct as written —
    identical to collect_packing_demos.py's single-env version)."""
    view = robot.root_physx_view
    kp = view.get_dof_stiffnesses()[0]
    kd = view.get_dof_dampings()[0]
    logging.info(
        "PhysX arm gains at startup (env 0, representative — same config every env): kp=%s | kd=%s",
        [round(float(kp[i]), 1) for i in arm_joint_ids],
        [round(float(kd[i]), 1) for i in arm_joint_ids],
    )

    arm_kp_min = min(float(kp[i]) for i in arm_joint_ids)
    if arm_kp_min < 300.0:
        logging.warning(
            "Soft arm gains reached the sim (min kp=%.0f) — writing kp=400, kd=80 to PhysX "
            "(applies to every env's robot instance).",
            arm_kp_min,
        )
        robot.write_joint_stiffness_to_sim(400.0, joint_ids=arm_joint_ids)
        robot.write_joint_damping_to_sim(80.0, joint_ids=arm_joint_ids)

    grav = view.get_disable_gravities()  # (num_articulations, max_links) — spans every env
    grav_t = torch.as_tensor(grav).reshape(view.count, -1)
    if not bool(grav_t.all()):
        num_affected = int((grav_t == 0).sum().item())
        logging.warning(
            "Gravity is enabled on %d robot link(s) across all envs — disabling for tracking.",
            num_affected,
        )
        all_off = torch.ones_like(grav_t)
        indices = torch.arange(view.count, dtype=torch.int32, device=all_off.device)
        view.set_disable_gravities(all_off, indices)

    kp_after = view.get_dof_stiffnesses()[0]
    logging.info(
        "Effective arm gains for this run: kp=%s | gravity disabled on all robot links: %s",
        [round(float(kp_after[i]), 1) for i in arm_joint_ids],
        bool(torch.as_tensor(view.get_disable_gravities()).all()),
    )


# --------------------------------------------------------------------------
# Per-env worker state
# --------------------------------------------------------------------------

@dataclass
class EnvWorker:
    """Persistent per-env state — lives for the whole run; its `generator` /
    `pending_action` fields are replaced each time a new episode starts for
    this env, everything else is fixed for the process lifetime."""
    env_id: int
    env: ManagerBasedEnv
    robot: Articulation
    ee_frame: FrameTransformer
    planner: PackMotionPlanner
    grasp_client: GraspGenXDepthClient
    isaac_client: GraspGenXIsaacClient
    arm_joint_ids: list[int]
    finger_joint_ids: list[int]
    arm_joint_names: list[str]
    args: argparse.Namespace
    recorder: ParallelLeRobotRecorder
    record_camera_names: list[str]
    generator: Generator | None = None
    pending_action: torch.Tensor | None = None
    finished_forever: bool = False
    step_idx: int = 0
    task_index: int = 0


def _capture_pre_step(worker: EnvWorker) -> tuple[object, dict] | None:
    """Snapshot state/camera frames BEFORE this tick's batched env.step() — obs_i
    must be the state action_i was CHOSEN from, not the state it produces (see
    collect_packing_demos.py's _EpisodeRecorder docstring for the full rationale).
    Returns None if this env isn't currently recording an episode."""
    if not worker.recorder.is_recording(worker.env_id):
        return None
    env_id = worker.env_id
    state = torch.cat([
        worker.robot.data.joint_pos[env_id, worker.arm_joint_ids],
        worker.robot.data.joint_pos[env_id, worker.finger_joint_ids],
    ]).detach().cpu().numpy()
    frames = {
        cam: worker.env.scene[cam].data.output["rgb"][env_id].detach().cpu().numpy()
        for cam in worker.record_camera_names
    }
    return (state, frames)


def _commit_step(worker: EnvWorker, pre: tuple[object, dict] | None, action: torch.Tensor, step_dt: float) -> None:
    """Pair the pre-step snapshot with the action just applied this tick."""
    if pre is None:
        return
    state, frames = pre
    action_np = action.detach().cpu().numpy()
    worker.recorder.record_step(
        worker.env_id, state, action_np, frames,
        timestamp=worker.step_idx * step_dt, task_index=worker.task_index,
    )
    worker.step_idx += 1


# --------------------------------------------------------------------------
# Generator-based waypoint execution (mirrors collect_packing_demos.py's
# _execute_waypoints / _hold_gripper / _return_home, but yields one action per
# env.step() tick instead of calling env.step() directly, so the driver can
# batch it together with every other active env's action for that same tick.
# --------------------------------------------------------------------------

def _execute_waypoints_gen(
    worker: EnvWorker,
    waypoints: list[torch.Tensor],
    gripper_cmd: float,
    label: str = "",
    wp_ee_positions: list[torch.Tensor] | None = None,
    settle_tol: float = 0.01,
    max_settle_steps: int = 24,
) -> Generator[torch.Tensor, bool, bool]:
    """Yields one (action_dim,) action per tick; caller .send()s back whether
    THIS env auto-reset that tick. Returns True if all waypoints executed and the
    arm settled; False if the env auto-reset mid-execution (episode must abort)."""
    env = worker.env
    robot = worker.robot
    ee_frame = worker.ee_frame
    arm_joint_ids = worker.arm_joint_ids
    env_id = worker.env_id

    if not waypoints:
        logging.warning("    [env%d:%s] empty waypoint list — nothing to execute.", env_id, label)
        return True

    num_arm = len(arm_joint_ids)

    q_now = robot.data.joint_pos[env_id, arm_joint_ids].detach().cpu()
    start_gap = (waypoints[0].detach().cpu()[:num_arm] - q_now).abs().max().item()
    logging.debug(
        "[env%d:%s] start check: q_now=%s | wp[0]=%s | gap=%.4f rad",
        env_id, label, _fmt(q_now), _fmt(waypoints[0].detach().cpu()[:num_arm]), start_gap,
    )
    if start_gap > 0.05:
        logging.warning(
            "    [env%d:%s] plan starts %.4f rad away from current joint state — "
            "plan start state is stale or wrong.", env_id, label, start_gap,
        )

    diverged = False
    for i, wp in enumerate(waypoints):
        action = _make_action(
            robot=robot, arm_target_q=wp, gripper_cmd=gripper_cmd,
            arm_joint_ids=arm_joint_ids, env_id=env_id, device=env.device,
        )
        env_reset = yield action

        q_cmd = wp.detach().cpu()[:num_arm]
        q_meas = robot.data.joint_pos[env_id, arm_joint_ids].detach().cpu()
        track_err = (q_cmd - q_meas).abs().max().item()
        ee_meas = (ee_frame.data.target_pos_w[env_id, 0, :] - env.scene.env_origins[env_id]).detach().cpu()
        ee_cmd = wp_ee_positions[i] if wp_ee_positions is not None and i < len(wp_ee_positions) else None
        logging.debug(
            "[env%d:%s] wp %03d/%03d | q_cmd=%s | action=%s | q_meas=%s | max_err=%.4f rad | ee_cmd=%s | ee_meas=%s",
            env_id, label, i + 1, len(waypoints),
            _fmt(q_cmd), _fmt(action), _fmt(q_meas), track_err,
            _fmt(ee_cmd) if ee_cmd is not None else "n/a", _fmt(ee_meas),
        )

        if track_err > 0.3 and not diverged:
            diverged = True
            logging.warning(
                "    [env%d:%s] tracking error %.3f rad at waypoint %d/%d | worst joint: %s | q_cmd=%s | q_meas=%s",
                env_id, label, track_err, i + 1, len(waypoints),
                robot.data.joint_names[arm_joint_ids[int((q_cmd - q_meas).abs().argmax())]],
                _fmt(q_cmd), _fmt(q_meas),
            )

        if env_reset:
            logging.warning(
                "    [env%d:%s] env auto-reset at waypoint %d/%d — aborting episode.",
                env_id, label, i + 1, len(waypoints),
            )
            return False

    final_wp = waypoints[-1]
    final_target = final_wp.detach().cpu()[:num_arm]
    settle_steps = 0
    for _ in range(max_settle_steps):
        q_meas = robot.data.joint_pos[env_id, arm_joint_ids].detach().cpu()
        if (final_target - q_meas).abs().max().item() < settle_tol:
            break
        action = _make_action(
            robot=robot, arm_target_q=final_wp, gripper_cmd=gripper_cmd,
            arm_joint_ids=arm_joint_ids, env_id=env_id, device=env.device,
        )
        env_reset = yield action
        settle_steps += 1
        if env_reset:
            logging.warning("    [env%d:%s] env auto-reset while settling — aborting episode.", env_id, label)
            return False

    q_meas = robot.data.joint_pos[env_id, arm_joint_ids].detach().cpu()
    final_err = (final_target - q_meas).abs().max().item()
    ee_meas = (ee_frame.data.target_pos_w[env_id, 0, :] - env.scene.env_origins[env_id]).detach().cpu()
    logging.info(
        "    [env%d:%s] done: %d waypoints + %d settle steps | final joint err %.4f rad | ee_meas=%s",
        env_id, label, len(waypoints), settle_steps, final_err, _fmt(ee_meas),
    )
    if final_err > settle_tol:
        logging.warning(
            "    [env%d:%s] arm did not converge (%.4f rad > %.4f) — PD tracking is lagging.",
            env_id, label, final_err, settle_tol,
        )
    return True


def _hold_gripper_gen(
    worker: EnvWorker,
    gripper_cmd: float,
    steps: int,
    label: str = "gripper",
) -> Generator[torch.Tensor, bool, bool]:
    robot = worker.robot
    env_id = worker.env_id
    arm_joint_ids = worker.arm_joint_ids
    finger_joint_ids = worker.finger_joint_ids

    current_arm_q = robot.data.joint_pos[env_id, arm_joint_ids].detach().clone()
    for k in range(steps):
        action = _make_action(
            robot=robot, arm_target_q=current_arm_q, gripper_cmd=gripper_cmd,
            arm_joint_ids=arm_joint_ids, env_id=env_id, device=worker.env.device,
        )
        env_reset = yield action

        fingers = robot.data.joint_pos[env_id, finger_joint_ids].detach().cpu()
        logging.debug(
            "[env%d:%s] step %02d/%02d | cmd=%+.1f | fingers=%s",
            env_id, label, k + 1, steps, gripper_cmd, _fmt(fingers),
        )
        if env_reset:
            logging.warning("    [env%d:%s] env auto-reset while toggling gripper — aborting episode.", env_id, label)
            return False

    fingers = robot.data.joint_pos[env_id, finger_joint_ids].detach().cpu()
    logging.info("    [env%d:%s] final finger positions: %s", env_id, label, _fmt(fingers))
    return True


def _return_home_gen(worker: EnvWorker) -> Generator[torch.Tensor, bool, bool]:
    env = worker.env
    robot = worker.robot
    env_id = worker.env_id

    home_q = robot.data.default_joint_pos[env_id].detach().clone()
    if not worker.planner.plan_home(home_q, list(robot.data.joint_names)):
        logging.warning("    [env%d] CuRobo home planning failed — continuing from current pose.", env_id)
        return True
    waypoints = worker.planner.get_arm_waypoints(worker.arm_joint_names)
    _log_plan_stats(f"env{env_id}:home", waypoints, env.step_dt)
    ok = yield from _execute_waypoints_gen(
        worker, waypoints, GRIPPER_OPEN_CMD, label="home",
        wp_ee_positions=worker.planner.get_waypoint_ee_positions(),
    )
    return ok


# --------------------------------------------------------------------------
# Per-env episode coroutine — mirrors collect_packing_demos.py's per-episode
# for-loop body (object loop + attempt loop) almost verbatim.
# --------------------------------------------------------------------------

def _run_episode(worker: EnvWorker) -> Generator[torch.Tensor, bool, bool]:
    """One full episode for worker.env_id: settle -> for each object, query grasp
    -> attempt pick/place -> return home. Returns episode_success (bool)."""
    env = worker.env
    env_id = worker.env_id
    robot = worker.robot
    wargs = worker.args
    recorder = worker.recorder

    # -- settle after reset (NOT recorded — mirrors the original script's raw
    # env.sim.step() settle loop, adapted here to a held default-pose action
    # since physics can't be stepped for a single env in isolation; see module
    # docstring / project notes for why this is an equivalent adaptation). --
    default_arm_q = robot.data.default_joint_pos[env_id, worker.arm_joint_ids].detach().clone()
    for _ in range(wargs.settle_steps):
        action = _make_action(
            robot=robot, arm_target_q=default_arm_q, gripper_cmd=GRIPPER_OPEN_CMD,
            arm_joint_ids=worker.arm_joint_ids, env_id=env_id, device=env.device,
        )
        env_reset = yield action
        if env_reset:
            logging.warning("  [env%d] env auto-reset during post-reset settle — restarting.", env_id)
            return False

    recorder.start_episode(env_id)
    worker.step_idx = 0
    worker.task_index = 0

    num_packed = 0
    episode_aborted = False
    object_success: dict[int, bool] = {}

    for obj_idx, obj_name in enumerate(OBJECT_NAMES, start=1):
        logging.info("  [env%d] Object %d/3: %s", env_id, obj_idx, obj_name)
        worker.task_index = obj_idx - 1
        object_success[obj_idx - 1] = False

        obj_pos = (env.scene[obj_name].data.root_pos_w[env_id] - env.scene.env_origins[env_id]).detach().cpu()
        if obj_pos[2] < 0.0:
            logging.warning("    [env%d] %s has fallen off the table (z=%.3f) — skipping.", env_id, obj_name, obj_pos[2])
            continue

        _refresh_cameras(env)
        raw_grasps = worker.grasp_client.query(obj_name, env, env_id=env_id)
        if not raw_grasps:
            logging.info(
                "    [env%d] Depth client got 0 points for %s — falling back to USD mesh client.",
                env_id, obj_name,
            )
            raw_grasps = worker.isaac_client.query(obj_name, env, env_id=env_id)

        if raw_grasps:
            dists = [float(torch.linalg.vector_norm(g.tcp_position() - obj_pos)) for g in raw_grasps]
            downs = [g.downwardness() for g in raw_grasps]
            logging.info(
                "    [env%d] Candidates for %s: %d raw | TCP dist to object root %s: min %.3f / mean %.3f m "
                "| downwardness max %.2f / mean %.2f",
                env_id, obj_name, len(raw_grasps), _fmt(obj_pos), min(dists), sum(dists) / len(dists),
                max(downs), sum(downs) / len(downs),
            )

        candidate_grasps = filter_grasps(
            raw_grasps, confidence_threshold=wargs.grasp_threshold, top_k=MAX_PICK_ATTEMPTS, object_pos=obj_pos,
        )
        if not candidate_grasps:
            logging.warning(
                "    [env%d] No grasp above threshold for %s (got %d raw), skipping.",
                env_id, obj_name, len(raw_grasps),
            )
            continue

        best_grasp = None
        grasp_confirmed = False
        for attempt_idx, candidate in enumerate(candidate_grasps, start=1):
            logging.info(
                "    [env%d] Grasp attempt %d/%d: conf %.3f | TCP dist to object %.3f m | downwardness %.2f | "
                "hand pos=%s tcp=%s quat=%s",
                env_id, attempt_idx, len(candidate_grasps), candidate.confidence,
                float(torch.linalg.vector_norm(candidate.tcp_position() - obj_pos)), candidate.downwardness(),
                _fmt(candidate.position), _fmt(candidate.tcp_position()), _fmt(candidate.quaternion),
            )
            hand_z = float(candidate.position[2])
            tcp_z = float(candidate.tcp_position()[2])
            obj_z = float(obj_pos[2])
            logging.debug(
                "    [env%d][height] %s | hand_base_z=%+.4f | tcp_z=%+.4f | object_root_z=%+.4f "
                "| hand_gap=%+.4f m | tcp_gap=%+.4f m",
                env_id, obj_name, hand_z, tcp_z, obj_z, hand_z - obj_z, tcp_z - obj_z,
            )

            if not worker.planner.plan_pick(candidate):
                logging.warning(
                    "    [env%d] CuRobo pick planning failed for %s (attempt %d/%d).",
                    env_id, obj_name, attempt_idx, len(candidate_grasps),
                )
                continue

            pick_waypoints = worker.planner.get_arm_waypoints(worker.arm_joint_names)
            _log_plan_stats(f"env{env_id}:pick:{obj_name}", pick_waypoints, env.step_dt)

            ok = yield from _execute_waypoints_gen(
                worker, pick_waypoints, GRIPPER_OPEN_CMD, label=f"pick:{obj_name}",
                wp_ee_positions=worker.planner.get_waypoint_ee_positions(),
            )
            if not ok:
                episode_aborted = True
                break

            ee_meas = (worker.ee_frame.data.target_pos_w[env_id, 0, :] - env.scene.env_origins[env_id]).detach().cpu()
            reach_err = torch.linalg.vector_norm(ee_meas - candidate.position).item()
            logging.info("    [env%d] Pick reach error: %.4f m (EE vs grasp target)", env_id, reach_err)

            ok = yield from _hold_gripper_gen(worker, GRIPPER_CLOSE_CMD, wargs.gripper_steps, label=f"close:{obj_name}")
            if not ok:
                episode_aborted = True
                break

            fingers = worker.robot.data.joint_pos[env_id, worker.finger_joint_ids].detach().cpu()
            if _grasp_gripped_object(worker.robot, env_id, worker.finger_joint_ids):
                logging.info(
                    "    [env%d] Gripper closed on %s (fingers=%s) — grasp CONFIRMED, proceeding to place.",
                    env_id, obj_name, _fmt(fingers),
                )
                best_grasp = candidate
                grasp_confirmed = True
                break

            logging.warning(
                "    [env%d] Grasp check failed for %s (attempt %d/%d): fingers=%s <= %.4f m — closed on air.",
                env_id, obj_name, attempt_idx, len(candidate_grasps), _fmt(fingers), MIN_GRASP_FINGER_POS,
            )
            ok = yield from _hold_gripper_gen(worker, GRIPPER_OPEN_CMD, wargs.gripper_steps, label=f"open:{obj_name}")
            if not ok:
                episode_aborted = True
                break

        if episode_aborted:
            break
        if not grasp_confirmed:
            logging.warning(
                "    [env%d] All %d grasp attempt(s) failed for %s (planning or verification) — skipping.",
                env_id, len(candidate_grasps), obj_name,
            )
            # Return home before the next object even though this one failed — every
            # object boundary (success or failure) must leave the arm at the SAME known
            # pose, so that collect_mode="all" can excise just this failed segment
            # without splicing together two frames that jump discontinuously (the arm
            # may be stranded wherever the last failed attempt left it otherwise).
            ok = yield from _return_home_gen(worker)
            if not ok:
                episode_aborted = True
                break
            continue

        place_ok = worker.planner.plan_place(obj_name, slot_index=obj_idx - 1)
        if not place_ok:
            logging.warning("    [env%d] CuRobo place planning failed for %s.", env_id, obj_name)
            ok_open = yield from _hold_gripper_gen(worker, GRIPPER_OPEN_CMD, wargs.gripper_steps, label=f"open:{obj_name}")
            ok_home = (yield from _return_home_gen(worker)) if ok_open else False
            if not ok_open or not ok_home:
                episode_aborted = True
                break
            continue

        place_waypoints = worker.planner.get_arm_waypoints(worker.arm_joint_names)
        _log_plan_stats(f"env{env_id}:place:{obj_name}", place_waypoints, env.step_dt)

        ok = yield from _execute_waypoints_gen(
            worker, place_waypoints, GRIPPER_CLOSE_CMD, label=f"place:{obj_name}",
            wp_ee_positions=worker.planner.get_waypoint_ee_positions(),
        )
        if not ok:
            episode_aborted = True
            break

        ok = yield from _hold_gripper_gen(worker, GRIPPER_OPEN_CMD, wargs.gripper_steps, label=f"release:{obj_name}")
        if not ok:
            episode_aborted = True
            break

        obj_final = (env.scene[obj_name].data.root_pos_w[env_id] - env.scene.env_origins[env_id]).detach().cpu()
        bin_final = (env.scene["packing_bin"].data.root_pos_w[env_id] - env.scene.env_origins[env_id]).detach().cpu()
        logging.info(
            "    [env%d] Released %s at %s (bin at %s, horizontal offset %.4f m)",
            env_id, obj_name, _fmt(obj_final), _fmt(bin_final),
            torch.linalg.vector_norm(obj_final[:2] - bin_final[:2]).item(),
        )

        num_packed += 1
        object_success[obj_idx - 1] = True
        logging.info("    [env%d] Packed %s successfully.", env_id, obj_name)

        ok = yield from _return_home_gen(worker)
        if not ok:
            episode_aborted = True
            break

    episode_success = (num_packed == len(OBJECT_NAMES)) and not episode_aborted
    written = recorder.close_episode(
        env_id, episode_success, object_success=object_success, aborted=episode_aborted,
    )
    logging.info(
        "  [env%d] Episode done — packed %d/3, success=%s%s | %s",
        env_id, num_packed, episode_success, " (aborted: env auto-reset)" if episode_aborted else "",
        "written" if written else "discarded (not written)",
    )
    return episode_success


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

ALL_SCENE_CAMERAS = [
    "wrist_cam", "env_top_cam", "table_top_cam", "front_cam",
    "table_side_cam_1", "table_side_cam_2", "table_side_cam_3", "table_side_cam_4",
]


def _trim_unused_camera_render(env_cfg, camera_names: list[str], record_camera_names: list[str]) -> None:
    """Cut rendering work nothing in this pipeline reads (verified 2026-07-23: ~37-40%
    faster per tick, at any num_envs, no correctness downside):
      - the env cfg's 'rgb_camera' observation group (RL-policy image obs, hardcoded
        ObsTerms for rgb+depth on several cameras) is never read by this collector —
        it only calls env.step(action), never touches the returned obs dict.
      - GraspGenXDepthClient only ever reads distance_to_image_plane/semantic_segmentation
        (never rgb); the recorder only ever reads rgb (never depth/semantic) — so a
        camera used for grasp querying doesn't need rgb, and a recorded camera doesn't
        need depth/semantic, UNLESS it happens to serve both roles at once.
      - any scene camera in neither list (e.g. front_cam, unused by default) is dropped
        entirely.
    """
    env_cfg.observations.rgb_camera = None
    grasp_only = set(camera_names) - set(record_camera_names)
    record_only = set(record_camera_names) - set(camera_names)
    both_roles = set(camera_names) & set(record_camera_names)
    for cam_name in ALL_SCENE_CAMERAS:
        if cam_name in both_roles:
            continue  # needs the full data_types set — no reduction possible
        elif cam_name in grasp_only:
            getattr(env_cfg.scene, cam_name).data_types = ["distance_to_image_plane", "semantic_segmentation"]
        elif cam_name in record_only:
            getattr(env_cfg.scene, cam_name).data_types = ["rgb"]
        else:
            setattr(env_cfg.scene, cam_name, None)
    logging.info(
        "Camera rendering trimmed (--disable_camera_trim to opt out): grasp-only cams -> "
        "depth+semantic, record-only cams -> rgb, unused cams removed entirely: %s",
        [c for c in ALL_SCENE_CAMERAS if c not in both_roles and c not in grasp_only and c not in record_only],
    )


def main() -> None:
    # ---- environment (num_envs=N, one shared Isaac Sim process) ----
    env_cfg = parse_env_cfg(ENV_ID, device=args.device, num_envs=args.num_envs)
    env_cfg.env_name = ENV_ID

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    record_camera_names = [c.strip() for c in args.record_cameras.split(",") if c.strip()]
    if not args.disable_camera_trim:
        _trim_unused_camera_render(env_cfg, camera_names, record_camera_names)

    output_path = Path(args.output)

    # Same rationale as collect_packing_demos.py: the script owns episode
    # boundaries via explicit env.reset(env_ids=...) calls, so disable every
    # termination that would make ManagerBasedRLEnv auto-reset mid-episode.
    env_cfg.episode_length_s = 120.0
    env_cfg.terminations.object_1_dropping = None
    env_cfg.terminations.object_2_dropping = None
    env_cfg.terminations.object_3_dropping = None
    env_cfg.terminations.success = None

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
            logging.info("Object placement FIXED — objects reset to their init_state poses every episode.")
        else:
            logging.warning(
                "--object_placement fixed given but no 'reset_objects_pose' event was found on the env cfg."
            )
    else:
        logging.info("Object placement RANDOM — x/y positions randomized per env, per reset.")

    env: ManagerBasedEnv = gym.make(ENV_ID, cfg=env_cfg).unwrapped
    env.reset()
    _reassert_logging()

    logging.info(
        "Timing: physics %.1f Hz | control (env.step) %.1f Hz | plan waypoint dt %.4f s | num_envs=%d",
        1.0 / env.physics_dt, 1.0 / env.step_dt, env.step_dt, args.num_envs,
    )
    if args.filter_radius > 0.30:
        logging.warning(
            "filter_radius=%.2f m is large — the cropped cloud will include table/neighbour points.",
            args.filter_radius,
        )

    robot: Articulation = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    arm_joint_ids, finger_joint_ids = _get_joint_ids(robot)
    arm_joint_names = [robot.data.joint_names[i] for i in arm_joint_ids]

    _enforce_tracking_gains(robot, arm_joint_ids)
    _read_arm_action_transform(env, arm_joint_ids)

    # ---- GraspGenX clients (SHARED across all envs — the server already
    # serializes REQ/REP requests, so one connection is correct/sufficient) ----
    missing_cams = [c for c in camera_names if c not in env.scene.keys()]
    if missing_cams:
        raise ValueError(f"--cameras contains unknown scene camera(s): {missing_cams}.")
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
    isaac_client = GraspGenXIsaacClient(
        host=args.grasp_host,
        port=args.grasp_port,
        gripper_name="franka_panda",
        num_grasps=200,
        topk_num_grasps=args.grasp_topk,
    )
    logging.info(
        "Connected to GraspGenX server at %s:%s (fused cameras=%s, filter_radius=%.3f m, shared by %d envs)",
        args.grasp_host, args.grasp_port, ",".join(camera_names), args.filter_radius, args.num_envs,
    )

    # ---- N independent CuRobo motion planners (one per env_id) ----
    planners = ParallelPackMotionPlanner(
        env=env, robot=robot, num_envs=args.num_envs,
        num_trajopt_seeds=args.curobo_seeds, num_graph_seeds=args.curobo_seeds,
    )

    # ---- LeRobot dataset recorder (SHARED — contiguous episode_index/global
    # frame index across every env's written episodes) ----
    missing_record_cams = [c for c in record_camera_names if c not in env.scene.keys()]
    if missing_record_cams:
        raise ValueError(f"--record_cameras contains unknown scene camera(s): {missing_record_cams}.")
    first_cam_cfg = env.scene[record_camera_names[0]].cfg
    image_hw = (first_cam_cfg.height, first_cam_cfg.width)

    state_modalities = [
        Modality("arm_joint_pos", len(arm_joint_ids)),
        Modality("gripper_joint_pos", len(finger_joint_ids)),
    ]
    action_modalities = [Modality("arm_action", len(arm_joint_ids)), Modality("gripper_action", 1)]

    recorder = ParallelLeRobotRecorder(
        output_dir=output_path,
        camera_names=record_camera_names,
        state_modalities=state_modalities,
        action_modalities=action_modalities,
        task_descriptions=OBJECT_TASK_DESCRIPTIONS,
        fps=1.0 / env.step_dt,
        image_hw=image_hw,
        collect_mode=args.collect_mode,
    )
    logging.info(
        "Recording GR00T LeRobot dataset to %s (cameras=%s, state_dim=%d, action_dim=%d, fps=%.1f Hz, "
        "num_envs=%d). Only successful episodes are written.",
        output_path.resolve(), ",".join(record_camera_names),
        sum(m.dim for m in state_modalities), sum(m.dim for m in action_modalities),
        1.0 / env.step_dt, args.num_envs,
    )

    workers = [
        EnvWorker(
            env_id=i, env=env, robot=robot, ee_frame=ee_frame, planner=planners[i],
            grasp_client=grasp_client, isaac_client=isaac_client,
            arm_joint_ids=arm_joint_ids, finger_joint_ids=finger_joint_ids, arm_joint_names=arm_joint_names,
            args=args, recorder=recorder, record_camera_names=record_camera_names,
        )
        for i in range(args.num_envs)
    ]

    # Shared, first-come-first-served counter: args.num_episodes is a TOTAL
    # across all envs, not per-env.
    launched_count = 0

    def _start_new_episode(worker: EnvWorker) -> torch.Tensor:
        nonlocal launched_count
        if launched_count >= args.num_episodes:
            worker.finished_forever = True
            logging.info("  [env%d] episode quota reached — freezing this env for the rest of the run.", worker.env_id)
            default_arm_q = robot.data.default_joint_pos[worker.env_id, arm_joint_ids].detach().clone()
            return _make_action(
                robot=robot, arm_target_q=default_arm_q, gripper_cmd=GRIPPER_OPEN_CMD,
                arm_joint_ids=arm_joint_ids, env_id=worker.env_id, device=env.device,
            )
        launched_count += 1
        logging.info("=== [env%d] Starting episode (%d/%d launched total) ===", worker.env_id, launched_count, args.num_episodes)
        env.reset(env_ids=torch.tensor([worker.env_id], device=env.device, dtype=torch.long))
        _reassert_logging()  # env.reset() can reconfigure Python logging via the omni bridge
        gen = _run_episode(worker)
        worker.generator = gen
        try:
            return next(gen)
        except StopIteration:
            # Degenerate episode (e.g. --settle_steps 0) that finished with no
            # yields at all — try the next one immediately (bounded recursion:
            # args.num_episodes - launched_count strictly decreases).
            worker.generator = None
            return _start_new_episode(worker)

    for w in workers:
        w.pending_action = _start_new_episode(w)

    while not all(w.finished_forever for w in workers):
        active = [w for w in workers if not w.finished_forever]

        pre_captures = {w.env_id: _capture_pre_step(w) for w in active}

        action_batch = torch.stack([w.pending_action for w in workers]).to(device=env.device)
        _, _, terminated, truncated, _ = env.step(action_batch)

        for w in active:
            _commit_step(w, pre_captures[w.env_id], w.pending_action, env.step_dt)

        for w in active:
            env_reset = bool(terminated[w.env_id].item()) or bool(truncated[w.env_id].item())
            try:
                w.pending_action = w.generator.send(env_reset)
            except StopIteration:
                # Persist meta/*.json after EVERY episode close, across any env —
                # a killed process then leaves the dataset fully loadable up
                # through the last episode that finished, same crash-safety
                # convention as the single-env recorder. Note: as the dataset
                # grows this rescans every written parquet file each call — fine
                # at the scales this pipeline has been used at; if it becomes a
                # bottleneck with many envs finishing close together, throttle
                # this to every K closes instead of every single one.
                recorder.finalize()
                w.pending_action = _start_new_episode(w)

    grasp_client.close()
    isaac_client.close()
    env.close()
    recorder.finalize()
    logging.info(
        "Run complete — %d/%d episode(s) written to %s (num_envs=%d).",
        recorder.num_episodes_written, args.num_episodes, output_path.resolve(), args.num_envs,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
