#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
convert_to_mimic_hdf5.py — bridge collect_packing_demos.py's HDF5 output into the
schema Isaac Lab Mimic's annotate_demos.py / generate_dataset.py require.

Why this exists (not a rewrite of the live collector's action space)
----------------------------------------------------------------------
collect_packing_demos.py drives the robot via cuRobo joint-position waypoints — that
control loop is safety-critical (collision-free tracking, verified grasps) and already
tuned; re-deriving an IK-relative action space live during collection would mean
double-solving IK against a different tracking loop than the one already validated.
Isaac Lab Mimic needs a different HDF5 schema entirely: a top-level ``data`` group,
and per-episode ``initial_state`` + ``actions`` that are REPLAYED verbatim via
``env.step()`` against the IK-relative Mimic env (``Isaac-Pack-Object-Franka-IK-Rel-
Mimic-v0`` / ``...-Visuomotor-Mimic-v0``). This script derives that action space
OFFLINE from the already-recorded ground-truth end-effector trajectory (``ee_pos``/
``ee_quat``, measured by the FrameTransformer every step) — the measured trajectory
is exactly what physically happened, a strictly better source than re-deriving from
planned/FK poses, and it requires zero changes to the live collection control loop.

Action derivation
------------------
For consecutive recorded steps i -> i+1::

    delta_pos      = ee_pos[i+1]  - ee_pos[i]
    delta_rot_mat  = R(ee_quat[i+1]) @ R(ee_quat[i]).T   # world-frame convention,
                                                          # matches compute_pose_error /
                                                          # apply_delta_pose exactly
                                                          # (verified against
                                                          # differential_ik.py)
    delta_rotation = axis_angle(quat_from_matrix(delta_rot_mat))
    action[i]      = concat(delta_pos, delta_rotation) / ik_scale, gripper_cmd[i+1]

Dividing the 6-D pose part by the IK action term's ``scale`` (0.5 on every packing
IK-Rel cfg) inverts the action term's own scaling exactly (command = raw_action *
scale), so replaying action[i] through the IK-relative Mimic env reproduces the
recorded delta pose in one step — the same exact-inversion approach
collect_packing_demos.py already uses for the joint-position action term (see
``_make_action``'s docstring there). The gripper dimension is a raw pass-through
(``BinaryJointPositionActionCfg`` has no scaling), so ``gripper_cmd`` is used as-is.

Requires episodes recorded with the ``gripper_cmd`` field and the synthetic step-0
pre-action snapshot (both added to HDF5EpisodeRecorder / collect_packing_demos.py
alongside this script) — episodes missing either are skipped with a warning; re-run
collect_packing_demos.py to regenerate them.

Usage
-----
  ./isaaclab.sh -p data_collection/scripts/convert_to_mimic_hdf5.py \\
      --input datasets/packing_demos.hdf5 \\
      --output datasets/packing_demos.mimic.hdf5

Arguments
---------
  --input         Custom-format HDF5 produced by collect_packing_demos.py.
  --output        Mimic-native HDF5 to write (data/demo_N/{initial_state,actions}).
  --env_name      Gym id stored in env_args (default:
                  Isaac-Pack-Object-Franka-IK-Rel-Mimic-v0). Use
                  Isaac-Pack-Object-Franka-IK-Rel-Visuomotor-Mimic-v0 for the
                  camera-equipped variant — the action space (and so this script's
                  math) is identical, only the observation/replay env differs.
  --ik_scale      DifferentialInverseKinematicsActionCfg.scale used by the target
                  env's arm action term (default: 0.5, matches every packing IK-Rel
                  cfg — see franka_pack_ik_rel_env_cfg.py /
                  franka_pack_ik_rel_visuomotor_env_cfg.py).
  --only_success  Only convert episodes with success=True (default: on; pass
                  --no-only_success to include failed episodes too, e.g. for a
                  generation_keep_failed workflow).
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description="Convert packing demos to Isaac Lab Mimic HDF5 format")
parser.add_argument("--input", type=str, required=True, help="Custom-format HDF5 from collect_packing_demos.py")
parser.add_argument("--output", type=str, required=True, help="Mimic-native HDF5 to write")
parser.add_argument(
    "--env_name", type=str, default="Isaac-Pack-Object-Franka-IK-Rel-Mimic-v0",
    help="Gym id stored in env_args / used as the Mimic action-space reference",
)
parser.add_argument(
    "--ik_scale", type=float, default=0.5,
    help="DifferentialInverseKinematicsActionCfg.scale of the target env's arm action term",
)
parser.add_argument(
    "--only_success", action=argparse.BooleanOptionalAction, default=True,
    help="Only convert episodes with success=True (default: on)",
)
args = parser.parse_args()

# --------------------------------------------------------------------------
# Bootstrap IsaacSim — isaaclab.utils.math/datasets pull in pxr (USD bindings),
# which is only importable through Isaac Sim's own Python; AppLauncher sets that
# up. No simulated env is created below, so headless with no cameras is enough.
# --------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------
# Normal imports (IsaacSim is now running)
# --------------------------------------------------------------------------
import h5py  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as PoseUtils  # noqa: E402
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler  # noqa: E402

OBJECT_ENTITIES = [
    ("object_01", "obj1_pos", "obj1_quat"),
    ("object_02", "obj2_pos", "obj2_quat"),
    ("object_03", "obj3_pos", "obj3_quat"),
    ("packing_bin", "bin_pos", "bin_quat"),
]


def _pose_delta_action(
    pos_a: torch.Tensor, quat_a: torch.Tensor, pos_b: torch.Tensor, quat_b: torch.Tensor, scale: float
) -> torch.Tensor:
    """Raw IK-relative action (7,) that moves the eef from (pos_a, quat_a) to
    (pos_b, quat_b) in one replayed step.

    This is the exact inverse of DifferentialInverseKinematicsActionCfg's own
    processing: command = raw_action * scale, then
    apply_delta_pose(pos_a, quat_a, command) == (pos_b, quat_b). Same world-frame
    delta convention as FrankaPackIKRelMimicEnv.target_eef_pose_to_action /
    action_to_target_eef_pose (verified against isaaclab.utils.math.apply_delta_pose
    / compute_pose_error, which both use R_error = R_target @ R_current^T).
    """
    rot_a = PoseUtils.matrix_from_quat(quat_a.unsqueeze(0))[0]
    rot_b = PoseUtils.matrix_from_quat(quat_b.unsqueeze(0))[0]

    delta_pos = pos_b - pos_a
    delta_rot_mat = rot_b.matmul(rot_a.transpose(-1, -2))
    delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat.unsqueeze(0))[0]
    delta_rot = PoseUtils.axis_angle_from_quat(delta_quat.unsqueeze(0))[0]

    return torch.cat([delta_pos, delta_rot]) / scale


def _build_initial_state(ep: h5py.Group) -> dict:
    """Build the InteractiveScene.get_state(is_relative=True)-shaped dict Mimic's
    env.reset_to() needs, from step 0 (the synthetic pre-action snapshot) of a
    recorded episode. Leaf tensors are UN-batched ((7,), (6,), (9,)) — EpisodeData
    adds the leading batch dim itself when pre_export() stacks the single-entry list
    this gets appended to.
    """
    joint_pos0 = torch.as_tensor(ep["joint_pos"][0], dtype=torch.float32)  # (9,) = 7 arm + 2 finger
    num_joints = joint_pos0.shape[0]

    # Franka base is fixed at the env origin with identity orientation on every
    # packing env variant (FRANKA_PANDA_CFG / FRANKA_PANDA_HIGH_PD_CFG never
    # override AssetBaseCfg.InitialStateCfg's default pos=(0,0,0), rot=(1,0,0,0)).
    robot_state = {
        "root_pose": torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        "root_velocity": torch.zeros(6, dtype=torch.float32),
        "joint_position": joint_pos0,
        "joint_velocity": torch.zeros(num_joints, dtype=torch.float32),
    }

    rigid_objects = {}
    for name, pos_key, quat_key in OBJECT_ENTITIES:
        pos0 = torch.as_tensor(ep[pos_key][0], dtype=torch.float32)
        quat0 = torch.as_tensor(ep[quat_key][0], dtype=torch.float32)
        rigid_objects[name] = {
            "root_pose": torch.cat([pos0, quat0]),
            "root_velocity": torch.zeros(6, dtype=torch.float32),
        }

    return {"articulation": {"robot": robot_state}, "rigid_object": rigid_objects}


def _build_actions(ep: h5py.Group, scale: float) -> torch.Tensor:
    """Derive the (T-1, 7) IK-relative actions [dpos(3), daxis_angle(3), gripper(1)]
    from the recorded ee_pos/ee_quat/gripper_cmd trajectory. T recorded steps (step 0
    = pre-action initial state, steps 1..T-1 = post-action states) yield T-1 action
    transitions, matching len(actions) == number of env.step() calls during replay.
    """
    ee_pos = torch.as_tensor(ep["ee_pos"][:], dtype=torch.float32)
    ee_quat = torch.as_tensor(ep["ee_quat"][:], dtype=torch.float32)
    gripper_cmd = torch.as_tensor(ep["gripper_cmd"][:], dtype=torch.float32)

    num_steps = ee_pos.shape[0]
    actions = torch.zeros((num_steps - 1, 7), dtype=torch.float32)
    for i in range(num_steps - 1):
        pose_action = _pose_delta_action(ee_pos[i], ee_quat[i], ee_pos[i + 1], ee_quat[i + 1], scale)
        # gripper_cmd[i + 1]: the command that was actually active while producing
        # the state this action transitions INTO.
        actions[i] = torch.cat([pose_action, gripper_cmd[i + 1 : i + 2]])
    return actions


def main() -> None:
    src = h5py.File(args.input, "r")
    if "episodes" not in src:
        raise ValueError(
            f"{args.input} has no top-level 'episodes' group — not a collect_packing_demos.py dataset."
        )

    episode_names = sorted(src["episodes"].keys())
    handler = HDF5DatasetFileHandler()
    handler.create(args.output, env_name=args.env_name)

    num_written = 0
    num_skipped = 0
    for name in episode_names:
        ep = src["episodes"][name]
        success = bool(ep.attrs.get("success", False))
        if args.only_success and not success:
            num_skipped += 1
            continue
        if "gripper_cmd" not in ep or ep["joint_pos"].shape[0] < 2:
            print(
                f"[convert_to_mimic_hdf5] skipping {name}: missing 'gripper_cmd' field or fewer than 2 "
                "recorded steps — re-collect with the current collect_packing_demos.py (adds gripper_cmd "
                "+ the step-0 initial snapshot) to convert this episode."
            )
            num_skipped += 1
            continue

        episode = EpisodeData()
        episode.seed = 0
        episode.success = success
        episode.add("initial_state", _build_initial_state(ep))
        actions = _build_actions(ep, args.ik_scale)
        for t in range(actions.shape[0]):
            episode.add("actions", actions[t])
        episode.pre_export()
        handler.write_episode(episode)
        num_written += 1

    handler.flush()
    handler.close()
    src.close()

    print(
        f"[convert_to_mimic_hdf5] wrote {num_written} episode(s) to {args.output} "
        f"({num_skipped} skipped) | env_name={args.env_name!r} | ik_scale={args.ik_scale}"
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
