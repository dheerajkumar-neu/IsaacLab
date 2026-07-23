# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
data_collection_parallelism — multi-environment autonomous demonstration collection
for the packing task (Isaac-Pack-Object-Franka-Camera-v0), one shared Isaac Sim
process with num_envs=N.

Pipeline:
  GraspGenX server -> GraspResult -> filter -> CuRobo motion planner (one per env)
  -> IsaacLab vectorized env (N envs stepped together) -> GR00T LeRobot dataset

This is a separate pipeline from data_collection/ (which is single-env,
num_envs=1) — see data_collection_parallelism/scripts/collect_packing_demos_parallel.py
for the entry point and design notes.
"""
