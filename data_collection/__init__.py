# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
data_collection — autonomous demonstration collection for the packing task.

Pipeline:
  GraspGenX server → GraspResult → filter → CuRobo motion planner → IsaacLab env → HDF5
"""
