# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Pack-Object-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pack_joint_pos_env_cfg:FrankaPackEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Pack-Object-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pack_ik_rel_env_cfg:FrankaPackEnvCfg",
    },
    disable_env_checker=True,
)

##
# Joint Position Control + Cameras (RealSense D435-like, for dataset recording)
##

gym.register(
    id="Isaac-Pack-Object-Franka-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pack_camera_env_cfg:FrankaPackCameraEnvCfg",
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control + Cameras (RealSense D435-like, for dataset recording)
##

gym.register(
    id="Isaac-Pack-Object-Franka-IK-Rel-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_pack_camera_env_cfg:FrankaPackIKRelCameraEnvCfg",
    },
    disable_env_checker=True,
)
