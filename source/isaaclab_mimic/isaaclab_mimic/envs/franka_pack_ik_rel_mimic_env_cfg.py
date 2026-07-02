# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.isaaclab_int.config.franka.franka_pack_ik_rel_env_cfg import FrankaPackEnvCfg


@configclass
class FrankaPackIKRelMimicEnvCfg(FrankaPackEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic config for Franka packing with IK-relative control.

    Subtask structure for each episode (home -> grasp -> bin -> drop -> home, x3 objects    ):
        0. grasp_1  -- approach and grasp object_1  (object_ref = object_1)
        1. place_1  -- carry object_1 to bin, drop  (object_ref = packing_bin)
        2. grasp_2  -- return home + approach object_2 (object_ref = object_2)
        3. place_2  -- carry object_2 to bin, drop  (object_ref = packing_bin)
        4. grasp_3  -- return home + approach object_3 (object_ref = object_3)
        5. [final]  -- carry object_3 to bin, drop  (object_ref = packing_bin, signal = None)
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_packing_franka_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = []

        # Subtask 0 -- grasp object _1
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object_01",
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp object 1",
                next_subtask_description="Carry object 1 to bin and release",
            )
        )

        # Subtask 1 -- carry cube_1 to bin and drop
        subtask_configs.append(
            SubTaskConfig(
                object_ref="packing_bin",
                subtask_term_signal="place_1",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place object 1 in bin",
                next_subtask_description="Return home, approach and grasp object 2",
            )
        )

        # Subtask 2 -- return home + approach + grasp cube_2
        # The home-return path is absorbed into the beginning of this segment.
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object_02",
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp object 2",
                next_subtask_description="Carry object 2 to bin and release",
            )
        )

        # Subtask 3 -- carry cube_2 to bin and drop
        subtask_configs.append(
            SubTaskConfig(
                object_ref="packing_bin",
                subtask_term_signal="place_2",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place object 2 in bin",
                next_subtask_description="Return home, approach and grasp object 3",
            )
        )

        # Subtask 4 -- return home + approach + grasp cube_3
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object_03",
                subtask_term_signal="grasp_3",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp object 3",
                next_subtask_description="Carry object 3 to bin and release",
            )
        )

        # Subtask 5 (final) -- carry cube_3 to bin and drop
        subtask_configs.append(
            SubTaskConfig(
                object_ref="packing_bin",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place object 3 in bin",
            )
        )

        self.subtask_configs["franka"] = subtask_configs
