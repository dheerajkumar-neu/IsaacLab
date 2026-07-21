# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.isaaclab_int.config.franka.franka_pack_ik_rel_visuomotor_env_cfg import (
    FrankaPackIKRelVisuomotorEnvCfg,
)


@configclass
class FrankaPackIKRelVisuomotorMimicEnvCfg(FrankaPackIKRelVisuomotorEnvCfg, MimicEnvCfg):
    """Isaac Lab Mimic environment config for Franka packing with IK-relative control
    and wrist/table-top RGB observations (for VLA/visuomotor training data).

    Subtask structure is identical to ``FrankaPackIKRelMimicEnvCfg`` — see that file
    for the per-subtask breakdown (home -> grasp -> bin -> drop -> home, x3 objects).
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_packing_franka_visuomotor_task_D0"
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

        # Subtask 0 -- grasp object_1
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

        # Subtask 1 -- carry object_1 to bin and drop
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

        # Subtask 2 -- return home + approach + grasp object_2
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

        # Subtask 3 -- carry object_2 to bin and drop
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

        # Subtask 4 -- return home + approach + grasp object_3
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

        # Subtask 5 (final) -- carry object_3 to bin and drop
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
