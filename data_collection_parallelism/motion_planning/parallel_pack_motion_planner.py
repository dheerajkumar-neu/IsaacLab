# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
ParallelPackMotionPlanner — one PackMotionPlanner per environment, all living in
the same Isaac Sim process / CUDA context.

Scoping note (see project discussion): this is NOT cuRobo's fused batched-planning
API (MotionGen.plan_batch_env, which plans all N environments in a single GPU call
against N per-env collision worlds). That would require rewriting CuroboPlanner's
world-model construction to build a proper batched WorldConfig list instead of one
world per instance, plus reworking per-env attachment/dynamic-object-sync so it
still applies to whichever env just grasped. That rewrite touches
isaaclab_mimic's shared CuroboPlanner (used elsewhere in the repo) and cannot be
safely verified without many live Isaac Sim iterations.

What this DOES give you: N independent CuroboPlanner instances (each with its own
MotionGen, own warmup, own collision world scoped to /World/envs/env_{i}) that all
share ONE Isaac Sim process, ONE CUDA context, and ONE GraspGenX server connection —
so you pay Kit-boot + GraspGenX-model-load + camera-pipeline-setup ONCE for N
environments instead of N times, and N robots run concurrently in sim time. cuRobo's
own per-instance warmup/setup cost is still paid once per env (that part doesn't
shrink under this design) — the fused-batch approach above is the further upgrade
if that warmup cost or per-step planning throughput becomes the bottleneck.
"""

from __future__ import annotations

import logging

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from data_collection_parallelism.motion_planning.pack_motion_planner import PackMotionPlanner


class ParallelPackMotionPlanner:
    """Container of ``num_envs`` independent :class:`PackMotionPlanner` instances.

    Indexing (``planners[env_id]``) returns the planner bound to that environment.
    Each instance owns its own CuRobo MotionGen and collision world, scoped to
    ``/World/envs/env_{env_id}`` — planning calls for different env_ids do not
    interact and can be issued in any order.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        num_envs: int,
        num_trajopt_seeds: int | None = None,
        num_graph_seeds: int | None = None,
    ) -> None:
        """See PackMotionPlanner's docstring for the num_trajopt_seeds/num_graph_seeds
        VRAM-vs-planning-robustness tradeoff — passed through unchanged to every
        per-env instance. None (default) keeps CuroboPlannerCfg's factory default (12)."""
        self.num_envs = num_envs
        self._planners: list[PackMotionPlanner] = []
        for env_id in range(num_envs):
            logging.info("Constructing PackMotionPlanner for env_id=%d/%d ...", env_id, num_envs - 1)
            planner = PackMotionPlanner(
                env=env, robot=robot, env_id=env_id,
                num_trajopt_seeds=num_trajopt_seeds, num_graph_seeds=num_graph_seeds,
            )
            # Surface cuRobo's planning-failure reasons (result.status, per-phase
            # failures) — CuroboPlanner logs them at DEBUG on its own per-env logger,
            # which defaults to INFO (mirrors collect_packing_demos.py's single-env setup).
            logging.getLogger(f"CuroboPlanner_{env_id}").setLevel(logging.DEBUG)
            self._planners.append(planner)
        logging.info("ParallelPackMotionPlanner ready: %d planner instances constructed.", num_envs)

    def __getitem__(self, env_id: int) -> PackMotionPlanner:
        return self._planners[env_id]

    def __len__(self) -> int:
        return self.num_envs

    def __iter__(self):
        return iter(self._planners)
