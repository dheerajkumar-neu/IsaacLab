# IsaacLab — Complete Developer Reference Guide

> **Version**: 2.3.2 | **Isaac Sim**: 4.5 / 5.0 / 5.1 | **Python**: 3.11 | **License**: BSD-3 (core), Apache-2.0 (mimic)

This document is a complete reference for understanding and building with IsaacLab. It covers every package, module, class, script, and configuration file in the repository.

---

## Table of Contents

1. [What Is IsaacLab?](#1-what-is-isaaclab)
2. [Repository Layout](#2-repository-layout)
3. [Core Package: `isaaclab`](#3-core-package-isaaclab)
   - [app](#31-app--applauncherpy)
   - [sim](#32-sim--simulation-context--usd-utilities)
   - [scene](#33-scene--interactivescene)
   - [assets](#34-assets--physical-objects)
   - [actuators](#35-actuators)
   - [sensors](#36-sensors)
   - [controllers](#37-controllers)
   - [devices](#38-devices--teleoperation-input)
   - [envs](#39-envs--environment-interfaces)
   - [managers](#310-managers)
   - [terrains](#311-terrains)
   - [markers](#312-markers)
   - [utils](#313-utils)
   - [ui](#314-ui)
4. [Package: `isaaclab_assets`](#4-package-isaaclab_assets)
5. [Package: `isaaclab_tasks`](#5-package-isaaclab_tasks)
   - [Direct Workflow](#51-direct-workflow-tasks)
   - [Manager-Based Workflow](#52-manager-based-workflow-tasks)
6. [Package: `isaaclab_rl`](#6-package-isaaclab_rl)
7. [Package: `isaaclab_mimic`](#7-package-isaaclab_mimic)
8. [Package: `isaaclab_contrib`](#8-package-isaaclab_contrib)
9. [Scripts Directory](#9-scripts-directory)
10. [Apps & Kit Files](#10-apps--kit-files)
11. [Docker & Deployment](#11-docker--deployment)
12. [Tools & Build System](#12-tools--build-system)
13. [Two Environment Workflows Compared](#13-two-environment-workflows-compared)
14. [How to Build Your Own Project](#14-how-to-build-your-own-project)
15. [Key APIs Quick Reference](#15-key-apis-quick-reference)

---

## 1. What Is IsaacLab?

**Isaac Lab** is a GPU-accelerated, open-source robotics simulation framework built on top of [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/). Its purpose is to unify and simplify robotics research workflows including:

- **Reinforcement Learning (RL)** — train policies in massively parallel simulated environments
- **Imitation Learning** — record human demonstrations, generate synthetic data, and train via behavior cloning
- **Motion Planning** — use cuRobo or Pink IK solvers
- **Sim-to-Real Transfer** — accurate physics + sensors to minimize reality gap

Key technical features:
- Runs thousands of robot environments simultaneously on a single GPU using PhysX tensors
- RTX-based realistic sensor simulation (cameras, LiDAR, contact sensors, IMU)
- Supports URDF, MJCF, and USD asset formats
- Wraps popular RL frameworks (RSL-RL, RL-Games, Stable Baselines3, SKRL, Ray RLlib)
- Built on [Omniverse USD](https://openusd.org/) for scene description

The codebase evolved from the [Orbit framework](https://isaac-orbit.github.io/).

---

## 2. Repository Layout

```
IsaacLab/
├── source/                    # All Python packages (pip-installable)
│   ├── isaaclab/              # CORE framework
│   ├── isaaclab_assets/       # Robot & sensor asset configs
│   ├── isaaclab_tasks/        # Ready-to-train task environments
│   ├── isaaclab_rl/           # RL framework wrappers
│   ├── isaaclab_mimic/        # Imitation learning / data generation
│   └── isaaclab_contrib/      # Community extensions
├── scripts/                   # Standalone runnable scripts
│   ├── tutorials/             # Step-by-step learning scripts
│   ├── demos/                 # Robot showcase demos
│   ├── reinforcement_learning/ # RL training/evaluation
│   ├── imitation_learning/    # Demo recording & BC training
│   ├── environments/          # Env interaction utilities
│   ├── tools/                 # Asset conversion & data tools
│   ├── benchmarks/            # Performance profiling
│   └── sim2sim_transfer/      # Cross-simulator policy transfer
├── apps/                      # Isaac Sim .kit application configs
├── docker/                    # Docker/container setup
├── tools/                     # Build/test infrastructure
├── docs/                      # Sphinx documentation source
├── isaaclab.sh                # Master launcher script (Linux)
├── isaaclab.bat               # Master launcher script (Windows)
├── pyproject.toml             # Linter/formatter config
└── VERSION                    # 2.3.2
```

Each directory under `source/` is a self-contained Python package with its own `pyproject.toml` or `config/extension.toml`.

---

## 3. Core Package: `isaaclab`

**Path**: `source/isaaclab/isaaclab/`
**Version**: 0.54.3
**Install**: automatically via `isaaclab.sh -i`
**Dependencies**: numpy, gymnasium==0.29.0, trimesh, websockets, toml, hidapi, prettytable==3.3.0

This is the heart of the framework. It has 14 submodules:

```
isaaclab/
├── app/          # App launcher — MUST be imported first
├── sim/          # Simulation context, USD utilities, spawners
├── scene/        # InteractiveScene — the environment container
├── assets/       # Articulation, RigidObject, DeformableObject
├── actuators/    # Actuator models (PD, neural net, etc.)
├── sensors/      # Camera, RayCaster, ContactSensor, IMU, etc.
├── controllers/  # IK solvers, operational space control
├── devices/      # Keyboard, gamepad, SpaceMouse, VR input
├── envs/         # RL environment base classes + MDP terms
├── managers/     # Action/Reward/Observation/Event managers
├── terrains/     # Procedural terrain generators
├── markers/      # Visual debug markers
├── utils/        # Math, buffers, noise, IO, configclass, etc.
└── ui/           # UI widgets and XR visualizers
```

---

### 3.1 `app` — AppLauncher.py

**What it does**: Initializes the Isaac Sim application before any other imports.

```python
# ALWAYS the very first import in any Isaac Lab script
from isaaclab.app import AppLauncher

args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

`AppLauncher` handles:
- Selecting the `.kit` configuration file (headless vs interactive vs XR)
- Parsing CLI args (`--headless`, `--num_envs`, `--video`, etc.)
- Setting up CUDA device selection
- Enabling/disabling rendering (RTX)

**Key class**: `AppLauncher`

---

### 3.2 `sim` — Simulation Context & USD Utilities

**What it does**: Wraps Isaac Sim's simulation lifecycle and provides USD scene manipulation utilities.

#### `sim.SimulationContext` / `sim.SimulationCfg`
The global simulation controller. Only one instance should exist.

```python
from isaaclab.sim import SimulationContext, SimulationCfg

sim_cfg = SimulationCfg(dt=1/60, render_interval=4)
sim = SimulationContext(sim_cfg)
sim.reset()
while simulation_app.is_running():
    sim.step()
```

`SimulationCfg` fields:
- `dt`: physics timestep (default 1/60 s)
- `render_interval`: render every N physics steps
- `gravity`: gravity vector
- `physics_prim_path`: USD path for PhysicsScene
- `physx`: `PhysxCfg` — solver iterations, GPU memory, etc.
- `rendering`: `RenderCfg` — RTX settings

#### `sim.spawners` — Creating USD Prims
Functions to instantiate geometry, lights, materials, and sensors directly into the stage:

```python
from isaaclab.sim import spawners

# Spawn a sphere
spawners.spawn_sphere("/World/ball", spawners.SphereCfg(radius=0.1))

# Spawn a ground plane
spawners.spawn_ground_plane("/World/ground", spawners.GroundPlaneCfg())

# Spawn from a USD file
spawners.spawn_from_usd("/World/robot", spawners.UsdFileCfg(usd_path="path/to/robot.usd"))
```

Spawner config types:
- `SphereCfg`, `CuboidCfg`, `CylinderCfg`, `ConeCfg`, `CapsuleCfg` — basic shapes
- `GroundPlaneCfg` — infinite ground plane
- `UsdFileCfg` — load any USD file
- `UrdfFileCfg`, `MjcfFileCfg` — load URDF/MJCF (auto-converted to USD)
- `MeshCfg` — triangle mesh
- `DomeLightCfg`, `DistantLightCfg`, `SphereLightCfg`, `DiskLightCfg` — lights

#### `sim.converters` — Asset Format Conversion
```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

cfg = UrdfConverterCfg(asset_path="robot.urdf", usd_dir="output/")
converter = UrdfConverter(cfg)
# Produces a .usd file
```

Converters: `UrdfConverter`, `MeshConverter`, `MjcfConverter`

#### `sim.schemas` — USD Physics Properties
Low-level functions to apply physics schemas to prims:
```python
from isaaclab.sim import schemas
schemas.define_rigid_body_properties(prim_path="/World/ball", mass=1.0)
schemas.activate_contact_sensors(prim_path="/World/robot")
```

#### `sim.utils` — USD Utilities
```python
from isaaclab.sim.utils import (
    find_matching_prims,    # find prims matching a regex pattern
    get_prim_at_path,       # get a USD prim
    set_prim_property,      # set any USD attribute
    apply_nested,           # apply function to prim and descendants
)
```

---

### 3.3 `scene` — InteractiveScene

**What it does**: Central container that holds all simulation entities (robots, objects, sensors, terrain) and manages their lifecycle.

```python
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = GroundPlaneCfg()
    # Robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Sensor
    camera: CameraCfg = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", ...)
    # Object
    cube: RigidObjectCfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube", ...)

scene = InteractiveScene(MySceneCfg(num_envs=4096, env_spacing=2.5))
scene.reset()
# Access entities
robot: Articulation = scene["robot"]
cube: RigidObject = scene["cube"]
```

**Key concept — `{ENV_REGEX_NS}`**: This placeholder expands to `/World/envs/env_.*`, automatically creating N parallel environments. Each entity is vectorized across all environments.

`InteractiveSceneCfg` fields:
- `num_envs`: how many parallel environments
- `env_spacing`: spacing between environment origins
- `lazy_sensor_update`: update sensors only when requested

---

### 3.4 `assets` — Physical Objects

Assets represent simulated physical objects. All assets are vectorized across all environments.

#### `Articulation` — Robots with joints

```python
from isaaclab.assets import Articulation, ArticulationCfg

robot: Articulation = scene["robot"]

# State access (shape: [num_envs, num_dofs])
joint_pos = robot.data.joint_pos          # current joint positions
joint_vel = robot.data.joint_vel          # current joint velocities
root_pos = robot.data.root_pos_w          # root position in world frame
root_quat = robot.data.root_quat_w        # root quaternion in world frame
body_pos = robot.data.body_pos_w          # all body positions

# Set targets
robot.set_joint_position_target(target_pos)   # PD position target
robot.set_joint_velocity_target(target_vel)   # velocity target
robot.set_joint_effort_target(target_torque)  # direct torque
robot.write_data_to_sim()                      # flush to physics

# Reset
root_state = robot.data.default_root_state.clone()
robot.write_root_state_to_sim(root_state)
joint_state = robot.data.default_joint_state.clone()
robot.write_joint_state_to_sim(joint_state[:, :, 0], joint_state[:, :, 1])
```

`ArticulationData` key fields:
- `joint_pos`, `joint_vel`, `joint_acc`, `joint_effort`
- `root_pos_w`, `root_quat_w`, `root_lin_vel_w`, `root_ang_vel_w`
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- `body_pos_b`, `body_quat_b` — in body frame
- `jacobian`, `mass_matrix` — kinematic/dynamic quantities
- `default_joint_pos`, `default_root_state` — reset defaults

#### `RigidObject` — Non-articulated physics objects

```python
from isaaclab.assets import RigidObject, RigidObjectCfg

cube: RigidObject = scene["cube"]
pose = cube.data.root_pos_w          # position [num_envs, 3]
quat = cube.data.root_quat_w         # orientation [num_envs, 4]
lin_vel = cube.data.root_lin_vel_w   # linear velocity

# Teleport
cube.write_root_pose_to_sim(new_pose)
```

#### `DeformableObject` — Soft bodies

```python
from isaaclab.assets import DeformableObject, DeformableObjectCfg
# Accessible nodal positions for deformable mesh
nodal_pos = deformable.data.nodal_pos_w  # [num_envs, num_nodes, 3]
```

#### `RigidObjectCollection` — Batch of rigid objects as one entity

Useful when you have many objects of different types in the same scene and want to manage them as one vectorized batch.

#### `SurfaceGripper` — Suction-based gripper

Applies virtual suction forces to simulate vacuum grippers.

**`ArticulationCfg` key fields**:
```python
ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=UsdFileCfg(usd_path="path/to/robot.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.5),
        joint_pos={".*": 0.0},        # regex → value mapping
    ),
    actuators={
        "arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint.*"], ...),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

---

### 3.5 `actuators`

Actuator models simulate how motor commands translate to physical forces/torques.

| Class | Description |
|-------|-------------|
| `ImplicitActuator` | Physics engine handles PD internally (fastest) |
| `IdealPDActuator` | Explicit PD control computed in Python |
| `DelayedPDActuator` | PD with configurable command delay buffer |
| `RemotizedPDActuator` | Accounts for cable/belt transmission |
| `DCMotor` | DC motor model with torque-speed curve |
| `ActuatorNetMLP` | Neural network actuator model (MLP) |
| `ActuatorNetLSTM` | Neural network actuator model (LSTM) |

```python
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

# Simple implicit actuator (recommended for most tasks)
ImplicitActuatorCfg(
    joint_names_expr=[".*"],    # regex matching joint names
    stiffness=400.0,            # PD gains
    damping=40.0,
    effort_limit=87.0,
    velocity_limit=2.175,
)

# DC motor with torque curve
DCMotorCfg(
    joint_names_expr=[".*_hip_.*"],
    saturation_effort=33.5,
    stiffness=25.0,
    damping=0.5,
)
```

---

### 3.6 `sensors`

All sensors are vectorized — one object manages readings across all environments simultaneously.

#### `Camera` — USD-rendered camera (RGB, depth, segmentation)

```python
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils

camera_cfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/camera",
    update_period=0.1,
    height=480, width=640,
    data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, ...),
)
camera: Camera = scene["camera"]
# Data access
rgb_image = camera.data.output["rgb"]            # [num_envs, H, W, 4]
depth_image = camera.data.output["distance_to_image_plane"]  # [num_envs, H, W, 1]
```

#### `TiledCamera` — GPU-tiled rendering (fastest for many envs)

Uses GPU tiled rendering to render many camera views simultaneously. Much faster than individual Camera instances.

```python
from isaaclab.sensors import TiledCamera, TiledCameraCfg
# Same API as Camera, but uses tiled rendering backend
```

#### `RayCaster` — Distance/proximity sensor

```python
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns

ray_cfg = RayCasterCfg(
    prim_path="/World/envs/env_.*/Robot/base",
    mesh_prim_paths=["/World/ground"],
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
    attach_yaw_only=True,
    debug_vis=True,
)
# Access
distances = ray_caster.data.distances      # [num_envs, num_rays]
ray_hits_w = ray_caster.data.ray_hits_w    # [num_envs, num_rays, 3]
```

Patterns: `GridPatternCfg`, `LidarPatternCfg`, `BpearlPatternCfg`

#### `RayCasterCamera` — Camera using ray-casting (faster but less realistic)

#### `ContactSensor` — Contact force sensing

```python
from isaaclab.sensors import ContactSensor, ContactSensorCfg

contact_cfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/.*foot.*",
    update_period=0.0,
    history_length=3,
    track_air_time=True,
)
# Access
forces = contact_sensor.data.net_forces_w      # [num_envs, num_bodies, 3]
air_time = contact_sensor.data.last_air_time   # [num_envs, num_bodies]
```

#### `FrameTransformer` — Body pose computation

Computes transforms between arbitrary frame pairs — much more efficient than querying USD directly.

```python
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg

ft_cfg = FrameTransformerCfg(
    prim_path="/World/envs/env_.*/Robot/base",
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="/World/envs/env_.*/Robot/.*hand.*",
            name="hand",
            offset=OffsetCfg(pos=(0, 0, 0.1)),
        )
    ],
)
# Access
pos_b = frame_transformer.data.target_pos_source  # [num_envs, num_frames, 3]
quat_b = frame_transformer.data.target_quat_source
```

#### `Imu` — Inertial Measurement Unit

```python
from isaaclab.sensors import Imu, ImuCfg

imu_data = imu.data
lin_acc = imu_data.lin_acc_b   # linear acceleration [num_envs, 3]
ang_vel = imu_data.ang_vel_b   # angular velocity [num_envs, 3]
```

---

### 3.7 `controllers`

#### `DifferentialIKController` — Jacobian-based IK

```python
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

cfg = DifferentialIKControllerCfg(
    command_type="pose",         # "pose" or "position"
    use_relative_mode=True,      # relative or absolute
    ik_method="dls",             # "pinv", "svd", "trans", "dls"
    ik_params={"lambda_val": 0.01},
)
ik_controller = DifferentialIKController(cfg, num_envs=4096, device="cuda")

# Set goal (ee pose relative/absolute)
ik_controller.set_command(delta_pose)   # [num_envs, 6] or [num_envs, 7]

# Compute joint velocities
joint_vel = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
```

#### `OperationalSpaceController` — Full operational space control

Implements full OSC with gravity compensation, inertia shaping, and task/null-space decoupling.

---

### 3.8 `devices` — Teleoperation Input

Devices capture human input for teleoperation demos.

```python
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

keyboard = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
keyboard.reset()

delta_pose, gripper = keyboard.advance()
# delta_pose: [6] (dx, dy, dz, drx, dry, drz)
# gripper: bool (open/close)
```

Available devices:
- `Se2Keyboard` / `Se3Keyboard` — keyboard input (2D / 6D)
- `Se2Gamepad` / `Se3Gamepad` — PS4/Xbox gamepad
- `Se2SpaceMouse` / `Se3SpaceMouse` — 6DOF SpaceMouse
- `HaplyDevice` — Haply Inverse3 haptic device
- `OpenXRDevice` — VR/AR hand tracking via OpenXR
- `ManusVive` — Manus gloves for finger tracking

---

### 3.9 `envs` — Environment Interfaces

Two design patterns for defining environments.

#### Pattern 1: `ManagerBasedRLEnv`

Config-driven. Compose behavior from reusable MDP terms.

```python
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096)
    observations: MyObsCfg = MyObsCfg()
    actions: MyActionsCfg = MyActionsCfg()
    rewards: MyRewardsCfg = MyRewardsCfg()
    terminations: MyTerminationsCfg = MyTerminationsCfg()
    events: MyEventsCfg = MyEventsCfg()
    commands: MyCommandsCfg = MyCommandsCfg()
    curriculum: MyCurriculumCfg = MyCurriculumCfg()

env = ManagerBasedRLEnv(cfg=MyEnvCfg())
obs, info = env.reset()
obs, rew, terminated, truncated, info = env.step(actions)
```

#### Pattern 2: `DirectRLEnv`

Code-driven. Override methods directly (like classic gym environments).

```python
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg

class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1, 1)
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale)

    def _apply_action(self):
        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(self.robot.data)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.robot.data.root_pos_w[:, 2] < 0.1
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        # reset robot state for env_ids
```

#### `DirectMARLEnv` — Multi-agent RL

Extension for multi-agent scenarios where different agents may have different observation/action spaces.

#### MDP Terms (inside `envs.mdp`)

Pre-built building blocks for `ManagerBasedRLEnv`. Organized into categories:

**Actions** (`envs.mdp.actions`):
- `JointPositionAction` — joint position targets
- `JointVelocityAction` — velocity targets
- `JointEffortAction` — torque commands
- `DifferentialInverseKinematicsAction` — end-effector poses → joint targets via IK
- `OperationalSpaceControllerAction` — OSC
- `NonHolonomicAction` — mobile base (velocity commands)
- `BinaryJointPositionAction` — binary gripper open/close
- `WholeBodyAction` — full humanoid control

**Observations** (`envs.mdp.observations`):
- `joint_pos`, `joint_vel`, `joint_pos_rel` — joint state
- `root_pos_w`, `root_quat_w`, `root_lin_vel_b`, `root_ang_vel_b` — base state
- `body_pos_w`, `body_quat_w` — body states
- `base_pos_z` — height
- `image` — camera image
- `generated_commands` — velocity/pose commands

**Rewards** (`envs.mdp.rewards`):
- `track_lin_vel_xy_exp` — track linear velocity command
- `track_ang_vel_z_exp` — track yaw rate command
- `lin_vel_z_l2` — penalize vertical velocity
- `ang_vel_xy_l2` — penalize pitching/rolling
- `joint_deviation_l1` — penalize joint deviations
- `action_rate_l2` — penalize rapid action changes
- `undesired_contacts` — penalize unwanted collisions
- `is_alive` — alive bonus
- `progress_reward` — distance to goal
- `reaching_eucl_reward` — Euclidean distance to target

**Terminations** (`envs.mdp.terminations`):
- `time_out` — episode length exceeded
- `root_height_below_minimum` — fell down
- `bad_orientation` — tipped over
- `joint_pos_out_of_manual_limit` — joint limit violation
- `illegal_contact` — unwanted contact

**Commands** (`envs.mdp.commands`):
- `UniformVelocityCommand` — random velocity targets
- `NormalVelocityCommand` — Gaussian sampled velocity
- `UniformPoseCommand` — random 6D pose targets
- `TerrainBasedPositionCommand` — goal position on terrain

**Events** (`envs.mdp.events`):
- `randomize_rigid_body_mass` — domain randomization
- `randomize_rigid_body_material` — friction/restitution randomization
- `randomize_joint_default_state` — initial state noise
- `push_by_force` — external pushes
- `reset_root_state_uniform` — random reset positions

**Curriculums** (`envs.mdp.curriculums`):
- `terrain_levels_vel` — advance terrain level based on performance

---

### 3.10 `managers`

Managers orchestrate the MDP components in `ManagerBasedRLEnv`.

| Manager | Controls |
|---------|----------|
| `ActionManager` | Translates RL actions to robot commands |
| `ObservationManager` | Collects and concatenates observations |
| `RewardManager` | Computes weighted sum of reward terms |
| `TerminationManager` | Tracks done/truncated conditions |
| `CommandManager` | Manages velocity/pose goal commands |
| `EventManager` | Handles randomization (on reset, on step, on startup) |
| `CurriculumManager` | Adjusts difficulty during training |
| `RecorderManager` | Records trajectory data for datasets |

Configuration pattern:
```python
from isaaclab.managers import RewardTermCfg, ObservationTermCfg
from isaaclab.utils import configclass

@configclass
class MyRewardsCfg:
    # Each attribute is a RewardTermCfg
    track_velocity = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "velocity", "std": 0.5},
    )
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    joint_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
    )
```

`SceneEntityCfg` — reference to a scene entity from within a term:
```python
SceneEntityCfg("robot", joint_names=[".*hip.*"], body_names=["base"])
```

---

### 3.11 `terrains`

Procedural terrain generation for locomotion tasks.

```python
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

terrain_cfg = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
    collision_group=-1,
    num_envs=4096,
    env_spacing=3.0,
)
terrain_importer = TerrainImporter(terrain_cfg)
```

**Height-Field terrain types** (`terrains.height_field`):
- `HfRandomUniformTerrainCfg` — random bumps
- `HfWaveTerrainCfg` — sinusoidal waves
- `HfPyramidSlopedTerrainCfg` / `HfInvertedPyramidSlopedTerrainCfg` — ramps
- `HfPyramidStairsTerrainCfg` / `HfInvertedPyramidStairsTerrainCfg` — stairs
- `HfDiscreteObstaclesTerrainCfg` — random obstacles
- `HfSteppingStonesTerrainCfg` — stepping stones

**Triangle-mesh terrain types** (`terrains.trimesh`):
- `MeshPlaneTerrainCfg` — flat plane
- `MeshBoxTerrainCfg` — boxes
- `MeshGapTerrainCfg` — gaps
- `MeshPitTerrainCfg` — pits
- `MeshPyramidStairsTerrainCfg` / `MeshInvertedPyramidStairsTerrainCfg` — stairs
- `MeshRailsTerrainCfg` — balance beams
- `MeshRepeatedBoxesTerrainCfg` / `MeshRepeatedPyramidsTerrainCfg` / `MeshRepeatedCylindersTerrainCfg`
- `MeshRandomGridTerrainCfg` — random grid
- `MeshFloatingRingTerrainCfg` — floating rings
- `MeshStarTerrainCfg` — star pattern

The `TerrainGenerator` combines multiple sub-terrain types into a curriculum grid:
```python
TerrainGeneratorCfg(
    size=(8.0, 8.0),         # size per terrain cell
    border_width=20.0,
    num_rows=10,              # curriculum levels
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.2),
        "stairs": MeshPyramidStairsTerrainCfg(proportion=0.3, ...),
        "rough": HfRandomUniformTerrainCfg(proportion=0.3, ...),
    },
)
```

---

### 3.12 `markers`

Visual debug markers — rendered in the viewport but don't affect physics.

```python
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, CUBOID_MARKER_CFG

markers = VisualizationMarkers(FRAME_MARKER_CFG.replace(prim_path="/Visuals/frames"))
markers.visualize(translations=pos, orientations=quat)   # [N, 3], [N, 4]
```

Available marker configs: `FRAME_MARKER_CFG`, `CUBOID_MARKER_CFG`, `CYLINDER_MARKER_CFG`, `SPHERE_MARKER_CFG`, `ARROW_X_MARKER_CFG`, `POSITION_GOAL_MARKER_CFG`, `BICOLOR_DIAMOND_MARKER_CFG`

---

### 3.13 `utils`

A wide collection of general-purpose utilities.

#### `configclass` — Dataclass for configs

The most important utility. Extends Python dataclasses with:
- `replace()` — create a modified copy
- Supports nested configs
- Auto-generates `__repr__`

```python
from isaaclab.utils import configclass

@configclass
class MyConfig:
    learning_rate: float = 3e-4
    num_layers: int = 3
    hidden_size: int = 256

cfg = MyConfig()
cfg2 = cfg.replace(learning_rate=1e-3)  # returns modified copy
```

#### `utils.buffers`

```python
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer

# Delay buffer — useful for simulating sensor/actuator delays
delay_buf = DelayBuffer(delay=5, batch_size=4096, device="cuda")
delay_buf.push(current_values)
delayed_values = delay_buf.peek()
```

#### `utils.noise`

```python
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

GaussianNoiseCfg(mean=0.0, std=0.01, operation="add")
UniformNoiseCfg(n_min=-0.05, n_max=0.05, operation="scale")
ConstantNoiseCfg(bias=0.1, operation="add")
```

#### `utils.modifiers`

```python
from isaaclab.utils.modifiers import DigitalFilter, DigitalFilterCfg

# Low-pass filter for smoothing observations
filter_cfg = DigitalFilterCfg(A=[1.0, -0.9], B=[0.1])
filt = DigitalFilter(filter_cfg, data_dim=12, num_envs=4096, device="cuda")
```

#### `utils.math`

```python
from isaaclab.utils.math import (
    quat_mul, quat_inv, quat_rotate,
    euler_xyz_from_quat, quat_from_euler_xyz,
    matrix_from_quat, quat_from_matrix,
    combine_frame_transforms,
    subtract_frame_transforms,
    compute_pose_error,
    apply_delta_pose,
    random_orientation,
)
```

#### `utils.io`

```python
from isaaclab.utils.io import load_yaml, dump_yaml, load_torchscript_model
```

---

### 3.14 `ui`

#### `ui.widgets`

```python
from isaaclab.ui.widgets import ManagerLiveVisualizer, LiveLinePlot
# Real-time visualization in the Isaac Sim UI
```

#### `ui.xr_widgets`

For XR (VR/AR) scenarios — display instructions, manage visualizations, collect data during teleoperation.

---

## 4. Package: `isaaclab_assets`

**Path**: `source/isaaclab_assets/isaaclab_assets/`
**Version**: 0.2.4

Pre-configured robot and sensor configs. These are `ArticulationCfg` / `RigidObjectCfg` instances ready to use in your scene.

### Robots

| Category | Robots |
|----------|--------|
| **Manipulation Arms** | `FRANKA_PANDA_CFG`, `UR10_CFG`, `KINOVA_JACO2_N7S300_CFG`, `SAWYER_CFG` |
| **Manipulation Arms + Hand** | `KUKA_ALLEGRO_CFG` |
| **Dexterous Hands** | `ALLEGRO_HAND_CFG`, `SHADOW_HAND_CFG` |
| **Quadrupeds** | `ANYMAL_B_CFG`, `ANYMAL_C_CFG`, `ANYMAL_D_CFG`, `SPOT_CFG`, `UNITREE_GO2_CFG`, `UNITREE_A1_CFG` |
| **Bipeds / Humanoids** | `UNITREE_G1_CFG`, `UNITREE_H1_CFG`, `UNITREE_H1_INSPIRE_CFG`, `GALBOT_CFG`, `AGIBOT_CFG`, `FOURIER_GR1T2_CFG`, `AGILITY_DIGIT_CFG` |
| **Mobile Manipulation** | `RIDGEBACK_FRANKA_CFG` |
| **Classic Control** | `CARTPOLE_CFG`, `CART_DOUBLE_PENDULUM_CFG`, `ANT_CFG`, `HUMANOID_CFG`, `HUMANOID_28_DOF_CFG` |
| **Aerial** | `CRAZYFLIE_CFG` (quadcopter) |

```python
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# Use in scene
robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
)
```

### Sensors

| Name | Description |
|------|-------------|
| `GELSIGHT_MINI_CFG` | GelSight tactile sensor |
| `VELODYNE_VLP_16_CFG` | Velodyne 16-beam LiDAR (as RayCaster) |

---

## 5. Package: `isaaclab_tasks`

**Path**: `source/isaaclab_tasks/isaaclab_tasks/`
**Version**: 0.11.16

Provides 40+ ready-to-train environments registered as Gymnasium environments.

### 5.1 Direct Workflow Tasks

Single Python file per task. Good for tasks needing custom logic.

| Task | Environment ID | Description |
|------|---------------|-------------|
| **Classic Control** | | |
| CartPole | `Isaac-Cartpole-Direct-v0` | Balance pole on cart |
| CartPole w/ Camera | `Isaac-Cartpole-RGB-Camera-Direct-v0` | Vision-based balance |
| Cart Double Pendulum | `Isaac-Cart-Double-Pendulum-Direct-v0` | Double pendulum |
| Ant | `Isaac-Ant-Direct-v0` | Ant locomotion |
| Humanoid | `Isaac-Humanoid-Direct-v0` | Humanoid walking |
| Humanoid AMP | `Isaac-Humanoid-AMP-Dance-Direct-v0`, etc. | Motion prior imitation |
| **Manipulation** | | |
| Franka Cabinet | `Isaac-Franka-Cabinet-Direct-v0` | Open drawer |
| Allegro Hand | `Isaac-Allegro-Hand-Direct-v0` | Dexterous in-hand manipulation |
| Shadow Hand | `Isaac-Shadow-Hand-Direct-v0` | Shadow hand manipulation |
| Shadow Hand Over | `Isaac-Shadow-Hand-Over-Direct-v0` | Two-hand object passing |
| **Locomotion** | | |
| AnymalC | `Isaac-Anymal-C-Direct-v0` | Quadruped rough terrain |
| **Industrial** | | |
| Factory | `Isaac-Factory-*-Direct-v0` | Gear/nut assembly |
| AutoMate | `Isaac-AutoMate-*-Direct-v0` | Industrial assembly/disassembly |
| Forge | `Isaac-Forge-*-Direct-v0` | Industrial forge tasks |
| **Aerial** | | |
| Quadcopter | `Isaac-Quadcopter-Direct-v0` | Drone position control |

### 5.2 Manager-Based Workflow Tasks

Config-driven. Highly modular — easy to customize individual components.

| Category | Task | Key Environment IDs |
|----------|------|---------------------|
| **Classic** | CartPole | `Isaac-Cartpole-v0`, `Isaac-Cartpole-RGB-v0` |
| | Ant | `Isaac-Ant-v0` |
| | Humanoid | `Isaac-Humanoid-v0` |
| **Locomotion** | Velocity Tracking | `Isaac-Velocity-Rough-Anymal-C-v0`, `Isaac-Velocity-Flat-Unitree-Go2-v0`, etc. |
| **Manipulation** | Reach | `Isaac-Reach-Franka-v0`, `Isaac-Reach-UR10-v0` |
| | Lift | `Isaac-Lift-Cube-Franka-v0` |
| | Stack | `Isaac-Stack-Cube-Franka-IK-Rel-v0` |
| | Cabinet | `Isaac-Open-Drawer-Franka-v0` |
| | In-Hand | `Isaac-Repose-Cube-Allegro-v0` |
| | DexSuite | `Isaac-Shadow-Hand-*-v0` |
| | Pick & Place | Humanoid pick & place (GR1T2, G1) |
| **Navigation** | Point Navigation | Navigation with pre-trained locomotion policy |
| **Drone** | Chase | Drone chasing a target |
| **Loco-Manipulation** | G1 Pick & Place | Combined locomotion + manipulation |

### Task Registration

All environments are registered as Gymnasium environments and can be created:

```python
import gymnasium as gym
import isaaclab_tasks  # triggers registration

env = gym.make("Isaac-Velocity-Rough-Anymal-C-v0", cfg=overridden_cfg)
```

Or load config from registry:
```python
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-v0", num_envs=4096)
```

---

## 6. Package: `isaaclab_rl`

**Path**: `source/isaaclab_rl/`
**Version**: 0.5.1

Wrappers to connect Isaac Lab environments to popular RL training frameworks.

### RSL-RL (recommended for locomotion)

```python
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Wrap Isaac Lab env for RSL-RL
vec_env = RslRlVecEnvWrapper(env)

runner = OnPolicyRunner(vec_env, train_cfg.to_dict(), log_dir=log_dir, device="cuda")
runner.learn(num_learning_iterations=5000, init_at_random_ep_len=True)

# Export trained policy
from isaaclab_rl.rsl_rl.utils import export_policy_as_jit, export_policy_as_onnx
export_policy_as_jit(runner.alg.actor_critic, path="policy.pt")
```

`RslRlRndCfg` — config for Random Network Distillation (curiosity)
`RslRlSymmetryCfg` — config for symmetric policy training

### RL-Games

```python
# Uses rl_games.AlgoObserver pattern
# See scripts/reinforcement_learning/rl_games/train.py
```

### Stable Baselines3

```python
# Wraps env with SB3's VecEnv interface
# See scripts/reinforcement_learning/sb3/train.py
```

### SKRL

```python
# Wraps env with SKRL's IsaacLabWrapper
# See scripts/reinforcement_learning/skrl/train.py
```

---

## 7. Package: `isaaclab_mimic`

**Path**: `source/isaaclab_mimic/`
**Version**: 1.0.16
**License**: Apache-2.0

Enables imitation learning via synthetic data generation. Inspired by the MIMIC framework.

### Workflow

1. **Record human demos** using teleoperation (`scripts/tools/record_demos.py`)
2. **Annotate** subtask boundaries (`scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`)
3. **Generate dataset** — automatically diversify the few demos into thousands (`scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`)
4. **Train** policy via robomimic BC (`scripts/imitation_learning/robomimic/train.py`)
5. **Evaluate** (`scripts/imitation_learning/robomimic/play.py`)

### Core Module: `datagen`

```
isaaclab_mimic/datagen/
├── data_generator.py       # Main data generation orchestrator
├── datagen_info.py         # Per-environment datagen metadata
├── datagen_info_pool.py    # Pool of datagen configs
├── generation.py           # Episode generation logic
├── selection_strategy.py   # Source demo selection
├── waypoint.py             # Waypoint representation
└── utils/                  # Serialization, transforms, etc.
```

### Registered Mimic Environments

Tasks that support mimic data generation:

| Task | Environment ID |
|------|---------------|
| Stack Cube (Franka) | `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0` |
| Stack Cube + Vision | `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0` |
| Stack Cube (Galbot) | `Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-Rel-Mimic-v0` |
| Place Mug (Agibot) | `Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-Rel-Mimic-v0` |
| Place Toy to Box (Agibot) | `Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0` |

### Motion Planning (`motion_planners`)

Integration with **cuRobo** for GPU-accelerated motion planning:
- Used internally during data generation to plan collision-free paths
- Requires CUDA 12.8 and the `Dockerfile.curobo` image

---

## 8. Package: `isaaclab_contrib`

**Path**: `source/isaaclab_contrib/`
**Version**: 0.0.2

Community-contributed extensions that extend the core framework.

### `contrib.assets.Multirotor`

Drone/multirotor asset with rotor configuration:
```python
from isaaclab_contrib.assets import Multirotor, MultirotorCfg
```

### `contrib.actuators.Thruster`

Propeller actuator model that converts motor RPM to thrust and torque:
```python
from isaaclab_contrib.actuators import Thruster, ThrusterCfg
```

### `contrib.sensors.tacsl_sensor`

Visuo-tactile (GelSight-style) sensor simulation using ray casting.

### `contrib.mdp.actions`

Multirotor thrust action for drone control.

---

## 9. Scripts Directory

### 9.1 Tutorials (`scripts/tutorials/`)

Step-by-step learning progression — read in order when starting out.

| Step | Script | What you learn |
|------|--------|----------------|
| `00_sim/create_empty.py` | Create an empty USD stage |
| `00_sim/spawn_prims.py` | Spawn cubes, spheres, lights |
| `00_sim/launch_app.py` | AppLauncher patterns |
| `01_assets/run_rigid_object.py` | Physics rigid bodies |
| `01_assets/run_articulation.py` | Articulated robot control |
| `01_assets/run_deformable_object.py` | Soft body physics |
| `01_assets/run_surface_gripper.py` | Suction gripper |
| `01_assets/add_new_robot.py` | Integrate a custom URDF |
| `02_scene/create_scene.py` | Multi-entity scenes |
| `03_envs/create_cartpole_base_env.py` | Basic RL environment |
| `03_envs/run_cartpole_rl_env.py` | Full RL loop |
| `03_envs/create_quadruped_base_env.py` | Locomotion task |
| `04_sensors/run_ray_caster.py` | Distance sensing |
| `04_sensors/run_usd_camera.py` | Camera rendering |
| `04_sensors/run_frame_transformer.py` | Body frame transforms |
| `05_controllers/run_diff_ik.py` | Differential IK |
| `05_controllers/run_osc.py` | Operational space control |

### 9.2 Demos (`scripts/demos/`)

Full robot showcases. Run with `./isaaclab.sh -p scripts/demos/arms.py`.

| Script | Showcases |
|--------|-----------|
| `arms.py` | Franka, UR10, Kinova arm demos |
| `hands.py` | Allegro, Shadow Hand demos |
| `quadrupeds.py` | Anymal, Spot, Unitree Go2 walking |
| `bipeds.py` | Unitree H1, G1 bipedal walking |
| `pick_and_place.py` | Full pick & place pipeline |
| `bin_packing.py` | Bin packing with multiple objects |
| `multi_asset.py` | Many asset types in one scene |
| `procedural_terrain.py` | Terrain generation showcase |
| `deformables.py` | Cloth, soft body simulation |
| `markers.py` | Debug visualization tools |
| `sensors/cameras.py` | All camera types |
| `sensors/contact_sensor.py` | Contact sensing |
| `sensors/tacsl_sensor.py` | Tactile sensing |

### 9.3 Reinforcement Learning (`scripts/reinforcement_learning/`)

#### RSL-RL (most commonly used with locomotion)
```bash
# Train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 4096 \
    --headless

# Play trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_envs 32 \
    --checkpoint logs/rsl_rl/anymal_c/run/model_5000.pt
```

#### RL-Games
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Cartpole-v0 --headless
```

#### Stable Baselines3
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py \
    --task Isaac-Cartpole-v0 --headless
```

#### SKRL
```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Reach-Franka-v0 --headless
```

#### Distributed Training with Ray
```bash
# Multi-GPU / Multi-Node training
./isaaclab.sh -p scripts/reinforcement_learning/ray/launch.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 \
    --num_workers 4
```

### 9.4 Environments (`scripts/environments/`)

| Script | Purpose |
|--------|---------|
| `list_envs.py` | Print all registered environment IDs |
| `random_agent.py --task <NAME>` | Test env with random actions |
| `zero_agent.py --task <NAME>` | Test env with zero actions |
| `teleop_se3_agent.py --task <NAME>` | Keyboard/gamepad control |
| `state_machine/lift_cube_sm.py` | Scripted state machine for lifting |
| `state_machine/open_cabinet_sm.py` | Scripted cabinet opening |
| `export_IODescriptors.py` | Document env obs/action spaces |

### 9.5 Tools (`scripts/tools/`)

#### Asset Conversion
```bash
# URDF → USD
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    --input my_robot.urdf --output my_robot.usd

# MJCF → USD
./isaaclab.sh -p scripts/tools/convert_mjcf.py \
    --input my_model.xml --output my_model.usd

# Mesh formats → USD
./isaaclab.sh -p scripts/tools/convert_mesh.py \
    --input my_mesh.obj --output my_mesh.usd
```

#### Demo Recording and Replay
```bash
# Record teleoperation demos
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device keyboard

# Replay demos
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --dataset demos.hdf5
```

#### HDF5 / Video Processing
```bash
scripts/tools/hdf5_to_mp4.py     # convert dataset to video
scripts/tools/mp4_to_hdf5.py     # convert video to dataset
scripts/tools/merge_hdf5_datasets.py  # combine datasets
```

### 9.6 Imitation Learning (`scripts/imitation_learning/`)

```bash
# Step 1: Annotate demos with subtask waypoints
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --dataset demos.hdf5

# Step 2: Generate large dataset
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --dataset annotated_demos.hdf5 \
    --output_dataset generated_1000.hdf5 \
    --num_demos 1000

# Step 3: Train with robomimic
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --dataset generated_1000.hdf5

# Step 4: Evaluate
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --checkpoint checkpoints/bc/model_epoch_100.pth
```

### 9.7 Benchmarks (`scripts/benchmarks/`)

| Script | Measures |
|--------|----------|
| `benchmark_rsl_rl.py` | RL training throughput (steps/sec) |
| `benchmark_rlgames.py` | RL-Games training throughput |
| `benchmark_non_rl.py` | Pure physics sim throughput |
| `benchmark_cameras.py` | Camera rendering fps |
| `benchmark_load_robot.py` | Asset loading time |

---

## 10. Apps & Kit Files

`.kit` files are TOML-format Isaac Sim application manifests. They define which extensions to load and their settings.

| File | Use Case |
|------|----------|
| `isaaclab.python.kit` | Interactive training + full GUI |
| `isaaclab.python.headless.kit` | **RL training** — no GUI, fastest |
| `isaaclab.python.headless.rendering.kit` | Headless + RTX rendering (for cameras) |
| `isaaclab.python.rendering.kit` | Interactive + rendering modes |
| `isaaclab.python.xr.openxr.kit` | VR/AR teleoperation |
| `isaaclab.python.xr.openxr.headless.kit` | Headless VR |

The `isaaclab.sh` script selects the appropriate kit file based on CLI args.

---

## 11. Docker & Deployment

### Quick Start with Docker

```bash
# Build the base image
cd docker
python container.py start base

# Enter the container
python container.py enter base

# Inside container — run training
isaaclab -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

### Available Images

| Dockerfile | Image Name | Adds |
|------------|-----------|------|
| `Dockerfile.base` | `isaac-lab-base` | Core IsaacLab + RL frameworks |
| `Dockerfile.curobo` | `isaac-lab-curobo` | + CUDA 12.8 + cuRobo motion planner |
| `Dockerfile.ros2` | `isaac-lab-ros2` | + ROS 2 Humble integration |

### Docker Compose Volumes

Named volumes (persist between container restarts):
- `isaac-cache-kit` — Isaac Kit cache
- `isaac-cache-ov` — Omniverse cache
- `isaac-cache-pip` — pip package cache
- `isaac-logs` — simulator logs
- `isaac-lab-logs` — training logs

Bind mounts (live code updates without rebuild):
- `./source` → `/workspace/isaaclab/source`
- `./scripts` → `/workspace/isaaclab/scripts`

### Cluster Deployment (SLURM/Kubernetes)

```bash
# SLURM
docker/cluster/submit_job_slurm.sh

# Kubernetes via Ray
scripts/reinforcement_learning/ray/grok_cluster_with_kubectl.py
```

---

## 12. Tools & Build System

### `isaaclab.sh` — Master Launcher

The central entrypoint for everything:

```bash
# Install all Python dependencies and RL frameworks
./isaaclab.sh -i [all|none|framework]

# Create conda environment
./isaaclab.sh -c isaaclab

# Create uv virtual environment
./isaaclab.sh -u isaaclab

# Run a Python script using Isaac Sim's Python
./isaaclab.sh -p my_script.py [args...]

# Run all tests
./isaaclab.sh -t

# Run pre-commit formatters/linters
./isaaclab.sh -f

# Build documentation
./isaaclab.sh -d

# Generate VSCode settings (auto-complete, imports)
./isaaclab.sh -v

# Scaffold a new extension project
./isaaclab.sh -n

# Docker management
./isaaclab.sh -o [start|stop|push|...]
```

### `tools/install_deps.py`

Installs apt and ROS package dependencies declared in each package's `extension.toml`.

### `tools/run_all_tests.py`

Discovers and runs all pytest tests in the repository, with Isaac Sim initialization.

### `pyproject.toml` — Code Quality

- **Ruff**: linter + formatter (line-length 120, McCabe complexity ≤ 30)
- **Pyright**: type checker (basic mode)
- **Codespell**: spell checker
- **Pre-commit**: hooks for all of the above

---

## 13. Two Environment Workflows Compared

| Aspect | `DirectRLEnv` | `ManagerBasedRLEnv` |
|--------|--------------|---------------------|
| **Code style** | Override Python methods | Compose config objects |
| **Customization** | Full freedom, write any logic | Swap/add terms in cfg |
| **Reusability** | Lower (logic is in methods) | High (MDP terms are reusable) |
| **Boilerplate** | Less upfront | More upfront |
| **Best for** | Novel tasks, custom physics interactions | Standard RL tasks, research variants |
| **Examples** | `Cartpole`, `AnymalC`, `Franka Cabinet` | `Locomotion`, `Reach`, `Lift` |
| **Multi-agent** | `DirectMARLEnv` | Not natively |
| **Mimic support** | Limited | `ManagerBasedRLMimicEnv` |

---

## 14. How to Build Your Own Project

### Option A: Scaffold a new extension (recommended)

```bash
./isaaclab.sh -n
# Follow the prompts to generate a template project
```

This creates a project with the correct structure:
```
my_project/
├── config/extension.toml
├── my_project/
│   ├── __init__.py
│   ├── tasks/
│   │   └── my_task/
│   │       ├── __init__.py
│   │       ├── my_env.py          # DirectRLEnv or cfg file
│   │       └── agents/
│   │           └── rsl_rl_ppo_cfg.py
│   └── assets/
└── scripts/
    └── train.py
```

### Option B: Minimal custom DirectRLEnv

```python
# my_env.py
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    # Env dimensions
    observation_space = 18   # total obs size
    action_space = 7         # num joints
    state_space = 0          # for MARL critics (0 if not used)

    # Simulation
    decimation = 4           # policy runs every 4 physics steps
    episode_length_s = 5.0

    # Scene
    num_envs = 4096
    env_spacing = 2.5

    # Asset
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    action_scale = 7.5

class MyEnv(DirectRLEnv):
    cfg: MyEnvCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        # Add terrain, lights etc.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1, 1)
        targets = self.robot.data.default_joint_pos + self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target(targets)

    def _apply_action(self):
        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self.robot.data.joint_pos - self.robot.data.default_joint_pos,
            self.robot.data.joint_vel,
            self.actions,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Your reward function
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(truncated), truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
```

### Option C: Minimal Manager-Based Env

```python
# my_env_cfg.py
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import *
from isaaclab.envs import mdp
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

@configclass
class MySceneCfg(InteractiveSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = GroundPlaneCfg()

@configclass
class MyObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)

    policy: PolicyCfg = PolicyCfg()

@configclass
class MyActionsCfg:
    arm = ActionTermCfg(
        func=mdp.JointPositionAction,
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
    )

@configclass
class MyRewardsCfg:
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)

@configclass
class MyTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    scene = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations = MyObsCfg()
    actions = MyActionsCfg()
    rewards = MyRewardsCfg()
    terminations = MyTerminationsCfg()
    decimation = 4
    episode_length_s = 5.0
```

### Running Your Custom Task

```python
# train.py
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from my_env_cfg import MyEnvCfg

env = ManagerBasedRLEnv(cfg=MyEnvCfg())
obs, _ = env.reset()

for _ in range(10000):
    actions = env.action_space.sample()  # random policy
    obs, rew, done, trunc, info = env.step(actions)

env.close()
simulation_app.close()
```

### RSL-RL Agent Config

```python
# agents/rsl_rl_ppo_cfg.py
from rsl_rl.runners import OnPolicyRunner
from dataclasses import dataclass, field

@dataclass
class MyPPOCfg:
    seed = 42
    device = "cuda"

    class runner_cfg:
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 50
        experiment_name = "my_task"
        run_name = ""
        logger = "tensorboard"
        neptune_project = "isaaclab"
        wandb_project = "isaaclab"
        resume = False
        load_run = -1
        load_checkpoint = -1

    class policy_cfg:
        class_name = "ActorCritic"
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm_cfg:
        class_name = "PPO"
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
```

---

## 15. Key APIs Quick Reference

### Imports Cheat Sheet

```python
# App (FIRST)
from isaaclab.app import AppLauncher

# Simulation
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.sim import spawners as sim_utils
from isaaclab.sim.converters import UrdfConverter

# Scene
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Assets
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.assets import DeformableObject, DeformableObjectCfg

# Actuators
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

# Sensors
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors import Imu, ImuCfg

# Controllers
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.controllers import OperationalSpaceController

# Environments
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp

# Managers (for ManagerBased)
from isaaclab.managers import (
    ActionTermCfg, RewardTermCfg, ObservationTermCfg,
    ObservationGroupCfg, TerminationTermCfg, EventTermCfg,
    CommandTermCfg, CurriculumTermCfg, SceneEntityCfg
)

# Terrains
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg

# Markers
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

# Utils
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, euler_xyz_from_quat, compute_pose_error
from isaaclab.utils.buffers import DelayBuffer, CircularBuffer
from isaaclab.utils.noise import GaussianNoiseCfg

# Devices
from isaaclab.devices import Se3Keyboard

# Pre-configured robots
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG, UNITREE_G1_CFG
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

# RL wrappers
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Task utilities
from isaaclab_tasks.utils import parse_env_cfg
```

### Environment Step Loop

```python
obs, info = env.reset()
while simulation_app.is_running():
    with torch.inference_mode():
        actions = policy(obs["policy"])
    obs, rew, terminated, truncated, info = env.step(actions)
```

### Vectorized Reset Pattern

```python
def _reset_idx(self, env_ids: torch.Tensor):
    super()._reset_idx(env_ids)

    # Reset robot
    root_state = self.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += self.scene.env_origins[env_ids]  # add env offset!
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    joint_pos, joint_vel = self.robot.data.default_joint_pos[env_ids], torch.zeros(...)
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Reset tracked quantities
    self.prev_actions[env_ids] = 0
    self.episode_length_buf[env_ids] = 0
```

---

*Generated from full repository exploration of IsaacLab v2.3.2 at `/groot/data/ubuntu/IsaacLab/`*
