# Project Spec: VLA Data Collection Environment (Isaac Lab)

## Goal

Build a simulation-based data collection environment using Isaac Lab and Isaac Sim.
The robot will perform a long-horizon packing task: pick all objects from a table and place them into a basket.
Collected demonstrations (images + robot state + actions) will be used to fine-tune a VLA (Vision-Language-Action) model please follow pi-0 VLA models dataset requirements.

This is NOT an RL training project. There is no reward signal, no policy gradient, no gym training loop.
The environment uses Isaac Lab's ManagerBasedRLEnv infrastructure purely for:
- Parallel episode management across num_envs
- Per-env termination detection
- Per-env reset and domain randomization (EventManager)
- Observation and action management across envs

The "policy" driving the robot is a scripted motion primitive, not a neural network.

---

## Phase Plan

Start simple. Build and verify the full task end-to-end first. Do NOT decompose into subtasks yet.

Full task flow per episode:
  reset() → script scans table → picks object 1 → places in basket
           → picks object 2 → places in basket
           → ... until table is empty → termination fires → save episode → reset()


---

## Technical Stack

- Isaac Sim (NVIDIA Omniverse) — physics simulation, rendering
- Isaac Lab — scene management, sensor APIs, manager infrastructure
- Python 3.11
- HDF5 (h5py) — intermediate on-disk buffer during collection (fast GPU writes)
- LeRobot (HuggingFace) — final dataset format for pi-0 fine-tuning (Parquet + MP4)
- Franka Panda — robot arm (7-DOF + gripper) + cameras (Intel RealSense: RGB + Depth)

---

## Robot and Scene

**Robot:** Franka Panda
- Mounted at origin (0, 0, 0)
- Controlled via end-effector delta position (EE delta control) + gripper open/close
- Wrist camera attached to `panda_hand` link

**Scene:**
- Table: static kinematic object, surface at z=0.75m
- Objects on table: 3-5 rigid body objects (cubes, cylinders, small bottles)
  - Spawned at randomized positions on table surface per episode
  - Mass: ~0.1 kg each
- Basket: static kinematic object at fixed position on table edge
- Ground plane + dome light

**Cameras:**
- Overhead camera: fixed in world frame, 1.5m above table, pointing down
  - Resolution: 640x480, RGB + Depth
- Wrist camera: attached to panda_hand link, moves with robot
  - Resolution: 640x480, RGB + Depth
  - Offset: 5cm forward from hand, pointing toward fingers

---

## Folder Structure

Validated against the IsaacLab repo. Place the package at:
`/groot/data/ubuntu/IsaacLab/source/packing_collector/`

This mirrors `isaaclab_mimic/` which also handles data pipelines and lives at the same `source/` level — NOT inside `isaaclab_tasks/` (which is for task/env definitions only).

```
source/packing_collector/
├── config/
│   └── extension.toml            # REQUIRED — Isaac Lab extension metadata + dependencies
├── packing_collector/            # Python package root
│   ├── __init__.py               # gym.register("Isaac-Packing-Franka-v0") goes here
│   ├── config/
│   │   ├── __init__.py
│   │   ├── packing_env_cfg.py    # ManagerBasedRLEnvCfg subclass — full scene config
│   │   └── mdp/
│   │       ├── __init__.py
│   │       ├── observations.py   # overhead_rgb, wrist_rgb, joint_pos, ee_pose
│   │       ├── actions.py        # EE delta control + gripper binary
│   │       ├── events.py         # randomize object poses on reset
│   │       └── terminations.py   # all_objects_in_basket(), time_out()
│   │       # NO rewards.py — not needed
│   │
│   ├── control/
│   │   ├── __init__.py
│   │   └── scripted_policy.py    # full pick-and-place loop for all objects
│   │                             # motion primitives: move_to(), grasp(), release()
│   │
│   └── recording/
│       ├── __init__.py
│       └── recorder.py           # per-env HDF5 buffer (intermediate) + LeRobot converter
│                                 # wraps isaaclab.utils.datasets.HDF5DatasetFileHandler
│
├── setup.py                      # REQUIRED — follows isaaclab_tasks/setup.py pattern
└── scripts/
    ├── collect.py                # entry point: run N episodes, write intermediate HDF5
    └── convert_to_lerobot.py     # convert HDF5 → LeRobot Parquet+MP4 for pi-0 training
```

---

## File Specifications

### config/packing_env_cfg.py

Defines `PackingEnvCfg(ManagerBasedRLEnvCfg)` with:

Scene contents:
- `ground`: GroundPlaneCfg
- `dome_light`: DomeLightCfg(intensity=2000.0)
- `robot`: ArticulationCfg using FRANKA_PANDA_CFG
  - import: `from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG`
  - prim_path: "{ENV_REGEX_NS}/Robot"
  - init pos: (0, 0, 0)
- `table`: UsdFileCfg or CuboidCfg (kinematic=True)
  - position: (0.5, 0.0, 0.375) — center of table, 75cm tall
- `basket`: RigidObjectCfg with UsdFileCfg or CuboidCfg (kinematic=True)
  - position: (0.5, 0.45, 0.80) — on table edge
- `object_0` through `object_4`: RigidObjectCfg
  - Each a CuboidCfg (5x5x5 cm) or SphereCfg or CylinderCfg
  - Different colors per object
  - Initial positions randomized by EventManager on reset
- `overhead_cam`: CameraCfg
  - prim_path: "{ENV_REGEX_NS}/OverheadCam"
  - offset pos: (0.5, 0.0, 1.5), pointing down
  - 640x480, rgb + depth, 10Hz
- `wrist_cam`: CameraCfg
  - prim_path: "{ENV_REGEX_NS}/Robot/panda_hand/WristCam"
  - offset from hand: (0, 0, 0.05), convention="ros"
  - 640x480, rgb + depth, 10Hz

Env settings:
- `num_envs`: 64 (configurable)
- `episode_length_s`: 60.0 seconds
- `sim.dt`: 0.01 (100Hz physics)
- `decimation`: 10 (10Hz policy)

Manager configs:
- `observations`: ObservationsCfg (see mdp/observations.py)
- `actions`: ActionsCfg (see mdp/actions.py)
- `events`: EventCfg (see mdp/events.py)
- `terminations`: TerminationsCfg (see mdp/terminations.py)

---

### config/mdp/observations.py

Define `ObservationsCfg` with one group `policy`:

Terms:
- `overhead_rgb`: raw RGB tensor from overhead_cam via `mdp.image` — shape (H, W, 3)
  - registered as `ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("overhead_cam"), "data_type": "rgb", "normalize": False})`
- `wrist_rgb`: raw RGB tensor from wrist_cam via `mdp.image` — shape (H, W, 3)
- `joint_pos`: robot joint positions — shape (7,)
- `joint_vel`: robot joint velocities — shape (7,)
- `ee_pose`: end-effector position + quaternion in world frame — shape (7,)
- `gripper_state`: gripper open amount — shape (1,)
- `object_positions`: all object positions in world frame — shape (num_objects, 3)

---

### config/mdp/actions.py

Define `ActionsCfg` with:
- `ee_delta`: EeDeltaActionCfg — 6D EE delta (dx, dy, dz, droll, dpitch, dyaw)
- `gripper`: BinaryJointPositionActionCfg — open (1.0) or close (0.0)

---

### config/mdp/events.py

Define `EventCfg` with:

On reset:
- `reset_robot_joints`: reset robot to home joint configuration
- `randomize_object_positions`: for each object, sample random (x, y) on table surface
  - x range: [0.3, 0.7], y range: [-0.3, 0.3], z = table_surface + half_object_height
  - ensure minimum separation of 0.08m between objects
- `randomize_object_orientations`: random yaw rotation per object

---

### config/mdp/terminations.py

Define `TerminationsCfg` with:

- `time_out`: `mdp.time_out` — fires when episode_length_s exceeded
- `all_objects_packed`: custom term that checks if all object positions are
  within the basket bounding box (x: ±0.15, y: ±0.12, z above basket floor)
  Returns done=True when all N objects satisfy this condition.

---

### control/scripted_policy.py

Class `ScriptedPackingPolicy`:

```
__init__(env):
  store env reference
  build list of object prim paths to track
  define home_joint_pos (safe neutral pose)

get_action(obs) -> action_tensor:
  internal state machine per env:
    state 0 - FIND_NEXT_OBJECT:
      from obs.object_positions, find nearest unpacked object
      set target = object position
      transition to APPROACH

    state 1 - APPROACH:
      compute EE delta toward target position + 10cm above
      if ee_pos close enough to pre-grasp pose: transition to DESCEND

    state 2 - DESCEND:
      compute EE delta downward toward object
      set gripper = open
      if ee_pos at grasp height: transition to GRASP

    state 3 - GRASP:
      set gripper = close
      wait N steps for gripper to close
      transition to LIFT

    state 4 - LIFT:
      compute EE delta upward (+20cm)
      if ee_pos high enough: transition to MOVE_TO_BASKET

    state 5 - MOVE_TO_BASKET:
      compute EE delta toward basket position + 15cm above
      if close enough: transition to DESCEND_TO_BASKET

    state 6 - DESCEND_TO_BASKET:
      compute EE delta downward into basket
      if low enough: transition to RELEASE

    state 7 - RELEASE:
      set gripper = open
      wait N steps
      mark current object as packed
      transition to LIFT_AWAY

    state 8 - LIFT_AWAY:
      compute EE delta upward
      if done, check if more objects remain:
        yes → transition to FIND_NEXT_OBJECT
        no  → hold position (episode will terminate via termination manager)

IK solving: use Isaac Lab's DifferentialIKController for EE delta → joint delta conversion
All state transitions are per-env (vectorized across num_envs)
```

---

### recording/recorder.py

Wraps `isaaclab.utils.datasets.HDF5DatasetFileHandler` (already in the repo at
`source/isaaclab/isaaclab/utils/datasets/`) rather than reimplementing HDF5 logic.
Writes intermediate HDF5 files during collection (fast GPU-side buffering).
Final conversion to LeRobot format happens in `scripts/convert_to_lerobot.py`.

Class `EpisodeRecorder`:

```
__init__(output_dir, num_envs):
  initialize per-env buffers: dict of lists
  each buffer holds: overhead_rgb, wrist_rgb, joint_pos, ee_pose, action, gripper

step(obs, action, done_mask):
  for each env: append current (obs, action) to that env's buffer
  for envs where done_mask=True: flush buffer to HDF5, clear buffer

flush_to_hdf5(env_idx, episode_idx):
  open file: output_dir/intermediate/episode_{episode_idx:06d}.hdf5
  write datasets:
    observations/overhead_rgb:  (T, 480, 640, 3)  uint8
    observations/wrist_rgb:     (T, 480, 640, 3)  uint8
    observations/joint_pos:     (T, 7)             float32
    observations/ee_pose:       (T, 7)             float32
    actions/ee_delta:           (T, 6)             float32
    actions/gripper:            (T, 1)             float32
    metadata/episode_length:    scalar
    metadata/success:           bool (was termination from all_objects_packed?)
    metadata/num_objects:       scalar
  note: keep all tensors on GPU until this flush call (.cpu() only here)
```

---

### scripts/collect.py

Entry point. Writes intermediate HDF5 files; call convert_to_lerobot.py after to produce
the final LeRobot dataset.

```
args:
  --num-envs      int    default=64
  --num-episodes  int    default=1000
  --output-dir    str    default="./dataset/intermediate"
  --task          str    default="Isaac-Packing-Franka-v0"
  --headless      bool   default=True

main():
  launch SimulationApp (headless if specified)
  env = gym.make(args.task, num_envs=args.num_envs)
  policy = ScriptedPackingPolicy(env)
  recorder = EpisodeRecorder(args.output_dir, args.num_envs)

  obs, _ = env.reset()
  episode_count = 0

  while episode_count < args.num_episodes:
      action = policy.get_action(obs)
      obs, _, terminated, truncated, _ = env.step(action)
      done = terminated | truncated
      recorder.step(obs, action, done)
      episode_count += done.sum().item()
      print(f"Episodes collected: {episode_count}/{args.num_episodes}")

  env.close()
```

### scripts/convert_to_lerobot.py

Converts intermediate HDF5 files to LeRobot Parquet+MP4 format for pi-0 fine-tuning.
Mirrors the pattern in openpi `examples/libero/convert_libero_data_to_lerobot.py`.

```
args:
  --input-dir     str    path to intermediate HDF5 files
  --output-dir    str    path for LeRobot dataset output
  --task-desc     str    language instruction string (e.g. "pick all objects and place in basket")
  --fps           int    default=10

main():
  dataset = LeRobotDataset.create(
      repo_id="local/packing_dataset",
      robot_type="panda",
      fps=args.fps,
      features={
          "observation.images.overhead_rgb": {dtype: "video", shape: (480, 640, 3)},
          "observation.images.wrist_rgb":    {dtype: "video", shape: (480, 640, 3)},
          "observation.state":               {dtype: "float32", shape: (14,)},
            # concatenation of joint_pos(7) + ee_pose(7)
          "action":                          {dtype: "float32", shape: (7,)},
            # concatenation of ee_delta(6) + gripper(1)
      }
  )

  for hdf5_file in sorted(input_dir.glob("episode_*.hdf5")):
      with h5py.File(hdf5_file) as f:
          T = f["metadata/episode_length"][()]
          for t in range(T):
              dataset.add_frame({
                  "observation.images.overhead_rgb": f["observations/overhead_rgb"][t],
                  "observation.images.wrist_rgb":    f["observations/wrist_rgb"][t],
                  "observation.state": np.concatenate([
                      f["observations/joint_pos"][t],   # (7,)
                      f["observations/ee_pose"][t],     # (7,)
                  ]),
                  "action": np.concatenate([
                      f["actions/ee_delta"][t],         # (6,)
                      f["actions/gripper"][t],          # (1,)
                  ]),
              })
          dataset.save_episode(task=args.task_desc)

  dataset.finalize()
  # also writes meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl,
  # meta/episodes_stats.jsonl (normalization stats required by pi-0 training)
```

---

## Dataset Structure

### Stage 1 — Intermediate HDF5 (written during collection)

Fast on-disk buffer, one file per episode. NOT used directly for training.

```
dataset/intermediate/
  episode_000000.hdf5
  episode_000001.hdf5
  ...

Each file:
  observations/
    overhead_rgb    (T, 480, 640, 3)   uint8
    wrist_rgb       (T, 480, 640, 3)   uint8
    joint_pos       (T, 7)             float32
    ee_pose         (T, 7)             float32   [x,y,z, qw,qx,qy,qz]
  actions/
    ee_delta        (T, 6)             float32   [dx,dy,dz, droll,dpitch,dyaw]
    gripper         (T, 1)             float32   [0=closed, 1=open]
  metadata/
    episode_length  scalar             int
    success         scalar             bool
    num_objects     scalar             int
```

### Stage 2 — LeRobot Dataset (final, for pi-0 fine-tuning)

Pi-0 fine-tuning uses LeRobot (HuggingFace) as its primary interface.
LeRobot does NOT use HDF5 — it uses Parquet (tabular data) + MP4 (video).
Run `scripts/convert_to_lerobot.py` after collection to produce this layout.

```
dataset/lerobot/
├── meta/
│   ├── info.json             # feature schema, fps=10, robot_type="panda", codebase_version="v2.1"
│   ├── tasks.jsonl           # {"task_index": 0, "task": "pick all objects and place in basket"}
│   ├── episodes.jsonl        # {episode_index, length, task_index, success}
│   └── episodes_stats.jsonl  # per-episode normalization stats (required by pi-0 training)
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet   # tabular: observation.state, action, timestamp,
│       │                            #          frame_index, episode_index, task_index, next.done
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.overhead_rgb/
        │   └── episode_000000.mp4   # H.264, 480×640, 10fps
        └── observation.images.wrist_rgb/
            └── episode_000000.mp4

Parquet columns per frame:
  observation.state   float32 (14,)  — joint_pos(7) + ee_pose(7) concatenated
  action              float32 (7,)   — ee_delta(6) + gripper(1) concatenated
  timestamp           float          — seconds since episode start
  frame_index         int            — frame index within episode
  episode_index       int            — episode identifier
  task_index          int            — links to tasks.jsonl language instruction
  next.done           bool           — True on final frame of episode

Image keys (referenced in Parquet, stored as MP4):
  observation.images.overhead_rgb   — (480, 640, 3) RGB uint8, encoded as MP4
  observation.images.wrist_rgb      — (480, 640, 3) RGB uint8, encoded as MP4
  note: pi-0 internally resizes all images to 224×224 during training

Action/state dimension limits (pi-0 hard constraints):
  max_action_dim = 32   (7 used here — well within limit)
  max_state_dim  = 32   (14 used here — well within limit)
```

---

## Verification Checklist (Build in This Order)

```
[ ] 1.  Sim launches, table + robot render at correct positions
[ ] 2.  Objects spawn on table surface, not floating or clipping
[ ] 3.  Overhead camera returns non-black 640x480 RGB frame
[ ] 4.  Wrist camera returns non-black frame + depth image, moves with robot
[ ] 5.  Scripted policy moves robot arm without self-collision
[ ] 6.  Gripper closes on object (object translates with gripper)
[ ] 7.  Object is carried to basket and released inside it
[ ] 8.  Termination fires after all objects are in basket
[ ] 9.  Reset randomizes object positions correctly, robot returns to home
[ ] 10. Intermediate HDF5 written with correct tensor shapes after 1 episode
[ ] 11. Run 10 episodes without crash
[ ] 12. Inspect saved frames in HDF5 — images are meaningful, not corrupted
[ ] 13. convert_to_lerobot.py produces valid LeRobot dataset structure
[ ] 14. Verify Parquet columns: observation.state (14,), action (7,), task_index present
[ ] 15. Verify MP4 videos decode correctly for both cameras
```

Stop at any failed checkpoint. Do not proceed to the next until current passes.

---

## What NOT to Build Yet

- Do not build PICK / SCAN / PLACE phase separation
- Do not build replay.py or visualize.py
- Do not build teleop driver or policy driver
- Do not build reward functions
- Do not add ROS2 integration
- Do not add domain randomization for lighting or textures yet (only object position randomization)

---

## Notes

- Isaac Lab version: use latest stable with compatible Isaac Sim stable version (5.1)
- All tensor operations stay on GPU — do not .cpu() in the hot loop, only at HDF5 flush time
- num_envs=1 for initial debugging, scale to 64 after checklist passes
- Use ENV_REGEX_NS pattern for all prim paths to support multi-env correctly
- Wrist camera prim_path must be a child of the robot's hand link for automatic pose tracking
- FRANKA_PANDA_CFG import: `from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG`
- Camera observations use `mdp.image` (already in isaaclab core) — do not reimplement
- gym task ID: `"Isaac-Packing-Franka-v0"` — register in `packing_collector/__init__.py`
- `setup.py` and `config/extension.toml` are required for the package to be recognized by Isaac Lab
- LeRobot fine-tuning requires a language task description in `meta/tasks.jsonl` — pi-0 will not condition correctly without it
- pi-0 resizes all camera images to 224×224 internally — storing at 480×640 in MP4 is fine