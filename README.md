# SO-101 Multi-Task Data Collection & Imitation Learning

A complete system for collecting robot demonstration data, training imitation learning policies, and executing learned behaviors on the SO-101 robot arm. Built on top of [HuggingFace LeRobot](https://github.com/huggingface/lerobot).

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
  - [Task Configuration (tasks_config.json)](#task-configuration-tasks_configjson)
  - [Hardware Configuration (in code)](#hardware-configuration-in-code)
- [Using the GUI](#using-the-gui)
  - [Task Overview](#task-overview)
  - [Recording Episodes](#recording-episodes)
  - [Failed Episodes](#failed-episodes)
  - [Training](#training)
  - [Policy Execution](#policy-execution)
- [Data Storage](#data-storage)
  - [Directory Layout](#directory-layout)
  - [Dataset Format](#dataset-format)
  - [Task State](#task-state)
- [Training](#training-overview)
  - [Local Training](#local-training)
  - [Remote Training](#remote-training)
- [How Imitation Learning Works](#how-imitation-learning-works)
  - [Action Chunking Transformer (ACT)](#action-chunking-transformer-act)
  - [From Demonstrations to Policy](#from-demonstrations-to-policy)
  - [Training Pipeline](#training-pipeline)
- [File Reference](#file-reference)

---

## Overview

This application provides an interactive booth experience where visitors can teach a robot arm new tasks by demonstrating them. The system:

1. **Collects demonstration data** — a human operator physically guides the robot through a task using a leader arm (teleoperation)
2. **Stores the data** — joint positions, camera images, and sensor readings are recorded as structured datasets
3. **Trains a policy** — an imitation learning model learns to reproduce the demonstrated behavior
4. **Executes the policy** — the robot performs the task autonomously using the trained model

Multiple tasks can be configured and managed through a graphical interface.

---

## How It Works

```
┌──────────────┐     teleoperation     ┌──────────────┐
│  Leader Arm  │ ───────────────────▸  │ Follower Arm │
│  (human)     │                       │  (robot)     │
└──────────────┘                       └──────┬───────┘
                                              │
                           records joint positions + camera frames
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  LeRobot Dataset  │
                                    │  (parquet + mp4)  │
                                    └────────┬─────────┘
                                             │
                              lerobot-train (ACT policy)
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  Trained Policy   │
                                    │  (neural network) │
                                    └────────┬─────────┘
                                             │
                              autonomous execution
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │ Robot performs the │
                                    │ task on its own    │
                                    └──────────────────┘
```

---

## Prerequisites

- **Python 3.10+** with conda
- **LeRobot** installed (`pip install lerobot` or from source)
- **SO-101 robot arm** (follower) + SO-101 leader arm connected via USB
- **USB camera** (e.g. webcam at `/dev/video4`)
- Optional: Arduino with touch sensor on `/dev/ttyACM0`

```bash
conda activate lerobot
```

---

## Running the Application

```bash
# Default (uses tasks_config.json in the same directory)
python record_dataset_multi_task.py

# With a custom config file
python record_dataset_multi_task.py --config my_tasks.json
```

The GUI launches fullscreen. The recording thread runs in the background. Close the window or click "Exit" to shut down cleanly (all open datasets are finalized on exit).

---

## Configuration

### Task Configuration (`tasks_config.json`)

This JSON file defines the tasks and training parameters:

```json
{
    "policy_type": "act",
    "training_steps": 100000,
    "batch_size": 8,
    "training_mode": "local",
    "remote_server_url": "http://gpu-server:5000",
    "remote_api_key": "your-secret-key",
    "tasks": [
        {
            "name": "Pick up the cup",
            "description": "Grab the cup from the table and lift it up",
            "required_episodes": 50,
            "max_episode_time_s": 60
        }
    ]
}
```

#### Global Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_type` | string | `"act"` | Policy architecture to train. `"act"` = Action Chunking Transformer. |
| `training_steps` | int | `100000` | Number of gradient steps for training. |
| `batch_size` | int | `8` | Batch size during training. Reduce if running out of GPU memory. |
| `training_mode` | string | `"local"` | `"local"` to train on the same machine, `"remote"` to offload training to a GPU server. |
| `remote_server_url` | string | `""` | URL of the remote training server (only used when `training_mode` is `"remote"`). |
| `remote_api_key` | string | `""` | Shared secret for authenticating with the remote training server. |

#### Per-Task Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Display name of the task. Also used as the dataset folder name (lowercased, spaces → underscores). |
| `description` | string | No | Shown in the GUI to explain the task to the operator. |
| `required_episodes` | int | Yes | Number of successful demonstrations needed. Training starts automatically when this count is reached. |
| `max_episode_time_s` | int | No (default: 60) | Maximum recording time per episode in seconds. The episode is automatically discarded if time runs out. |

### Hardware Configuration (in code)

These settings are defined at the top of `record_dataset_multi_task.py` and must be edited directly:

| Variable | Default | Description |
|----------|---------|-------------|
| `FPS` | `30` | Recording frame rate (frames per second). |
| `DATA_DIR` | `"~/datasets"` | Base directory for all datasets, policies, and state files. |
| `camera_config` | `/dev/video4`, 640×480 | Camera device path, resolution, and FPS. |
| `robot_config` | Port `/dev/ttyACM0` | Follower arm serial port and identifier. |
| `teleop_config` | Port `/dev/ttyACM1` | Leader arm serial port and identifier. |

To add multiple cameras, add entries to `camera_config`:

```python
camera_config = {
    "scene_top": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30),
    "wrist":     OpenCVCameraConfig(index_or_path="/dev/video6", width=640, height=480, fps=30),
}
```

---

## Using the GUI

### Task Overview

The main screen shows all configured tasks as cards with:

- **Progress bar** — episodes collected vs. required
- **Training status badge** — Not trained / Training... / Policy ready / Training failed
- **📋 Collect Data** — start recording demonstrations for this task
- **🧠 Train** — manually trigger training (available even before `required_episodes` is reached)
- **📄 Log** — view the training log in real time
- **🤖 Run Policy** — execute the trained policy (appears after training completes)

The overview auto-refreshes every 2 seconds to reflect training progress.

### Recording Episodes

1. Select a task and click **Collect Data**
2. Read the task description and click **▶ Start Recording**
3. Use the leader arm to guide the robot through the task
4. Click **⏹ Stop Recording** when done (or wait for the countdown timer)
5. Review: click **✓ Yes, save it!** if the demonstration was good

### Failed Episodes

If the demonstration was not successful, click **✗ No, let me retry**. You'll be shown a screen asking **"What went wrong?"** with these options:

- Robot did not grab the object
- Robot dropped the object
- Robot moved to the wrong position
- Robot collided with something
- I made a mistake operating the leader arm
- The task took too long
- Other / not sure

The failed episode data is **not discarded** — it is saved to a separate `_bad` dataset (e.g. `pick_up_the_cup_bad/`) for research purposes. The failure reason is stored as the task label in the dataset, so episodes can be filtered by reason later.

### Training

Training starts automatically when the required number of episodes is collected. You can also trigger it manually with the **🧠 Train** button on any task card.

Use the **📄 Log** button to open a live log viewer that shows the `lerobot-train` output, refreshing every 3 seconds.

### Policy Execution

Once training completes, a **🤖 Run Policy** button appears. Click it to have the robot perform the task autonomously. Press **⏹ Stop Policy** to interrupt.

---

## Data Storage

### Directory Layout

All data is stored under `DATA_DIR` (default: `~/datasets/`):

```
~/datasets/
├── task_state.json                      # Progress for all tasks
│
├── pick_up_the_cup/                     # Good episodes dataset
│   ├── data/
│   │   └── chunk-000/
│   │       └── file-000.parquet         # All frame data (actions, states, etc.)
│   ├── videos/
│   │   └── observation.images.scene_top/
│   │       └── chunk-000/
│   │           └── file-000.mp4         # Camera recordings
│   ├── meta/
│   │   ├── info.json                    # Dataset metadata
│   │   ├── stats.json                   # Global statistics
│   │   ├── tasks.parquet                # Task name lookup table
│   │   └── episodes/
│   │       └── chunk-000/
│   │           └── file-000.parquet     # Per-episode metadata & stats
│   └── train.log                        # Training log (if trained)
│
├── pick_up_the_cup_bad/                 # Failed episodes dataset (same structure)
│   └── ...                              # Task labels include "| <failure reason>"
│
└── pick_up_the_cup_policy/              # Training output
    └── checkpoints/
        ├── 010000/                      # Checkpoint at step 10000
        ├── ...
        └── 100000/
            └── pretrained_model/        # Final model weights
```

### Dataset Format

LeRobot uses a **chunked parquet format** (v3.0):

- **Frame data** (`data/chunk-NNN/file-NNN.parquet`) — one row per frame, columns include:
  - `action` — joint positions sent to the robot (6 floats)
  - `observation.state` — current joint positions read from the robot
  - `observation.images.scene_top` — reference to the video frame
  - `episode_index` — which episode this frame belongs to
  - `frame_index`, `timestamp`, `task_index`

- **Episode metadata** (`meta/episodes/...`) — one row per episode with statistics (min, max, mean, std, quantiles for each feature)

- **Chunk size** is 1000 episodes per file — so with fewer than 1000 episodes, everything fits in a single `chunk-000/file-000.parquet`.

### Task State

`task_state.json` tracks progress across sessions:

```json
{
  "Pick up the cup": {
    "episodes_collected": 12,
    "training_status": "trained",
    "policy_path": "/home/user/datasets/pick_up_the_cup_policy/checkpoints/100000/pretrained_model"
  }
}
```

Possible `training_status` values: `"not_started"`, `"training"`, `"trained"`, `"failed"`.

This file is updated automatically. If you restart the application, it picks up where it left off — including resuming remote training polls.

---

## Training Overview

### Local Training

When `training_mode` is `"local"`, the application runs `lerobot-train` as a subprocess on the same machine. This works but can be slow without a GPU.

### Remote Training

When `training_mode` is `"remote"`, the application:
1. Tars and uploads the dataset to the remote server
2. The server runs `lerobot-train` on a GPU
3. The local app polls every 30 seconds until training completes
4. Downloads the trained policy and extracts it locally

See [REMOTE_TRAINING.md](REMOTE_TRAINING.md) for server setup instructions.

---

## How Imitation Learning Works

### The Goal

The robot needs to learn a **policy** — a function that maps what the robot sees and feels (observations) to what it should do next (actions):

$$\pi(a_t \mid o_t) : \text{observations} \rightarrow \text{actions}$$

Where:
- $o_t$ = current joint positions + camera image at time step $t$
- $a_t$ = target joint positions to move to

### Learning from Demonstrations

Instead of hand-coding rules or using reinforcement learning (which requires millions of trials), **imitation learning** trains the policy directly from human demonstrations:

1. A human performs the task while the robot mirrors the movements (teleoperation)
2. Every frame, the system records the observation $o_t$ and the action $a_t$ the human chose
3. A neural network is trained to predict $a_t$ given $o_t$, minimizing the difference between predicted and demonstrated actions

This is essentially **supervised learning** on trajectory data:

$$\mathcal{L} = \sum_{t} \| \pi_\theta(o_t) - a_t^{\text{demo}} \|^2$$

### Action Chunking Transformer (ACT)

The default policy architecture is **ACT** ([Zhao et al., 2023](https://tonyzhaozh.github.io/aloha/)). It improves on naive behavior cloning in several key ways:

**1. Action Chunking**
Instead of predicting one action at a time, ACT predicts a **chunk** of future actions (e.g. the next 100 timesteps). This helps overcome compounding errors — small prediction mistakes that accumulate when each action depends on the previous prediction.

**2. Transformer Architecture**
ACT uses a transformer encoder-decoder:
- The **encoder** processes the current observation (joint positions + camera images via a CNN backbone)
- The **decoder** generates a sequence of future actions using cross-attention over the encoded observations

**3. CVAE Training**
ACT is trained as a Conditional Variational Autoencoder (CVAE). During training, an additional encoder sees the *future* action sequence and produces a latent style variable $z$. This captures the multimodality in demonstrations — different humans might perform the same task differently. At inference time, $z$ is sampled from the prior.

**4. Temporal Ensembling**
At execution time, the robot doesn't just use the first action from each chunk. It queries the policy at every timestep, getting overlapping action chunks, and averages them using exponential weighting. This smooths out the robot's motion and reduces jitter.

### Training Pipeline

When `lerobot-train` runs, it:

1. **Loads the dataset** — reads the parquet files and video frames
2. **Creates data loaders** — samples random batches of (observation, action_chunk) pairs
3. **Initializes the ACT model** — transformer encoder-decoder with a ResNet vision backbone
4. **Runs gradient descent** — for the configured number of steps:
   - Sample a batch of demonstrations
   - Forward pass: encode observations → decode action chunks
   - Compute loss: MSE between predicted and demonstrated actions + KL divergence on the latent variable
   - Backward pass: update network weights
5. **Saves checkpoints** — model weights saved periodically (e.g. every 10,000 steps)

The final checkpoint is what gets loaded for policy execution.

### Why More Episodes Help

- **Coverage**: More demonstrations cover more variations of the task (different object positions, speeds, approaches)
- **Robustness**: The policy sees a wider distribution of states, making it less likely to fail in unfamiliar situations
- **Averaging**: Random noise in individual demonstrations gets averaged out

Typically 50–100 high-quality demonstrations are needed for reliable task execution, though simple tasks may work with fewer.

---

## File Reference

| File | Description |
|------|-------------|
| `record_dataset_multi_task.py` | Main application — data collection, training orchestration, policy execution |
| `faire_multi_task_gui.py` | Tkinter GUI — task overview, recording flow, failure reasons, training log viewer |
| `tasks_config.json` | Task definitions and training parameters |
| `sensing_so101_with_button.py` | SO-101 subclass adding touch sensor and button observations |
| `remote_training_server.py` | Flask server for offloading training to a GPU machine |
| `requirements-server.txt` | Python dependencies for the remote training server |
| `REMOTE_TRAINING.md` | Deployment guide for the remote training server |
| `faire_gui.py` | Original single-task GUI (standalone, not used by multi-task system) |
| `record_dataset_faire.py` | Original single-task recording script (standalone) |
| `teleoperate_so101.py` | Basic teleoperation script (no data recording) |
| `custom_keyboard_listener.py` | Keyboard event listener utility |
