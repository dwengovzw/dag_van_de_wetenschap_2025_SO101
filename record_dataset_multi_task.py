"""
Multi-Task Faire Booth Script for SO-101 Robot

Supports:
    - Multiple tasks defined in tasks_config.json
    - Per-task data collection with GUI
    - Background policy training when enough episodes are collected
    - Policy execution mode for trained tasks

Usage:
    python record_dataset_multi_task.py
    python record_dataset_multi_task.py --config my_tasks.json
"""

import json
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import argparse
from pathlib import Path

import requests

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from sensing_so101_with_button import SO101FollowerWithTouch
from faire_multi_task_gui import init_faire_multi_task_gui


# ===== Hardware Configuration =====

PYTHON_BIN = sys.executable  # Use the same Python that launched this script
TRAIN_BIN = str(Path(PYTHON_BIN).parent / "lerobot-train")
FPS = 30
DATA_DIR = "~/datasets"

camera_config = {
    "scene_top": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30),
}

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="toms_follower_arm",
    cameras=camera_config,
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="toms_leader_arm",
)

# ===== End Hardware Configuration =====

# Remote training configuration (set from config file in main())
_training_config = {
    "training_mode": "local",
    "remote_server_url": "",
    "remote_api_key": "",
}


def load_task_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def get_task_data_root(task_name: str) -> Path:
    """Get the dataset root directory for a given task."""
    safe_name = task_name.lower().replace(" ", "_").replace("/", "_")
    return Path(DATA_DIR).expanduser() / safe_name


def get_task_output_dir(task_name: str) -> Path:
    """Get the training output directory for a given task."""
    safe_name = task_name.lower().replace(" ", "_").replace("/", "_")
    return Path(DATA_DIR).expanduser() / f"{safe_name}_policy"


def load_task_state(state_path: Path) -> dict:
    """Load persisted task state from disk, or return empty dict."""
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def save_task_state(state_path: Path, task_state: dict):
    """Persist task state to disk."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(task_state, f, indent=2)


# Cache of open datasets, keyed by task name
_dataset_cache: dict[str, LeRobotDataset] = {}


def get_or_create_dataset(task_name: str, dataset_features: dict, robot_name: str) -> LeRobotDataset:
    """Get an existing dataset for a task or create a new one. Cached per task."""
    if task_name in _dataset_cache:
        return _dataset_cache[task_name]

    root = get_task_data_root(task_name)
    safe_name = task_name.lower().replace(" ", "_").replace("/", "_")
    repo_id = f"local/{safe_name}"

    # Check if a previous session left a finalized dataset (with episodes parquet)
    episodes_dir = root / "meta" / "episodes"
    has_saved_episodes = episodes_dir.exists() and any(episodes_dir.glob("*.parquet"))

    if has_saved_episodes:
        # Reopen the existing dataset for appending
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
    else:
        # Either no dataset exists, or a previous session created the dir
        # but never saved any episodes. Clean up the incomplete dir and start fresh.
        if root.exists():
            shutil.rmtree(root)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=FPS,
            root=root,
            features=dataset_features,
            robot_type=robot_name,
            use_videos=True,
            image_writer_threads=4,
        )

    _dataset_cache[task_name] = dataset
    return dataset


def recording_thread(robot, teleop_device, events, gui, task_configs, task_state, dataset_features, state_path):
    """
    Main worker thread: waits for GUI signals to collect episodes or run policies.
    """
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    while not events["stop_recording"]:
        # Wait for either a data collection start or a policy execution request
        while (
            not events["start_episode"]
            and not events["run_policy_task"]
            and not events["stop_recording"]
        ):
            time.sleep(0.1)

        if events["stop_recording"]:
            break

        # ---- Policy execution ----
        if events["run_policy_task"]:
            task_name = events["run_policy_task"]
            events["run_policy_task"] = None
            _run_policy(
                task_name, robot, events, gui,
                task_state, robot_action_processor, robot_observation_processor,
                teleop_action_processor
            )
            continue

        # ---- Data collection episode ----
        task_name = events.get("current_task_name")
        if not task_name:
            events["start_episode"] = False
            continue

        events["start_episode"] = False
        events["exit_early"] = False
        events["rerecord_episode"] = False
        events["episode_accepted"] = False

        # Find the task config
        task_cfg = next((t for t in task_configs if t["name"] == task_name), None)
        if not task_cfg:
            continue

        dataset = get_or_create_dataset(task_name, dataset_features, robot.name)
        ep_num = task_state.get(task_name, {}).get("episodes_collected", 0) + 1
        log_say(f"Recording episode {ep_num} for: {task_name}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop_device,
            dataset=dataset,
            control_time_s=task_cfg.get("max_episode_time_s", 60),
            single_task=task_name,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Wait for accept/reject from GUI
        while (
            not events["episode_accepted"]
            and not events["rerecord_episode"]
            and not events["stop_recording"]
        ):
            time.sleep(0.1)

        if events["stop_recording"]:
            break

        if events["rerecord_episode"]:
            log_say("Discarding episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            gui.notify_episode_discarded()
            continue

        # Save the episode (data is appended to the open parquet writer).
        # Don't finalize here — closing the writer and re-opening for the next
        # episode would overwrite the parquet file.
        dataset.save_episode()
        log_say(f"Episode saved for: {task_name}")
        gui.notify_episode_saved()

        # Persist state
        save_task_state(state_path, task_state)

        # Check if we should start training
        ts = task_state.get(task_name, {})
        if (
            ts.get("episodes_collected", 0) >= task_cfg["required_episodes"]
            and ts.get("training_status", "not_started") == "not_started"
        ):
            # Finalize the dataset so parquet metadata is flushed to disk
            # before the training subprocess tries to read it.
            if task_name in _dataset_cache:
                _dataset_cache[task_name].finalize()
                _dataset_cache.pop(task_name)
            _start_background_training(task_name, task_cfg, task_state, state_path)

    # Finalize all open datasets on exit
    for ds in _dataset_cache.values():
        ds.finalize()
    _dataset_cache.clear()

    # Cleanup
    log_say("Stopping...")
    robot.disconnect()
    teleop_device.disconnect()


def _run_policy(task_name, robot, events, gui, task_state, robot_action_processor, robot_observation_processor, teleop_action_processor):
    """Load and execute a trained policy for a task."""
    ts = task_state.get(task_name, {})
    policy_path = ts.get("policy_path")
    if not policy_path or not Path(policy_path).exists():
        log_say(f"No trained policy found for: {task_name}")
        gui.notify_policy_finished()
        return

    log_say(f"Loading policy for: {task_name}")

    try:
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cls = get_policy_class(policy_cfg.type)
        policy = policy_cls.from_pretrained(policy_path)
        policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
        )
    except Exception as e:
        log_say(f"Failed to load policy: {e}")
        gui.notify_policy_finished()
        return

    events["exit_early"] = False
    events["stop_policy"] = False

    log_say(f"Running policy for: {task_name}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=None,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=None,  # Don't record during policy execution
        control_time_s=60,
        single_task=task_name,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    log_say("Policy execution finished")
    events["stop_policy"] = False
    gui.notify_policy_finished()


def _start_background_training(task_name: str, task_cfg: dict, task_state: dict, state_path: Path):
    """Launch policy training locally or remotely based on config."""
    ts = task_state.setdefault(task_name, {})
    ts["training_status"] = "training"
    save_task_state(state_path, task_state)

    if _training_config["training_mode"] == "remote":
        target = _remote_training_worker
    else:
        target = _training_worker

    thread = threading.Thread(
        target=target,
        args=(task_name, task_cfg, task_state, state_path),
        daemon=True,
    )
    thread.start()


def _training_worker(task_name: str, task_cfg: dict, task_state: dict, state_path: Path):
    """Run lerobot-train in a subprocess."""
    data_root = get_task_data_root(task_name)
    output_dir = Path(get_task_output_dir(task_name))
    safe_name = task_name.lower().replace(" ", "_").replace("/", "_")
    repo_id = f"local/{safe_name}"

    # Read global training parameters from the task config's parent
    policy_type = task_cfg.get("policy_type", "act")
    training_steps = task_cfg.get("training_steps", 100000)
    batch_size = task_cfg.get("batch_size", 8)

    cmd = [
        TRAIN_BIN,
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={data_root}",
        f"--policy.type={policy_type}",
        f"--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--steps={training_steps}",
        f"--batch_size={batch_size}",
    ]

    # Write training log next to the dataset, NOT inside output_dir
    # (lerobot-train fails if output_dir already exists)
    log_file = data_root / "train.log"

    log_say(f"Starting training for: {task_name}")
    log_say(f"Training log: {log_file}")
    try:
        with open(log_file, "w") as lf:
            lf.write(f"Command: {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)
        with open(log_file, "a") as lf:
            lf.write(f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}\n")
        if result.returncode == 0:
            # Find the last checkpoint
            checkpoint_dir = _find_latest_checkpoint(output_dir)
            if checkpoint_dir:
                ts = task_state.setdefault(task_name, {})
                ts["training_status"] = "trained"
                ts["policy_path"] = str(checkpoint_dir)
                save_task_state(state_path, task_state)
                log_say(f"Training complete for: {task_name}")
            else:
                task_state.setdefault(task_name, {})["training_status"] = "failed"
                save_task_state(state_path, task_state)
                log_say(f"Training finished but no checkpoint found for: {task_name}")
        else:
            task_state.setdefault(task_name, {})["training_status"] = "failed"
            save_task_state(state_path, task_state)
            log_say(f"Training failed for: {task_name}\n{result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        task_state.setdefault(task_name, {})["training_status"] = "failed"
        save_task_state(state_path, task_state)
        log_say(f"Training timed out for: {task_name}")
    except Exception as e:
        task_state.setdefault(task_name, {})["training_status"] = "failed"
        save_task_state(state_path, task_state)
        log_say(f"Training error for {task_name}: {e}")


def _safe_tar_extract(tar: tarfile.TarFile, dest: Path):
    """Extract tar safely, preventing path traversal attacks."""
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise ValueError(f"Path traversal detected in tar member: {member.name}")
    tar.extractall(dest)


def _remote_training_worker(task_name: str, task_cfg: dict, task_state: dict, state_path: Path):
    """Upload dataset to remote server, poll for completion, download policy."""
    server_url = _training_config["remote_server_url"].rstrip("/")
    api_key = _training_config["remote_api_key"]
    headers = {"X-API-Key": api_key}

    data_root = get_task_data_root(task_name)
    output_dir = Path(get_task_output_dir(task_name))
    ts = task_state.setdefault(task_name, {})

    job_id = ts.get("remote_job_id")

    # If no existing job, upload dataset and create one
    if not job_id:
        log_say(f"Uploading dataset for remote training: {task_name}")
        tar_path = data_root / "dataset_upload.tar.gz"
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(str(data_root), arcname="dataset")

            with open(tar_path, "rb") as f:
                resp = requests.post(
                    f"{server_url}/jobs",
                    files={"dataset": ("dataset.tar.gz", f, "application/gzip")},
                    data={
                        "task_name": task_name,
                        "policy_type": task_cfg.get("policy_type", "act"),
                        "training_steps": str(task_cfg.get("training_steps", 100000)),
                        "batch_size": str(task_cfg.get("batch_size", 8)),
                    },
                    headers=headers,
                    timeout=600,
                )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            ts["remote_job_id"] = job_id
            save_task_state(state_path, task_state)
            log_say(f"Remote training job submitted: {job_id}")
        except Exception as e:
            log_say(f"Failed to submit remote training for {task_name}: {e}")
            ts["training_status"] = "failed"
            save_task_state(state_path, task_state)
            return
        finally:
            tar_path.unlink(missing_ok=True)

    # Poll for completion
    log_say(f"Polling remote training status for: {task_name} (job {job_id})")
    while True:
        time.sleep(30)
        try:
            resp = requests.get(f"{server_url}/jobs/{job_id}", headers=headers, timeout=30)
            resp.raise_for_status()
            status = resp.json()["status"]
        except Exception as e:
            log_say(f"Remote status check failed (will retry): {e}")
            continue

        if status == "completed":
            break
        elif status == "failed":
            log_say(f"Remote training failed for: {task_name}")
            ts["training_status"] = "failed"
            ts.pop("remote_job_id", None)
            save_task_state(state_path, task_state)
            return

    # Download trained policy
    log_say(f"Downloading trained policy for: {task_name}")
    policy_tar_path = data_root / "policy_download.tar.gz"
    try:
        resp = requests.get(
            f"{server_url}/jobs/{job_id}/policy",
            headers=headers,
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()
        with open(policy_tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        output_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(policy_tar_path, "r:gz") as tar:
            _safe_tar_extract(tar, output_dir)

        checkpoint = _find_latest_checkpoint(output_dir)
        if checkpoint:
            ts["training_status"] = "trained"
            ts["policy_path"] = str(checkpoint)
            log_say(f"Remote training complete for: {task_name}")
        else:
            ts["training_status"] = "failed"
            log_say(f"Policy downloaded but no checkpoint found for: {task_name}")
    except Exception as e:
        log_say(f"Failed to download policy for {task_name}: {e}")
        ts["training_status"] = "failed"
    finally:
        ts.pop("remote_job_id", None)
        save_task_state(state_path, task_state)
        policy_tar_path.unlink(missing_ok=True)


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Find the latest checkpoint directory under an output dir."""
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    # Checkpoints are numbered directories like 010000, 020000, ...
    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if checkpoint_dirs:
        # Look for pretrained_model subdirectory or use the checkpoint dir itself
        pretrained = checkpoint_dirs[0] / "pretrained_model"
        return pretrained if pretrained.exists() else checkpoint_dirs[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Multi-task data collection and policy training")
    parser.add_argument("--config", default="tasks_config.json", help="Path to tasks config JSON file")
    args = parser.parse_args()

    # Load task config
    config = load_task_config(args.config)
    task_configs = config["tasks"]

    # Propagate global training params into each task config for convenience
    for task in task_configs:
        task.setdefault("policy_type", config.get("policy_type", "act"))
        task.setdefault("training_steps", config.get("training_steps", 100000))
        task.setdefault("batch_size", config.get("batch_size", 8))

    # Load or initialize task state
    state_path = Path(DATA_DIR).expanduser() / "task_state.json"
    task_state = load_task_state(state_path)

    # Initialize missing task state entries
    for task in task_configs:
        if task["name"] not in task_state:
            task_state[task["name"]] = {
                "episodes_collected": 0,
                "training_status": "not_started",
                "policy_path": None,
            }

    # Check if any tasks already have trained policies on disk
    for task in task_configs:
        name = task["name"]
        ts = task_state[name]
        if ts["training_status"] == "trained" and ts.get("policy_path"):
            if not Path(ts["policy_path"]).exists():
                ts["training_status"] = "not_started"
                ts["policy_path"] = None
        # Also check if a policy was trained outside this script
        if ts["training_status"] == "not_started":
            output_dir = get_task_output_dir(name)
            checkpoint = _find_latest_checkpoint(output_dir)
            if checkpoint:
                ts["training_status"] = "trained"
                ts["policy_path"] = str(checkpoint)

    save_task_state(state_path, task_state)

    # Set up training mode config
    _training_config["training_mode"] = config.get("training_mode", "local")
    _training_config["remote_server_url"] = config.get("remote_server_url", "")
    _training_config["remote_api_key"] = config.get("remote_api_key", "")

    # Resume polling for any in-progress remote training jobs
    if _training_config["training_mode"] == "remote":
        for task in task_configs:
            name = task["name"]
            ts = task_state[name]
            if ts.get("training_status") == "training" and ts.get("remote_job_id"):
                log_say(f"Resuming remote training poll for: {name}")
                thread = threading.Thread(
                    target=_remote_training_worker,
                    args=(name, task, task_state, state_path),
                    daemon=True,
                )
                thread.start()

    # Initialize robot and teleoperator
    robot = SO101FollowerWithTouch(robot_config, sensor_serial_port="/dev/ttyACM0")
    teleop_device = SO101Leader(teleop_config)

    # Build dataset features (same for all tasks)
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Initialize visualization
    init_rerun(session_name="multi_task_recording")

    # Connect hardware
    robot.connect()
    teleop_device.connect()

    # Initialize GUI
    gui, events = init_faire_multi_task_gui(task_configs, task_state)

    # Start worker thread
    worker = threading.Thread(
        target=recording_thread,
        args=(robot, teleop_device, events, gui, task_configs, task_state, dataset_features, state_path),
        daemon=True,
    )
    worker.start()

    # Run GUI on main thread
    gui.run()

    # Cleanup
    events["stop_recording"] = True
    events["exit_early"] = True
    worker.join(timeout=10)

    save_task_state(state_path, task_state)
    log_say("Done")


if __name__ == "__main__":
    main()
