"""
Faire Booth Data Collection Script for SO-101 Robot

This script is designed for autonomous data collection at a faire/exhibition.
It uses a GUI to guide participants through the recording process instead of
keyboard controls.

Usage:
    python record_dataset_faire.py
"""

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

from sensing_so101_with_button import SO101FollowerWithTouch
from faire_gui import init_faire_gui

import threading
import time
import datetime
from pathlib import Path


# ===== Configuration =====

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Make me a coffee"
DATA_DIR = "~/datasets/test"  # Set to a path (e.g. "/home/tom/datasets") to store data locally, or None for default (~/.cache/huggingface/lerobot/)

# ---- Camera config ----
camera_config = {
    "scene_top": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30),
}

# ---- Robot (follower) config ----
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="toms_follower_arm",
    cameras=camera_config,
)

# ---- Teleoperator (leader) config ----
teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="toms_leader_arm",
)

# ===== End Configuration =====


def recording_thread(robot, teleop_device, dataset, events, gui):
    """
    Runs the recording loop in a background thread so the GUI stays responsive.
    """
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        # Wait for GUI to signal start
        while not events["start_episode"] and not events["stop_recording"]:
            time.sleep(0.1)

        if events["stop_recording"]:
            break

        events["start_episode"] = False
        events["exit_early"] = False
        events["rerecord_episode"] = False
        events["episode_accepted"] = False

        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop_device,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Wait for user to accept or reject via GUI
        while (
            not events["episode_accepted"]
            and not events["rerecord_episode"]
            and not events["stop_recording"]
        ):
            time.sleep(0.1)

        if events["stop_recording"]:
            break

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            gui.notify_episode_discarded()
            continue

        # Episode accepted - save it
        dataset.save_episode()
        log_say(f"Episode {episode_idx + 1} saved")
        gui.notify_episode_saved()
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    robot.disconnect()
    teleop_device.disconnect()

    if episode_idx > 0:
        log_say("Pushing dataset to hub...")
        dataset.push_to_hub()
        log_say("Dataset pushed successfully")


def main():
    # Initialize robot and teleoperator
    robot = SO101FollowerWithTouch(robot_config, sensor_serial_port="/dev/ttyACM0")
    teleop_device = SO101Leader(teleop_config)

    # Configure dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Create dataset
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H.%M.%S")
    root = Path(DATA_DIR).expanduser() / timestamp if DATA_DIR else None

    dataset = LeRobotDataset.create(
        repo_id=f"tomneutens/up-grab-test-session_{timestamp}",
        fps=FPS,
        root=root,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize visualization
    init_rerun(session_name="recording")

    # Connect hardware
    robot.connect()
    teleop_device.connect()

    # Initialize GUI
    gui, events = init_faire_gui(
        task_description=TASK_DESCRIPTION,
        num_episodes=NUM_EPISODES,
        max_episode_time_s=EPISODE_TIME_SEC,
    )

    # Start recording in background thread
    rec_thread = threading.Thread(
        target=recording_thread,
        args=(robot, teleop_device, dataset, events, gui),
        daemon=True,
    )
    rec_thread.start()

    # Run GUI on main thread (tkinter requirement)
    gui.run()

    # If GUI closed, ensure recording stops
    events["stop_recording"] = True
    events["exit_early"] = True
    rec_thread.join(timeout=10)


if __name__ == "__main__":
    main()
