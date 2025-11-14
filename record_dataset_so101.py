from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
#from lerobot.utils.control_utils import init_keyboard_listener
from custom_keyboard_listener import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

from sensing_so101_with_button import SO101FollowerWithTouch

import time
import datetime

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 600
TASK_DESCRIPTION = "Make me a coffee"

# Create the robot and teleoperator configurations
# ---- Add the correct video path in the config ----
camera_config = {
    "wrist": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30)
}

# ---- Add the correct follower port in the config ----
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM6",
    id="toms_follower_arm", 
    cameras=camera_config
)

# ---- Add the correct leader port in the config ----
teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM0", 
    id="toms_leader_arm",
)

# Initialize the robot and teleoperator
# ---- Add the correct microcontroller port in the config ----
robot = SO101FollowerWithTouch(robot_config, sensor_serial_port="/dev/ttyACM1")
teleop_device = SO101Leader(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# 
# Create default processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="tomneutens/up-grab-test-session_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S'),
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4
)

# Initialize the keyboard listener and    visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop_device.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
    print(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")
    
    print("Waiting, press space to start episode.")
    while not events["start_episode"]:
        time.sleep(0.2)
        
    events["start_episode"] = False
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
  

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        print("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
print("Stop recording")
robot.disconnect()
teleop_device.disconnect()
dataset.push_to_hub()