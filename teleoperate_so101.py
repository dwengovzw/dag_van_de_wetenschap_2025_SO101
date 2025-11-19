from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

camera_config = {
    #"wrist": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=480, fps=30)
}

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="ingenieursproject2_follower",
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="ingenieursproject2_leader",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)