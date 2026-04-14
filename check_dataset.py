from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="tomneutens/up-grab-test-session_2026-04-14_12.02.36",
    root="/home/tom/datasets/test/2026-04-14_12.02.36"
)

# Check basic info
print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {dataset.num_frames}")
print(f"Features: {list(dataset.features.keys())}")

# Check a single frame - verify sensor data is present
frame = dataset[0]
print(f"\nFrame keys: {list(frame.keys())}")
print(f"State shape: {frame['observation.state'].shape}")
print(f"State values: {frame['observation.state']}")