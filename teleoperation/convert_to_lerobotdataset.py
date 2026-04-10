import numpy as np
import cv2
import random
from pathlib import Path
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Step 1: Define dataset features
features = {
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_state"]
    },
    "action_is_pad": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["action_is_pad"]
    },
    # "observation.is_gripper_in_action": {
    #     "dtype": "float32",
    #     "shape": (1,),
    #     "names": ["is_gripper_in_action"]
    # },
    "observation.image": {
        "dtype": "image",
        "shape": (128, 128, 3),
        "names": ["height", "width", "channels"]
    },
    # "observation.environment_state": {
    #     "dtype": "float32",
    #     "shape": (256,),
    #     "names": [f"depth_{i}" for i in range(256)]
    # },
    "observation.state":{
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_state", "is_gripper_in_action"]
    }
}

# Step 2: Helper function to safely wrap arrays
def safe_array(x, dtype, shape):
    arr = np.asarray(x).astype(dtype).reshape(shape)
    return arr

# Step 3: Create LeRobotDataset (single merged repo)
ds = LeRobotDataset.create(
    repo_id="crx/apr_25_pick_and_place_no_split",
    fps=7,
    root="/media/db4/dwu/lerobot/crx_dataset/lerobot_format/pick_and_place_dataset_no_split",
    features=features,
    use_videos=False,
)

# Step 4: Define your old dataset
old_data_path = Path("/media/db4/dwu/lerobot/crx_dataset/pick_and_place_dataset")

# Step 5: Gather all episodes
all_episode_folders = sorted([
    ep for subfolder in old_data_path.iterdir() if subfolder.is_dir()
    for ep in subfolder.iterdir() if ep.is_dir()
])

# Step 6: Shuffle episodes
random.seed(42)
random.shuffle(all_episode_folders)

split_idx = int(1 * len(all_episode_folders))
train_episodes = all_episode_folders[:split_idx]
val_episodes = all_episode_folders[split_idx:]

print(f"Total episodes: {len(all_episode_folders)}, Train: {len(train_episodes)}, Val: {len(val_episodes)}")

# Step 7: Function to process and assign task split
def process_episodes(episodes, split_name):
    for episode_folder in episodes:
        frame_files = sorted(episode_folder.glob("*.npz"))
        if len(frame_files) == 0:
            continue

        print(f"Processing {split_name} episode: {episode_folder}, {len(frame_files)} frames")

        for frame_file in frame_files:
            data = np.load(frame_file)

            # Process color image
            img = data["fixed_color"]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)

            # Process depth image
            depth = data["fixed_depth"]
            depth_resized = cv2.resize(depth, (128, 128), interpolation=cv2.INTER_NEAREST)
            depth_downsampled = cv2.resize(depth_resized, (16, 16), interpolation=cv2.INTER_NEAREST).flatten()
            is_gripper_in_action = np.array([1 if data["is_gripper_in_action"] else 0])

            # Construct frame
            frame = {
                "action": safe_array(np.concatenate([data["current_state"], np.array([data["gripper_state"]])]), np.float32, (7,)),
                "action_is_pad": safe_array([0], np.float32, (1,)),
                #"observation.is_gripper_in_action": safe_array([1 if data["is_gripper_in_action"] else 0], np.float32, (1,)),
                "observation.image": img,
                #"observation.environment_state": safe_array(depth_downsampled, np.float32, (256,)),
                "observation.state": safe_array(np.concatenate([data["current_state"], np.array([data["gripper_state"]]), is_gripper_in_action]), np.float32, (8,)),
                "task": "pick and place" if split_name == "train" else "pick and place (val)"
            }

            ds.add_frame(frame)

        ds.save_episode()

# Step 8: Actually process
process_episodes(train_episodes, split_name="train")
process_episodes(val_episodes, split_name="val")

# Step 9: Finalize splits inside metadata
# You need to overwrite info.json to say which episodes are train vs val
info_path = ds.root / "meta" / "info.json"
import json

with open(info_path, "r") as f:
    info = json.load(f)

info["splits"] = {
    "train": f"0:{split_idx}",
    "val": f"{split_idx}:{len(all_episode_folders)}",
}

with open(info_path, "w") as f:
    json.dump(info, f, indent=2)

print("Conversion complete with train/val split!")
