# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].


import os
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from moviepy import ImageSequenceClip
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import const
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Waypoint
from rlbench.demo import Demo
from rlbench.environment import DIR_PATH, Environment

from failgen.utils import (
    ObservationConfigExt,
    name_to_class,
    save_demo,
)

CURRENT_DIR = "/home/shuqi/AHA/aha/Data_Generation/rlbench-failgen/failgen" # os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(CURRENT_DIR, "configs")
RLBENCH_TASKPY_FOLDER = os.path.join(DIR_PATH, "tasks")

MAX_FAILURE_ATTEMPTS = 5

"""
    Replay the dataset on specific environment
"""

def reset_save():
    step_counter: int = 0
    cache_video: List[np.ndarray] = []
    cache_cameras = dict(
            front=[],
            overhead=[],
            left_shoulder=[],
            right_shoulder=[],
            # wrist=[],
        )
    return step_counter, cache_video, cache_cameras

def run_replay_dataset(
    task_name: str,
    save_data: bool = True,
    headless: bool = True,
    task_folder: str = RLBENCH_TASKPY_FOLDER,
    # save_path: str,
) -> None:
    # load initial config path
    config_filepath = os.path.join(CONFIGS_DIR, task_name)
    config = OmegaConf.load(f"{config_filepath}.yaml")
    obs_config = ObservationConfigExt(config.data)
    if not save_data:
        obs_config.set_all_high_dim(False)

    env: Environment = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=headless,
        # static_positions=True
    ) # MYC: first build the action mode of arm and gripper
    env.launch()

    task_class = name_to_class(task_name, task_folder)
    if task_class is None:
        raise RuntimeError(f"Couldn't instantiate task '{task_name}'")
    task_env = env.get_task(task_class) # MYC: then initialize the task envrionment
    
    # initial saving setup
    step_counter,cache_video,cache_cameras = reset_save()

    # load dataset for replay
    import pickle
    testdata_pth = '/mnt/ssd1/szhao/LeveFD/pick_and_lift/grasp_wp1/episodes/episode0/low_dim_obs.pkl'
    # testdata_pth = '/mnt/ssd1/szhao/LeveFD/rlbench_dataset/pick_and_lift/success/episodes/episode2/low_dim_obs.pkl'
    with open(testdata_pth, 'rb') as f:
        data = pickle.load(f)
    print('')
    traj_len = len(data)
    obs_sequence = []
    for i in range(traj_len):
        if data[i].gripper_open != 1.0:
            print(data[i].gripper_open)
        obs_sequence.append(data[i].get_low_dim_data())
    obs_sequence = np.stack(obs_sequence, axis=0) # joint velocity (0,7), gripper (28,30)
    
    for k in range(2):
        # # replay the dataset
        ignore_collision = 0
        task_env.reset()
        obs_list = []
        for i in range(obs_sequence.shape[0]):
            obs, rew, done = task_env.step(np.hstack((obs_sequence[i,1:8], obs_sequence[i,0], ignore_collision)))
            cache_cameras["front"].append(obs.front_rgb)
            cache_cameras["overhead"].append(obs.overhead_rgb)
            cache_cameras["left_shoulder"].append(obs.left_shoulder_rgb)
            cache_cameras["right_shoulder"].append(obs.right_shoulder_rgb)
            obs_list.append(obs)
            
        collected_demo = Demo(obs_list)
        
        episode_path = '/mnt/ssd1/szhao/LeveFD/replay'+str(k+1)
        # save camera images
        for cam_name in cache_cameras.keys():
            clip = ImageSequenceClip(cache_cameras[cam_name], fps=30)
            clip.write_videofile(
                os.path.join(episode_path, f"vid_{cam_name}.mp4"), logger=None,
            )

        save_demo(config.data, collected_demo, episode_path)
        print('Demo saved in:', episode_path)
    
    # for wp_idx in potential_waypoints:
    #     target_fail_obj.change_waypoint_fail_name(f"waypoint{wp_idx}")
    #     print(f"Triying to collect from waypoint {wp_idx}")
    #     for i in range(num_episodes):
    #         env_wrapper.reset() 
    #         attempts = max_tries
    #         tmp_test_action = np.array([ 1.00000000e+00,  4.76837158e-06, -2.20298767e-03, -4.29153442e-05,
    #                                     2.62260437e-04,  3.05175781e-04, -1.47819519e-04])
    #         env_wrapper._task_env.step(tmp_test_action)
    #         while attempts > 0:
    #             demo, success = env_wrapper.get_failure()
    #             if demo is not None and not success:
    #                 env_wrapper.save_keyframe_data(i, fail_type, wp_idx)
    #                 break
    #             else:
    #                 attempts -= 1
    #         if attempts <= 0:
    #             print(
    #                 f"Got an issue with task: {task_name}, failure: {fail_type}"
    #             )
    #         else:
    #             print(f"Saved episode {i+1} / {num_episodes}")
    #     print(
    #         f"Saved {num_episodes} for task {task_name}, failure: {fail_type}, "
    #         + f"waypoint-index: {wp_idx}"
    #     )

    # env_wrapper.shutdown()
"""
Entry point for the failure data collection script.

This function parses command-line arguments to determine the parameters for data
collection, including task name, number of episodes, maximum tries per episode,
whether to use multiprocessing, failure type, and the save path. Based on these
arguments, it then initiates the data collection process either in a sequential
or multiprocessing mode for each failure type specified.

Returns:
    int: An exit status code (0 indicates successful execution).
"""

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="basketball_in_hoop",
        help="The name of the task to load for this example",
    )
    # parser.add_argument(
    #     "--episodes",
    #     type=int,
    #     default=1,
    #     help="The number of episodes to collect",
    # )
    # parser.add_argument(
    #     "--max_tries",
    #     type=int,
    #     default=10,
    #     help="The maximum number of tries to test a single failure",
    # )
    # parser.add_argument(
    #     "--multiprocessing",
    #     action="store_true",
    #     help="Whether or not to use multiprocessing for data collection",
    # )
    # parser.add_argument(
    #     "--failtype",
    #     type=str,
    #     default="",
    #     help="The fail type to use for data collection of single failure"
    # )
    # parser.add_argument(
    #     "--savepath",
    #     type=str,
    #     default="",
    #     help="The path to the folder where to save all the data",
    # )

    args = parser.parse_args()

    run_replay_dataset(
        task_name=args.task,
        # fail_type=args.failtype,
        # num_episodes=args.episodes,
        # max_tries=args.max_tries,
        # save_path=args.savepath,
    )

    # global FAILURES_LIST
    # if args.failtype != "":
    #     FAILURES_LIST = [args.failtype]

    # if args.multiprocessing:
    #     processes = [
    #         Process(
    #             target=run_get_failures,
    #             args=(
    #                 args.task,
    #                 fail_type,
    #                 args.episodes,
    #                 args.max_tries,
    #                 args.savepath,
    #             ),
    #         )
    #         for fail_type in FAILURES_LIST
    #     ]
    #     [t.start() for t in processes]
    #     [t.join() for t in processes]
    # else:
    #     for fail_type in FAILURES_LIST:
    #         run_get_failures(
    #             task_name=args.task,
    #             fail_type=fail_type,
    #             num_episodes=args.episodes,
    #             max_tries=args.max_tries,
    #             save_path=args.savepath,
    #         )

    # return 0


if __name__ == "__main__":
    raise SystemExit(main())
