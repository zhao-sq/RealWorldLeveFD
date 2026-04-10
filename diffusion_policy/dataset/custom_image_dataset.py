from typing import Dict
import torch
import numpy as np
import copy
import sys
import traceback
import pickle
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_imagenet_normalizer

import math
import random
from typing import List, Iterator, Optional
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler
# import numpy.core as _np_core
# sys.modules.setdefault("numpy._core", _np_core)

def inplace_compact_first_axis(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    assert arr.shape[0] == mask.shape[0], \
        f"mask len {mask.shape[0]} != arr.shape[0] {arr.shape[0]}"
    n = arr.shape[0]
    write = 0
    for read in range(n):
        if mask[read]:
            if write != read:
                arr[write] = arr[read]
            write += 1
    return arr[:write]

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

class CustomImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            loss_type='mean',
            action_type='gripper',
            delta_action=False,
            smooth_label=False,
            from_real=False
            ):
        
        super().__init__()
        try:
            with open(zarr_path, "rb") as f:
                root_data_original = NumpyCompatUnpickler(f).load()
                # with open(zarr_path, "rb") as f:
                #     root_data_original = pickle.load(f)
        except Exception as e:
            print("!! ERROR in pickle.load:", repr(e))
            traceback.print_exc()
            sys.exit(1)
        self.delta_action = delta_action
        if smooth_label:
            self.label_key = 'smooth_label'
        else:
            self.label_key = 'label'
    
        # real data image keys
        self.img_key = ['in_hand', 'fixed_left']
        # simulation data image keys
        # self.img_key = ['front_rgb']
        # self.img_key = ['overhead_rgb','front_rgb']
        # self.img_key = ['left_shoulder_rgb', 'front_rgb']
        # self.img_key = ['overhead_rgb','left_shoulder_rgb']

        if loss_type == 'contrastive':
            include_contrastive = True
        else:
            include_contrastive = False
        self.from_real = from_real
        if from_real:
            root_data = self.define_action_type_real(root_data_original, delta_action=delta_action)
        else:
            root_data = self.define_action_type(root_data_original, 
                                                action_mode=action_type, 
                                                delta_action=delta_action,
                                                include_contrastive=include_contrastive)
        self.episode_ends = root_data['meta']['episode_ends']
        for ky in self.img_key:
            assert ky in root_data['meta']['applied_keys']

        self.replay_buffer = ReplayBuffer(root_data)
        if loss_type == 'contrastive':
            divide_index = root_data['meta']['divide_index']
            i = np.where(self.replay_buffer.episode_ends == divide_index)[0][0]
            val_n_episodes = i+1
        else:
            val_n_episodes = self.replay_buffer.n_episodes
        val_mask = get_val_mask(
            n_episodes=val_n_episodes, 
            val_ratio=val_ratio,
            seed=seed)

        # TODO change the train mask
        print(np.where(val_mask)[0])
        val_mask[:] = False

        if include_contrastive:
            val_mask_all = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
            val_mask_all[np.where(val_mask)[0]] = True
            val_mask = val_mask_all
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        # train_mask[1:] = False
        # train_mask[0] = True
        self.train_mask = train_mask
        self.val_mask = val_mask

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            include_contrastive=include_contrastive)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_episode_ends(self):
        return self.episode_ends

    def get_episode_index(self):
        return self.sampler.episode_index

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'low_dim_obs': np.hstack((self.replay_buffer['low_dim_obs'][...,8:15], self.replay_buffer['low_dim_obs'][...,:1]))
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in self.img_key:
            # normalizer[key] = get_image_range_normalizer()
            normalizer[key] = get_imagenet_normalizer()
        return normalizer

    def define_action_type_real(self, root, delta_action=False):
        if delta_action:
            action_mode = 'gripper_pose_euler_delta'
            index = root['meta']['low_dim_indice'][action_mode]
            delta_action = root['data']['low_dim_obs'][:,index[0]:index[1]]
            assert delta_action.shape[0] == root['data'][self.img_key[0]].shape[0]
            root['data']['action'] = delta_action
            return root
        else:
            action_mode = 'gripper_pose_euler'
            raise NotImplementedError
        

    def define_action_type(self, root, action_mode='gripper', delta_action=False, include_contrastive=False):
        if action_mode == 'gripper':
            index = [root['meta']['low_dim_indice']['gripper_pose_euler'], root['meta']['low_dim_indice']['gripper_open']]
        elif action_mode == 'joint_pos':
            index = [root['meta']['low_dim_indice']['joint_positions'], root['meta']['low_dim_indice']['gripper_open']]
        elif action_mode == 'joint_vel':
            index = [root['meta']['low_dim_indice']['joint_velocities'], root['meta']['low_dim_indice']['gripper_open']]
        else:
            return NotImplementedError
        # make actions
        # root['data']['action'] = np.concatenate([root['data']['low_dim_obs'][:, index[0][0]:index[0][1]], \
        #                                          root['data']['low_dim_obs'][:, index[1][0]:index[1][1]]], axis=1)
        episodes_ends = root['meta']['episode_ends']
        obs_mask = np.ones(episodes_ends[-1], dtype=int)
        obs_mask[episodes_ends-1] = int(0)
        episodes_starts = np.concatenate((np.array([0], dtype=int), episodes_ends[:-1]))
        action_mask = np.ones(episodes_ends[-1])
        action_mask[episodes_starts] = 0
        obs_key_list = list(root['data'].keys())
        offset = np.arange(1, len(episodes_ends) + 1)
        new_episode_end = episodes_ends - offset
        if include_contrastive:
            div_idx = np.where(root['meta']['divide_index']==episodes_ends)[0][0]
            root['meta']['divide_index'] = int(root['meta']['divide_index'] - div_idx - 1)
        if delta_action:
            all_actions = np.concatenate([root['data']['low_dim_obs'][:, index[0][0]:index[0][1]], \
                                                    root['data']['low_dim_obs'][:, index[1][0]:index[1][1]]], axis=1)
            # root['data']['action'] = all_actions[action_mask.astype(bool)]-all_actions[obs_mask.astype(bool)]
            delta_action = all_actions[action_mask.astype(bool)]-all_actions[obs_mask.astype(bool)]
            if action_mode == 'gripper':
                delta_rpy = delta_action[:,-4:-1]
                target_indice = np.where(delta_rpy>6.0)
                delta_rpy[target_indice] = (delta_rpy[target_indice] + np.pi) % (2*np.pi) - np.pi
                delta_action[:, -4:-1] = delta_rpy
            root['data']['action'] = delta_action
        else:
            root['data']['action'] = np.concatenate([root['data']['low_dim_obs'][:, index[0][0]:index[0][1]], \
                                                    root['data']['low_dim_obs'][:, index[1][0]:index[1][1]]], axis=1)[action_mask.astype(bool)]   
        for key in obs_key_list:
            # root['data'][key] = root['data'][key][obs_mask.astype(bool)]
            arr_new = inplace_compact_first_axis(root['data'][key], obs_mask.astype(bool))
            root['data'][key] = arr_new
            assert root['data'][key].shape[0] == new_episode_end[-1]     
        assert root['data']['action'].shape[0] == new_episode_end[-1]
        root['meta']['episode_ends'] = new_episode_end
        return root

    def __len__(self) -> int:
        return len(self.sampler)

    def get_divide_length(self):
        return self.sampler.get_divide_length()

    def _sample_to_data(self, sample):
        obs_dict = dict()
        agent_pos = np.hstack((sample['low_dim_obs'][:,8:15], sample['low_dim_obs'][:,:1])).astype(np.float32) # (agent_posx2, block_posex3)
        # image = np.moveaxis(sample[self.img_key],-1,1)/255
        obs_dict['low_dim_obs'] = agent_pos
        for ky in self.img_key:
            obs_dict[ky] = np.moveaxis(sample[ky],-1,1)/255
            # obs_dict[ky] = np.moveaxis(sample[ky],-1,1)[:, [2, 1, 0], :, :]/255 # Change! into rgb
        # image = np.moveaxis(sample[self.img_key][...,::-1],-1,1)/255
        assert np.unique(sample).size == 1, 'Different failure label in one sample'
        if self.label_key in list(sample.keys()):
            label = sample[self.label_key]
        else:
            label = np.zeros(agent_pos.shape[0],)
        obs_dict[self.label_key] = label

        data = {
            'obs': obs_dict,
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data#, idx

# def test():
#     import os
#     zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
#     dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)