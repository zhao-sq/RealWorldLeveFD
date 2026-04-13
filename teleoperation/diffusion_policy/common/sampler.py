from typing import List, Iterator, Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer

import math
import random
import torch
from torch.utils.data import Sampler


# @numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    assert episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)
    seg_count = 0

    indices = list()
    episode_index = list()
    for i in range(len(episode_ends)):
        # print(episode_ends[i])
        tmp_episode_index = []
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
            tmp_episode_index.append(seg_count)
            seg_count = seg_count + 1
        episode_index.append(tmp_episode_index)
    indices = np.array(indices)
    return indices, episode_index

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        include_contrastive=False
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        self.include_contrastive = include_contrastive
        if keys is None:
            keys = list(replay_buffer.keys())

        meta_keys = list(replay_buffer.meta.keys())
        if self.include_contrastive:
            assert 'divide_index' in meta_keys, "No failure data detected"
            divide_index = replay_buffer.meta['divide_index']

        episode_ends = replay_buffer.episode_ends[:]
        if self.include_contrastive:
            assert isinstance(divide_index, int), "dividing index must be int type"
            assert divide_index in episode_ends and divide_index < episode_ends[-1] and divide_index > 0, "divide_index is out of range"
            i = np.where(episode_ends == divide_index)[0][0]
            success_episode_ends = episode_ends[:i+1]

        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
            
        if include_contrastive:
            success_episode_mask = episode_mask[:i+1]

        if np.any(episode_mask):
            indices, episode_index = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
            if self.include_contrastive:
                success_indices, success_episode_index = create_indices(success_episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=success_episode_mask
                )
                self.success_indices_length = len(success_indices)
        else:
            indices = np.zeros((0,4), dtype=np.int64)
            episode_index = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.episode_index = episode_index
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)

    def get_divide_length(self):
        assert self.include_contrastive, "Not in contrastive mode"
        return self.success_indices_length

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

# class CustomSequenceSampler(SequenceSampler):
#     def __init__(self, replay_buffer, sequence_length, pad_before = 0, pad_after = 0, keys=None, key_first_k=dict(), episode_mask = None):
#         super().__init__(replay_buffer, sequence_length, pad_before, pad_after, keys, key_first_k, episode_mask)

class ProportionalSampler(Sampler[List[int]]):
    def __init__(
        self,
        total_length,
        divide_index,
        per,
        batch_size: int,
        drop_last: bool = True,
        with_replacement_success: bool = False,   # A 少时可设 True
        with_replacement_failure: bool = True,    # B 更少，常设 True
        generator: Optional[torch.Generator] = None,
    ):
        assert batch_size > 0
        self.bs = batch_size
        per = per
        self.success_per = math.ceil(batch_size * per)
        self.failure_per = self.bs - self.success_per
        self.drop_last = drop_last
        self.with_rep_success = with_replacement_success
        self.with_rep_failure = with_replacement_failure
        self.g = generator

        self.divide_index = divide_index
        self.total_length = total_length

        # 1) 标注每个 sequence 索引属于 A 还是 B
        # success_pool, failure_pool = [], []
        # for i in range(n_seq):
        #     epi = self.S.index_to_episode_id(i)    # 用上面补充的方法
        #     if episode_mask_A[epi]:
        #         success_pool.append(i)
        #     else:
        #         failure_pool.append(i)
        self.success_pool = np.arange(self.divide_index)
        self.failure_pool = np.arange(self.divide_index, self.total_length)

        # 2) 预估 epoch 能产出的 batch 数（无放回时）
        self.nsuccess = len(self.success_pool)
        self.nfailure = len(self.failure_pool)
        max_batches_success = (self.nsuccess // self.success_per)
        max_batches_failure = (self.nfailure // self.failure_per)
        # self.max_batches_no_rep = int(min(max_batches_success, max_batches_failure))
        self.epoch_batches = max_batches_success

        # 3) 建索引乱序
        self._shuffle()

    def _rng_choice(self, pool: List[int], k: int, with_rep: bool) -> List[int]:
        if k <= 0: return []
        if with_rep:
            if self.g is not None:
                idx = torch.randint(0, len(pool), (k,), generator=self.g).tolist()
            else:
                idx = [random.randrange(len(pool)) for _ in range(k)]
            return [pool[i] for i in idx]
        else:
            k = min(k, len(pool))
            if self.g is not None:
                perm = torch.randperm(len(pool), generator=self.g).tolist()
            else:
                perm = list(range(len(pool)))
                random.shuffle(perm)
            return [pool[i] for i in perm[:k]]

    def _shuffle(self):
        if self.g is not None:
            self.a_shuf = torch.tensor(self.success_pool)[torch.randperm(len(self.success_pool), generator=self.g)].tolist()
            self.b_shuf = torch.tensor(self.failure_pool)[torch.randperm(len(self.failure_pool), generator=self.g)].tolist()
        else:
            self.a_shuf = self.success_pool[:]; random.shuffle(self.a_shuf)
            self.b_shuf = self.failure_pool[:]; random.shuffle(self.b_shuf)
        self.ia = 0
        self.ib = 0

    def __len__(self) -> int:
        return self.epoch_batches if self.drop_last else self.epoch_batches

    def __iter__(self) -> Iterator[List[int]]:
        self._shuffle()
        for _ in range(len(self)):
            batch = []

            # ---- 取 A ----
            need = self.success_per
            remain = len(self.a_shuf) - self.ia
            take = min(need, remain)
            if take > 0:
                batch.extend(self.a_shuf[self.ia:self.ia+take])
                self.ia += take
                need -= take
            if need > 0:
                if not self.with_rep_success:
                    return
                # 放回补齐
                batch.extend(self._rng_choice(self.success_pool, need, with_rep=True))

            # ---- 取 B ----
            need = self.failure_per
            remain = len(self.b_shuf) - self.ib
            take = min(need, remain)
            if take > 0:
                batch.extend(self.b_shuf[self.ib:self.ib+take])
                self.ib += take
                need -= take
            if need > 0:
                if not self.with_rep_failure:
                    return
                batch.extend(self._rng_choice(self.failure_pool, need, with_rep=True))

            # 防御：保证 batch 大小
            if len(batch) == self.bs:
                yield batch
            else:
                if self.drop_last:
                    continue
                else:
                    yield batch        

class SamplerForTestClassifier(Sampler[List[int]]):
    def __init__(self, episode_index, rollout_traj_index):
        index_list = []
        for idx in rollout_traj_index:
            index_list.append(episode_index[idx])
        self.index_list = [x for row in index_list for x in row]

    def __iter__(self):
        return iter(self.index_list)

    def __len__(self):
        return len(self.index_list)