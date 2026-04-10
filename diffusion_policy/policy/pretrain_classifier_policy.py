from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np
import copy
import random

from diffusion_policy.scorer.ca_scorer import ScorerNetworkTransformer, ScorerNetworkMLP
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class PretrainClassfierPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            use_classifier: bool,
            smooth_label: bool,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            scorer_network,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            infer_loss_type='diffusion',
            temperature_threshold = 0.2,
            loss_threshold=None,
            softplus_margin=0.0,
            add_random_gripper=True,
            additional_data_num=16,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # self.scorer = ScorerNetworkTransformer(scorer_network)
        self.scorer = ScorerNetworkMLP(scorer_network)
        self.obs_encoder = obs_encoder

        if smooth_label:
            self.label_key = 'smooth_label'
        else:
            self.label_key = 'label'
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.path_type = "linear"
        self.add_random_gripper = add_random_gripper
        self.additional_data_num = additional_data_num
        self.kwargs = kwargs

        # temperature is fixed! TODO: adjustable temperature
        self.softplus_margin = softplus_margin
        if softplus_margin:
            self.softbound_fn = nn.Softplus(beta=1.0, threshold=20.0)
        else:
            self.softbound_fn = nn.ReLU()
        self.temperature_threshold = temperature_threshold
        if loss_threshold:
            self.loss_threshold = loss_threshold
        else:
            self.loss_threshold = 100

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.loss_fn_list = {
            'classifier': self.classifier_loss,
            'contrastive': self.classifier_loss,
            'flow_matching': self.classifier_loss,
            'diffusion': self.classifier_loss
        }

        self.sample_function_list = {
            'classifier': self.classifier_infer,
            'contrastive': self.classifier_infer,
            'flow_matching': self.classifier_infer,
            'diffusion': self.classifier_infer            
        }

        self.sample_function = self.sample_function_list[infer_loss_type]
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss(beta=1.0)

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def classifier_infer_direct(self, obs_feature, actions):
        nactions = self.normalizer['action'].normalize(actions)
        score_input = {'qkv': torch.cat([obs_feature, nactions], dim=-1)}
        temperature_vari = self.scorer(score_input).mean(dim=-1) # for now don't separate the different joints # for now don't separate the different joints
        return temperature_vari

    def classifier_infer(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # there is only one task
        failure_label = batch['obs'][self.label_key]
        del batch['obs'][self.label_key]
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        action_num = nactions.shape[2]
        # label = torch.stack([torch.zeros(batch_size,).to(failure_label.device), failure_label], dim=1)
        
        # label = torch.zeros((batch_size, 2)).to(nactions.device)
        # flat_idx = torch.randperm(batch_size)[:int(2*batch_size/3)] 
        # label[flat_idx,1] = 1 

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # self attention structure
        # # all_fail_label = (label[:,0]+label[:,1]).unsqueeze(1).unsqueeze(2).expand(-1, self.score_step, 1) # consider all failure as one type
        # # obs_step = horizon - self.score_step + 1
        # # cur_obs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1,*x.shape[2:]))
        # score_feature = global_cond.unsqueeze(1).expand(-1, 16, -1)
        # score_input = {'qkv': torch.cat([score_feature, nactions], dim=-1)}
        # temperature_vari = self.scorer(score_input).mean(dim=-1) # for now don't separate the different joints # for now don't separate the different joints
        # batch['obs'][self.label_key] = failure_label
        
        # # mlp structure one score for the final
        # score_input = {'image': global_cond, 'action': nactions}
        # score = self.scorer(score_input).mean(dim=-1) # for now don't separate the different joints
        # batch['obs'][self.label_key] = failure_label

        # mlp structure, two score: first and last
        score_input = {'image': global_cond, 'action': nactions}
        temperature_vari = self.scorer(score_input) # for now don't separate the different joints
        first_score = temperature_vari[:, :16].mean(dim=-1)
        second_score = temperature_vari[:, 16:].mean(dim=-1)
        score = torch.stack([first_score, second_score])
        batch['obs'][self.label_key] = failure_label
        return score

    def classifier_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # there is only one task
        if self.add_random_gripper:
            batch = self.add_randomized_gripper(batch)
        if self.label_key == 'label':
            score_label = batch['obs'][self.label_key][:, 0]
            del batch['obs'][self.label_key]
            label = torch.stack([torch.zeros(batch_size,).to(score_label.device), score_label], dim=1)
            all_fail_label = (label[:,0]+label[:,1]).unsqueeze(1).float() # consider all failure as one type
        elif self.label_key == 'smooth_label':
            all_fail_label = batch['obs'][self.label_key].float()
            del batch['obs'][self.label_key]   
        else:
            raise NotImplementedError()  
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        action_num = nactions.shape[2]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        
        # # cross attention structure
        # obs_step = horizon - self.score_step + 1
        # cur_obs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1,*x.shape[2:]))
        # score_feature = self.obs_encoder(cur_obs).reshape(batch_size, self.n_obs_steps, -1)
        # score_input = {'q': nactions, 'kv': score_feature}
        # temperature_vari = self.scorer.forward_cls(score_input).mean(dim=-1).unsqueeze(-1) # for now don't separate the different joints
        # loss = self.loss_fn(temperature_vari, all_fail_label)

        # self attention structure
        # # cur_obs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps].reshape(-1,*x.shape[2:]))
        # # score_feature = self.obs_encoder(cur_obs).reshape(batch_size, -1).unsqueeze(1).expand(-1, 16, -1)
        # score_feature = global_cond.unsqueeze(1).expand(-1, 16, -1)
        # score_input = {'qkv': torch.cat([score_feature, nactions], dim=-1)}
        # temperature_vari = self.scorer(score_input).mean(dim=-1) # for now don't separate the different joints
        # loss = self.loss_fn(temperature_vari, all_fail_label)  

        # # mlp structure
        # score_input = {'image': global_cond, 'action': nactions}
        # temperature_vari = self.scorer(score_input).mean(dim=-1) # for now don't separate the different joints
        # loss = self.loss_fn(temperature_vari, all_fail_label)  

        # mlp structure, two score: first and last
        score_input = {'image': global_cond, 'action': nactions}
        temperature_vari = self.scorer(score_input) # for now don't separate the different joints
        first_score = temperature_vari[:, :16].mean(dim=-1)
        second_score = temperature_vari[:, 16:].mean(dim=-1)
        score = torch.stack([first_score, second_score])
        loss = self.loss_fn(score, all_fail_label)

        # # mlp structure, one delta score
        # score_input = {'image': global_cond, 'action': nactions}
        # temperature_vari = self.scorer(score_input).mean(dim=-1) # for now don't separate the different joints
        # loss = self.loss_fn(temperature_vari, all_fail_label)

        return loss.mean(), {}

    def add_randomized_gripper(self, batch):
        bsz, horizon = batch['obs']['left_shoulder_rgb'].shape[0], batch['obs']['left_shoulder_rgb'].shape[1]
        addtion_data_num = min(self.additional_data_num, bsz)
        idx = np.random.choice(bsz, size=addtion_data_num, replace=False)
        augmented_action = copy.deepcopy(batch['action'][idx])
        # augmented_obs = copy.deepcopy(batch['obs'][key][idx])
        augmented_label = copy.deepcopy(torch.stack([batch['obs'][self.label_key][idx][:,0], 
                                        batch['obs'][self.label_key][idx][:,-1]], dim=0)) # only take the last score
        
        p_true = random.uniform(0.6, 1.0)
        mask = (np.random.rand(len(idx), horizon) < p_true)
        mask[:,0] = False
        flag = np.ones_like(mask, dtype=np.uint8)
        flag[mask & (augmented_action[:,:,-1].detach().cpu().numpy() == 0)] = 0
        rows_has_zero = (flag == 0).any(axis=1)
        augmented_action[:,:,-1][mask] = 1 - augmented_action[:,:,-1][mask]
        augmented_label[1,rows_has_zero] = 0

        batch['action'] = torch.cat((batch['action'], augmented_action))
        old_label = torch.stack([batch['obs'][self.label_key][:,0], 
                                        batch['obs'][self.label_key][:,-1]], dim=0)
        batch['obs'][self.label_key] = torch.cat([old_label, augmented_label], dim=-1).float()
        for key in batch['obs'].keys():
            if key!=self.label_key:
                batch['obs'][key] = torch.cat((batch['obs'][key], batch['obs'][key][idx]))

        return batch

if __name__ == "__main__":
    main()