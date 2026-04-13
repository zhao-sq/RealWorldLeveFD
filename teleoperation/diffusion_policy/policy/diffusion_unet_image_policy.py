from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np

# from diffusion_policy.scorer.ca_scorer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
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
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        # for p in self.obs_encoder.parameters():
        #     p.requires_grad = False
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
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
        self.kwargs = kwargs

        # temperature is fixed! TODO: adjustable temperature
        self.temperature = 0.02

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.loss_fn_list = {
            'contrastive': self.compute_cfm_loss,
            'flow_matching': self.compute_fm_loss,
            'diffusion': self.compute_df_loss
        }

        self.sample_function_list = {
            'contrastive': self.flow_matching_sample,
            'flow_matching': self.flow_matching_sample,
            'diffusion': self.conditional_sample            
        }

        self.sample_function = self.sample_function_list[infer_loss_type]

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def flow_matching_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        # if num_inference_steps is None:
        num_inference_steps = self.num_inference_steps

        bsz = condition_data.shape[0]
        device = condition_data.device
        dtype = condition_data.dtype
        trajectory = torch.randn_like(condition_data)
        # trajectory = torch.zeros_like(condition_data)
        trajectory[condition_mask] = condition_data[condition_mask]

        eps = 1e-4
        t_steps = torch.linspace(1.0 - eps, eps, num_inference_steps, device=device, dtype=dtype)

        for i in range(num_inference_steps - 1):
            t = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t
            trajectory[condition_mask] = condition_data[condition_mask]
            t_batch = torch.full((bsz,), t, device=device, dtype=dtype)
            v = self.model(
                trajectory,
                t_batch,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            trajectory = trajectory + dt * v
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.sample_function(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_df_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        if self.label_key in list(batch['obs'].keys()):
            del batch['obs'][self.label_key]
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

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

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss, {}

    def compute_fm_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        if self.label_key in list(batch['obs'].keys()):
            del batch['obs'][self.label_key]
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

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

        condition_mask = self.mask_generator(trajectory.shape)

        # TODO from here
        # Sample noise that we'll add to the traj
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # noise = torch.randn_like(trajectory)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each traj
        # timesteps = torch.rand(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (bsz,), device=trajectory.device
        # ).long()
        timesteps = torch.rand(bsz, device=trajectory.device, dtype=trajectory.dtype)
        timesteps = timesteps.clamp_(1e-4, 1.0-1e-4)
        # Add noise to the clean traj according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory, model_target = self.flow_matching_add_noise(
            trajectory, noise, timesteps)
        # model_target = d_alpha_t * trajectory + d_sigma_t * noise
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        model_output = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        # cfm loss prediction
        loss = F.mse_loss(model_output, model_target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss, {}

    def compute_cfm_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # there is only one task
        failure_label = batch['obs'][self.label_key][:, 0]
        del batch['obs'][self.label_key]
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        label = torch.stack([torch.zeros(batch_size,).to(failure_label.device), failure_label], dim=1)
        
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

        condition_mask = self.mask_generator(trajectory.shape)

        # TODO from here
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # noise = torch.randn_like(trajectory)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        # timesteps = torch.rand(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (bsz,), device=trajectory.device
        # ).long()
        timesteps = torch.rand(bsz, device=trajectory.device, dtype=trajectory.dtype)
        timesteps = timesteps.clamp_(1e-4, 1.0-1e-4)
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory, model_target = self.flow_matching_add_noise(
            trajectory, noise, timesteps)
        # model_target = d_alpha_t * trajectory + d_sigma_t * noise
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory = noisy_trajectory.clone()
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        model_output = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        # cfm loss prediction
        loss = self.cfm_loss_fn(model_output, model_target, label)
        
        return loss
    
    def flow_matching_add_noise(self, trajectory, noise, timesteps):
        bsz = trajectory.shape[0]
        if self.path_type == "linear":
            alpha_t = 1 - timesteps
            sigma_t = timesteps
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(timesteps * np.pi / 2)
            sigma_t = torch.sin(timesteps * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(timesteps * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(timesteps * np.pi / 2)
        else:
            raise NotImplementedError()
        model_input = alpha_t.view(bsz, 1, 1) * trajectory + sigma_t.view(bsz, 1, 1) * noise
        model_target = d_alpha_t * trajectory + d_sigma_t * noise
        return model_input, model_target
    
    def cfm_loss_fn_soft_bound(self, model_output, model_target, label):
        neg_label = torch.where(label[:,1]==1)[0]
        pos_label = torch.where(label[:,1]==0)[0]
        pos_output = model_output[pos_label]
        neg_output = model_output[neg_label]
        pos_distances = torch.cdist(pos_output, pos_output, p=2)
        pos_neg_distances = torch.cdist(pos_output, neg_output, p=2)
        loss_for_pos = F.mse_loss(model_output, model_target, reduction='none')[pos_label]

        # if pos_distances.mean() >= pos_neg_distances.mean():
        #     loss = loss_for_pos + temp_loss_for_neg # use the output of model, might not be converage
        # else:
        #     loss = loss_for_pos
        loss = loss_for_pos + self.temperature*F.relu(pos_distances-pos_neg_distances)

        return loss.mean(), {}

    def cfm_loss_fn(self, model_output, model_target, label):
        # label has two dimension
        # 0: task definition, 1: pos/neg defination  

        # Version 1: we assume there is only one task, so only second dimension is used in the calculation
        pos_label = torch.where(label[:,1]==1)[0]
        neg_label = torch.where(label[:,1]==0)[0]
        chosen_neg_label = neg_label[torch.randint(0, len(neg_label), (len(pos_label),))]
        loss_for_all_sample = F.mse_loss(model_output, model_target, reduction='none')
        # if loss_for_all_sample.dim() > 1:
        #     loss_for_all_sample = loss_for_all_sample.mean(dim=1)

        loss = loss_for_all_sample[pos_label] - self.temperature * loss_for_all_sample[chosen_neg_label]

        # TODO for different tasks, use the first dimension

        return loss.mean(), {}
