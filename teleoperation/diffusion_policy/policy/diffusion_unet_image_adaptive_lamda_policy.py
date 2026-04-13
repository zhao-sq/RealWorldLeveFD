from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np

from diffusion_policy.scorer.ca_scorer import ScorerNetworkTransformer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImageAdpLamdaPolicy(BaseImagePolicy):
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

        # self.scorer = ScorerNetworkTransformer(scorer_network)
        # self.score_step = self.scorer.score_step
        self.use_classifier = use_classifier

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
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.path_type = "linear"
        self.kwargs = kwargs

        if smooth_label:
            self.label_key = 'smooth_label'
        else:
            self.label_key = 'label'

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
            'contrastive': self.compute_cfm_loss,
            'flow_matching': self.compute_fm_loss,
            'diffusion': self.compute_df_loss,
            'classifier': self.compute_clsfm_loss
        }

        self.sample_function_list = {
            'contrastive': self.flow_matching_sample,
            'flow_matching': self.flow_matching_sample,
            'diffusion': self.conditional_sample,
            'classifier': self.flow_matching_sample,         
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
        # action_num = nactions.shape[2]
        label = torch.stack([torch.zeros(batch_size,).to(failure_label.device), failure_label], dim=1)

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

        # temperature = self.temperature_adjustment_initial(nobs, nactions, label, horizon)
        temperature = self.temperature_adjustment_pn(nobs, label, batch_size, horizon)

        # cfm loss prediction
        # loss = self.cfm_loss_fn(model_output, model_target, label, temperature*self.temperature_threshold)
        # loss = self.cfm_loss_fn_same(model_output, model_target, label, temperature*self.temperature_threshold)
        loss, temp_corr, loss_pos_neg, loss_for_pos_mean = self.cfm_loss_fn_soft_bound(model_output, model_target, label, temperature*self.temperature_threshold)
        
        return loss, {'temperature': temp_corr, 'loss_pos_neg': loss_pos_neg, 'flow_loss': loss_for_pos_mean}

    def compute_clsfm_loss(self, batch):
        # TODO
        # normalize input
        if self.use_classifier:
            score = batch[1]
            batch = batch[0]
        assert 'valid_mask' not in batch
        # there is only one task
        if self.label_key in batch['obs'].keys():
            batch_label = batch['obs'][self.label_key]
            del batch['obs'][self.label_key]
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # action_num = nactions.shape[2]
        # label = torch.stack([torch.zeros(batch_size,).to(failure_label.device), failure_label], dim=1)

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

        loss = self.clsfm_loss(model_output, model_target, batch_label, score)
        
        return loss, {}

    def temperature_adjustment_initial(self, nobs, nactions, label, horizon):
        # previous calculate temperature: without correct data in the batch
        action_num = nactions.shape[2]
        all_fail_label = (label[:,0]+label[:,1])>=1 # consider all failure as one type
        obs_step = horizon - self.score_step + 1
        cur_obs = dict_apply(nobs, lambda x: x[all_fail_label, :obs_step].reshape(-1,*x.shape[2:]))
        score_feature = self.obs_encoder(cur_obs).reshape(all_fail_label.sum().item(), obs_step, -1)
        score_input = {'q': nactions[all_fail_label,-self.score_step:], 'kv': score_feature}
        temperature_vari = self.scorer(score_input).mean(dim=-1).unsqueeze(-1) # for now don't separate the different joints
        temperature = torch.zeros((all_fail_label.sum().item(), horizon, action_num)).to(self.device)# make the other not score loss as 0
        temperature[:, -self.score_step:, :] = temperature_vari
        return temperature

    def temperature_adjustment_pn(self, nobs, label, bsz, horizon):
        all_fail_label = (label[:,0]+label[:,1])>=1 # consider all failure as one type
        success_label = ~all_fail_label
        cur_obs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        score_feature = self.obs_encoder(cur_obs).reshape(bsz, horizon, -1)
        score_input = {'q': score_feature[all_fail_label], 'kv': score_feature[success_label]}
        temperature = self.scorer(score_input).mean(dim=-1).unsqueeze(-1)
        return temperature

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
    
    def cfm_loss_fn(self, model_output, model_target, label, temperature):
        # label has two dimension
        # 0: task definition, 1: pos/neg defination  

        # Version 1: we assume there is only one task, so only second dimension is used in the calculation
        neg_label = torch.where(label[:,1]==1)[0]
        pos_label = torch.where(label[:,1]==0)[0]
        chosen_neg_label = neg_label[torch.randint(0, len(neg_label), (len(pos_label),))]
        loss_for_all_sample = F.mse_loss(model_output, model_target, reduction='none')
        # if loss_for_all_sample.dim() > 1:
        #     loss_for_all_sample = loss_for_all_sample.mean(dim=1)
        loss_for_pos = loss_for_all_sample[pos_label]
        loss_for_neg = loss_for_all_sample[neg_label]
        temp_loss_for_neg = (temperature*loss_for_neg)[chosen_neg_label-len(pos_label)]

        if loss_for_pos.mean() <= self.loss_threshold:
            loss = loss_for_pos - temp_loss_for_neg # use the output of model, might not be converage
        else:
            loss = loss_for_pos

        # TODO for different tasks, use the first dimension

        return loss.mean(), {}

    def cfm_loss_fn_soft_bound(self, model_output, model_target, label, temperature):
        bzs = model_output.shape[0]
        model_output_flat = model_output.reshape(bzs,-1)
        neg_label = torch.where(label[:,1]==1)[0]
        pos_label = torch.where(label[:,1]==0)[0]
        pos_output = model_output_flat[pos_label]
        neg_gt = model_target.reshape(bzs,-1)[neg_label]
        pos_distances = torch.cdist(pos_output, pos_output, p=2)
        pos_neg_distances = torch.cdist(pos_output, neg_gt, p=2)
        loss_for_pos = F.mse_loss(model_output, model_target, reduction='none').reshape(bzs,-1)[pos_label]
        temperature_flat = temperature.reshape(temperature.shape[0], -1)
        temperature_corr = torch.cdist(temperature_flat, temperature_flat, p=2)

        temp_corr = temperature_corr
        if pos_distances.mean() >= pos_neg_distances.mean():
            loss = loss_for_pos - temp_corr * pos_neg_distances # use the output of model, might not be converage
        else:
            loss = loss_for_pos

        # temp_corr = temperature_corr
        # loss_pos_neg = self.softbound_fn(pos_distances+self.softplus_margin-pos_neg_distances)
        # loss_for_pos_mean = loss_for_pos.mean()
        # loss = loss_for_pos_mean + (temp_corr*loss_pos_neg).mean() # + (0.02-temp_corr).mean()

        return loss, temp_corr.mean().detach().cpu(), pos_distances.mean().detach().cpu(), pos_neg_distances.mean().detach().cpu()

    def clsfm_loss(self, model_output, model_target, label, adjusted_label):
        # pos_label = torch.where(adjusted_label>=self.temperature_threshold)
        # neg_label = torch.where(adjusted_label<self.temperature_threshold)
        # all_loss = F.mse_loss(model_output, model_target, reduction='none') # ((adjusted_label - adjusted_label).unsqueeze(-1).repeat(1, 1, 8))*
        # loss = all_loss[pos_label].mean() - 0.05*all_loss[neg_label].mean()
        
        bzs = model_output.shape[0]
        neg_label = torch.where(label[:,1]==1)[0]
        pos_label = torch.where(label[:,1]==0)[0]
        all_loss = F.mse_loss(model_output, model_target, reduction='none').reshape(bzs,-1)
        pos_output = all_loss[pos_label]
        neg_output = all_loss[neg_label]
        pos_distances = torch.cdist(pos_output, pos_output, p=2)
        pos_neg_distances = torch.cdist(pos_output, neg_output, p=2)

        # # label for absolute value
        # failure_adjusted_label = adjusted_label[neg_label]
        # pos_in_neg = torch.where(failure_adjusted_label>=self.temperature_threshold)
        # neg_in_neg = torch.where(failure_adjusted_label<self.temperature_threshold)
        # loss = pos_output.mean() + 0.05*neg_output[pos_in_neg].mean() - 0.01*all_loss[neg_in_neg].mean() # + 0.05*F.relu(pos_distances.mean()-pos_neg_distances.mean())

        # label for delta value
        delta_adjusted_label = adjusted_label[1]-adjusted_label[0]
        # delta_adjusted_label = adjusted_label.clone()
        failure_adjusted_label = delta_adjusted_label[neg_label]
        pos_in_neg = torch.where(failure_adjusted_label>=self.temperature_threshold)
        neg_in_neg = torch.where(failure_adjusted_label<-0.03) # TODO: this is a hyperparameter!!
        # neg_in_neg = torch.where(failure_adjusted_label<self.temperature_threshold)
        # loss = pos_output.mean() + 0.05*neg_output[pos_in_neg].mean() - 0.01*all_loss[neg_in_neg].mean() # + 0.05*F.relu(pos_distances.mean()-pos_neg_distances.mean())
        regular_loss = pos_output.mean()
        pos_in_neg_loss = neg_output[pos_in_neg].mean()
        if torch.isnan(pos_in_neg_loss):
            pos_in_neg_loss = torch.tensor(0.0, device=regular_loss.device)
        neg_in_neg_loss = neg_output[neg_in_neg].mean()
        if torch.isnan(neg_in_neg_loss):
            neg_in_neg_loss = torch.tensor(0.0, device=regular_loss.device)
        # loss = regular_loss + 0.05*pos_in_neg_loss - 0.01*neg_in_neg_loss
        # loss = regular_loss - 0.01*neg_in_neg_loss
        const = pos_in_neg_loss.detach()
        loss = regular_loss + 0.05*pos_in_neg_loss + 0.01*F.relu(const - neg_in_neg_loss)
        # if torch.isnan(neg_in_neg_loss) or torch.isnan(pos_in_neg_loss):
        #     loss = regular_loss.clone()
        return loss.mean()

    def cfm_loss_fn_same(self, model_output, model_target, label, temperature):
        # label has two dimension
        # 0: task definition, 1: pos/neg defination  

        # Version 1: we assume there is only one task, so only second dimension is used in the calculation
        neg_label = torch.where(label[:,1]==1)[0]
        pos_label = torch.where(label[:,1]==0)[0]
        chosen_neg_label = neg_label[torch.randint(0, len(neg_label), (len(pos_label),))]
        loss_positive = F.mse_loss(model_output[pos_label], model_target[pos_label], reduction='none')
        loss_negtive = F.mse_loss(model_output[pos_label], model_target[chosen_neg_label], reduction='none')
        temp_negtive = temperature[chosen_neg_label-len(pos_label)]
        # if loss_for_all_sample.dim() > 1:
        #     loss_for_all_sample = loss_for_all_sample.mean(dim=1)
        loss = loss_positive - temp_negtive * loss_negtive
        return loss.mean(), {}

if __name__ == "__main__":
    main()