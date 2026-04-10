if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
import traceback
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import dill
import pickle
from diffusion_policy.common.sampler import ProportionalSampler
from diffusion_policy.common.sampler import SamplerForTestClassifier
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.pretrain_classifier_policy import PretrainClassfierPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
# from transformers import AutoImageProcessor, AutoModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

REPO_DIR = '/home/shuqi/dinov3'
CKP_DIR = '/home/shuqi/LeveFD'

def mem(tag=""):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{tag}] alloc={alloc:.1f}MB reserved={reserv:.1f}MB max_alloc={max_alloc:.1f}MB")

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)  

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        if cfg.load_model_ckp:
            model_payload = torch.load(open(cfg.ckp_path, 'rb'), pickle_module=dill)
            self.load_payload(model_payload, strict=False)

        self.use_classifier = cfg.policy.use_classifier
        if self.use_classifier:
            policy_cfg = copy.deepcopy(cfg.policy)
            policy_cfg._target_ = 'diffusion_policy.policy.pretrain_classifier_policy.PretrainClassfierPolicy' # PretrainClassfierPolicy 
            policy_cfg.obs_encoder.rgb_model.name = 'dinov3'
            policy_cfg.obs_encoder.rgb_model.weights = 'dinov3_vitb16'
            # policy_cfg.infer_loss_type = 'flow_matching' # PretrainClassfierPolicy 
            self.classifier = hydra.utils.instantiate(policy_cfg)

            payload = torch.load(open(cfg.classifier_ckp_path, 'rb'), pickle_module=dill)
            payload['pickles'].clear()
            if cfg.training.use_ema:
                payload['state_dicts']['classifier'] = payload['state_dicts'].pop('ema_model')
            self.load_payload(payload, exclude_keys=['model', 'ema_model', 'optimizer'], strict=False)
            self.classifier.eval()

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        if cfg.training.loss_type == 'contrastive':
            # configure dataset
            self.contrastive_sampler = ProportionalSampler(
                total_length=len(dataset),
                divide_index=dataset.get_divide_length(),
                per = cfg.training.success_per,
                batch_size = cfg.dataloader.batch_size
            )
            # train_dataloader = DataLoader(dataset, batch_sampler=self.contrastive_sampler, **cfg.dataloader)
            train_dataloader = DataLoader(dataset, batch_sampler=self.contrastive_sampler, 
                                        num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
                                        persistent_workers=cfg.dataloader.persistent_workers)
            normalizer = dataset.get_normalizer()
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        loss_fn = self.model.loss_fn_list[self.cfg.training.loss_type]
        
        if self.cfg.training.loss_type == 'diffusion':
            val_loss_fn = self.model.loss_fn_list['diffusion']
        else:
            val_loss_fn = self.model.loss_fn_list['flow_matching']  
        
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        if self.use_classifier:
            self.classifier.to(device)

        # save batch for sampling
        train_sampling_batch = None
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        train_indice_text = np.where(dataset.train_mask)
        val_indice_text = np.where(dataset.val_mask)
        dataset_indice_text = os.path.join(self.output_dir, 'dataset_indice.txt')
        with open(dataset_indice_text, 'wb') as f:
            np.savetxt(f, train_indice_text, fmt="%d", delimiter=",")
        with open(dataset_indice_text, 'ab') as f:
            np.savetxt(f, val_indice_text, fmt="%d", delimiter=",")
        text = input('Any comment on this training?\n')
        with open(os.path.join(self.output_dir, 'comment.txt'), "w", encoding="utf-8") as f:
            f.write(text)
        code_pth = os.path.join(self.output_dir, 'code')
        os.makedirs(code_pth, exist_ok=True)
        for name in ['diffusion_policy', 'policy_test_environment.py', 'related_tool.py', 'train.py']:
            # s = os.path.join('/home/msc-auto/szhao/LeveFD', name)
            s = os.path.join('/home/shuqi/LeveFD', name)
            d = os.path.join(code_pth, name)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            elif os.path.isfile(s):
                shutil.copy2(s, d)

        # mem('start')

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        # mem('after for')
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # mem('after batch')
                        # compute loss
                        # raw_loss, temp_corr, loss_pos_neg, loss_for_pos_mean = loss_fn(batch) # for our policy
                        if self.use_classifier:
                            with torch.no_grad():
                                score = self.classifier.classifier_infer(batch)
                            raw_loss, extra_info = loss_fn([batch, score])
                        else:
                            raw_loss, extra_info = loss_fn(batch)
                        # if self.cfg.training.loss_type == 'contrastive':
                        #     raw_loss = self.model.compute_cfm_loss(batch)
                        # elif self.cfg.training.loss_type == 'flow_matching':
                        #     raw_loss = self.model.compute_fm_loss(batch)
                        # elif :
                        #     raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        
                        if not torch.isfinite(loss):
                            self.optimizer.zero_grad(set_to_none=True)
                            # print("skip micro-batch due to nan/inf loss:", raw_loss)
                            continue
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        for key in extra_info.keys():
                            step_log[key] = extra_info[key]
                        # step_log['temperature'] = temp_corr
                        # step_log['loss_pos_neg'] = loss_pos_neg
                        # step_log['flow_loss'] = loss_for_pos_mean

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                # only test success data
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # if self.cfg.training.loss_type == 'contrastive':
                                #     loss = self.model.compute_cfm_loss(batch)
                                # else:
                                # del batch['obs']['label']
                                loss,_ = val_loss_fn(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and not isinstance(self.model, PretrainClassfierPolicy):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(freeze_encoder=cfg.training.freeze_encoder)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def generate_image_feature(self):
        cfg = copy.deepcopy(self.cfg)
        device = cfg.training.device

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # if cfg.training.loss_type == 'contrastive':
        #     # configure dataset
        #     self.contrastive_sampler = ProportionalSampler(
        #         total_length=len(dataset),
        #         divide_index=dataset.get_divide_length(),
        #         per = cfg.training.success_per,
        #         batch_size = cfg.dataloader.batch_size
        #     )
        #     # train_dataloader = DataLoader(dataset, batch_sampler=self.contrastive_sampler, **cfg.dataloader)
        #     train_dataloader = DataLoader(dataset, batch_sampler=self.contrastive_sampler, 
        #                                 num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
        #                                 persistent_workers=cfg.dataloader.persistent_workers)
        #     normalizer = dataset.get_normalizer()
        # else:
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.model.to(device)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        feature_list = []
        batch_idx_list = []
        self.model.eval()

        # if cfg.training.freeze_encoder:
        #     self.model.obs_encoder.eval()
        #     self.model.obs_encoder.requires_grad_(False)

        # self.epoch = 1

        offset = 0
        D = 1536
        N = len(train_dataloader.dataset) # + len(val_dataloader.dataset)  # 如果你的 dataset __len__ 是样本数
        feat_mm = np.lib.format.open_memmap(
            "/home/msc-auto/szhao/LeveFD/tmp_result/dino_feature.npy",
            mode="w+", shape=(N, D)
        )
        batch_idx_mm = np.lib.format.open_memmap(
            "/home/msc-auto/szhao/LeveFD/tmp_result/dino_batch_idx.npy",
            mode="w+", shape=(N,)
        )

        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch_all in enumerate(tepoch):
                # device transfer
                batch = batch_all[0]
                del batch['obs']['label']
                nobs = self.model.normalizer.normalize(batch['obs'])
                bsz = nobs['front_rgb'].shape[0]
                this_nobs = dict_apply(nobs, lambda x: x[:,:2,...].reshape(-1,*x.shape[2:]).to(device))
                # this_nobs = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                # this_nobs = {'image': batch['obs']['image']}
                nobs_features = self.model.obs_encoder(this_nobs).reshape(bsz, -1)
                # batch_idx_list.append(batch_all[1].detach().cpu().numpy())
                # feature_list.append(nobs_features.detach().cpu().numpy())
                feat_mm[offset:offset+bsz] = nobs_features.detach().cpu().numpy()
                batch_idx_mm[offset:offset+bsz] = batch_all[1].detach().cpu().numpy()
                offset = offset + bsz

        # with tqdm.tqdm(val_dataloader, desc=f"Training epoch {self.epoch}", 
        #         leave=False, mininterval=cfg.training.tqdm_interval_sec) as val_tepoch:
        #     for val_batch_idx, val_batch_all in enumerate(val_tepoch):
        #         val_batch = val_batch_all[0]
        #         del val_batch['obs']['label']
        #         nobs = self.model.normalizer.normalize(val_batch['obs'])
        #         bsz = nobs['image'].shape[0]
        #         this_nobs = dict_apply(nobs, 
        #             lambda x: x[:,:2,...].reshape(-1,*x.shape[2:]))
        #         # this_nobs = {'image': batch['image']}
        #         nobs_features = self.model.obs_encoder(this_nobs).reshape(bsz, -1)
        #         # batch_idx_list.append(val_batch_all[1].detach().cpu().numpy())
        #         # feature_list.append(nobs_features.detach().cpu().numpy())                
        #         feat_mm[offset:offset+bsz] = nobs_features.detach().cpu().numpy()
        #         batch_idx_mm[offset:offset+bsz] = val_batch_all[1].detach().cpu().numpy()
        #         offset = offset + bsz

        # batch_idx_np = np.hstack(batch_idx_list)
        # feature_list_np = np.hstack(feature_list, dim=0)
        # np.save('/home/shuqi/LeveFD/tmp_result/r3m_batch_idx.npy', batch_idx_np)
        # np.save('/home/shuqi/LeveFD/tmp_result/r3m_feature.npy', feature_list_np)
        feat_mm.flush()
        batch_idx_mm.flush()

    def test_image_similarity(self):
        import matplotlib.pyplot as plt
        from datetime import datetime
        import dill, time

        cfg = copy.deepcopy(self.cfg)
        label_test = True
        
        payload = torch.load(open(cfg.classifier_ckp_path, 'rb'), pickle_module=dill)
        # self.load_scorer_payload(payload, strict=False)
        self.load_payload(payload, strict=False)

        # ckp_folder = cfg.ckp_path.replace('/mnt/ssd1/szhao/LeveFD/ckp/', '')
        # ckp_folder = ckp_folder.replace('/checkpoints', '')
        # ckp_folder = ckp_folder.replace('.ckpt', '')
        # ckp_folder = ckp_folder.replace('/', '_')

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        normalizer = dataset.get_normalizer()
        # episode_ends = np.concatenate(([0], dataset.get_episode_ends()))
        episode_index = dataset.get_episode_index()
        rollout_traj_index = np.array([0]) # np.arange(200, 256) # np.array(sorted(random.sample(range(235, 240), 5))) # np.array([0,235]) 
        print(rollout_traj_index)
        st = []
        end = []
        for idx in rollout_traj_index:
            st.append(episode_index[idx][0])
            end.append(episode_index[idx][-1]+1)
        # st, end = episode_ends[rollout_traj_index], episode_ends[rollout_traj_index+1]
        # cr_index = 0
        rollout_cr_index = 0
        rollout_result = []
        tmp_list = []

        test_sampler = SamplerForTestClassifier(episode_index, rollout_traj_index)
        train_dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler,num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
                                        persistent_workers=cfg.dataloader.persistent_workers)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.model.eval()

        test_image = np.load('/home/msc-auto/szhao/LeveFD/test_image.npy')
        test_image = np.concatenate([test_image[:1].copy(), test_image], axis=0)
        test_image_traj = []

        for i in range(172):
            sample_img = np.moveaxis(test_image[i:i+2],-1,1)/255
            batch = {'obs':{'front_rgb': torch.tensor(sample_img, device=device).float()}}
            nobs = self.model.normalizer.normalize(batch['obs'])
            this_nobs = dict_apply(nobs, lambda x: x.to(device))
            nobs_features = self.model.obs_encoder(this_nobs).flatten().detach().cpu().numpy()
            if i == 0:
                debug_obs_test = copy.deepcopy(this_nobs)
                debug_obs_test_feature = copy.deepcopy(nobs_features)
            test_image_traj.append(nobs_features)
        test_image_traj = np.array(test_image_traj)

        correct_image = np.load('/home/msc-auto/szhao/LeveFD/test_image_true.npy')
        correct_image = np.concatenate([correct_image[:1].copy(), correct_image], axis=0)
        correct_image_traj = []

        for i in range(172):
            sample_img = np.moveaxis(correct_image[i:i+2],-1,1)/255
            batch = {'obs':{'front_rgb': torch.tensor(sample_img, device=device).float()}}
            nobs = self.model.normalizer.normalize(batch['obs'])
            this_nobs = dict_apply(nobs, lambda x: x.to(device))
            nobs_features = self.model.obs_encoder(this_nobs).flatten().detach().cpu().numpy()
            if i == 0:
                debug_obs_correct = copy.deepcopy(this_nobs)
                debug_obs_correct_feature = copy.deepcopy(nobs_features)
            correct_image_traj.append(nobs_features)
        correct_image_traj = np.array(correct_image_traj)

        dist = np.linalg.norm(correct_image_traj-test_image_traj, axis=-1)
        
        # now_faeture = np.array(rollout_traj_index[0])
        # test_image_traj = np.array(test_image_traj)
        # uuu = np.linalg.norm(now_faeture-test_image_traj, axis=-1)
        plt.plot(dist)
        plt.savefig('/home/msc-auto/szhao/LeveFD/tmp_result/dis.png', dpi=300, bbox_inches="tight")

    def test_classifier_feature(self):
        import matplotlib.pyplot as plt
        from datetime import datetime
        import dill, time

        cfg = copy.deepcopy(self.cfg)
        label_test = True
        
        payload = torch.load(open(cfg.classifier_ckp_path, 'rb'), pickle_module=dill)
        # self.load_scorer_payload(payload, strict=False)
        self.load_payload(payload, strict=False)

        # ckp_folder = cfg.ckp_path.replace('/mnt/ssd1/szhao/LeveFD/ckp/', '')
        # ckp_folder = ckp_folder.replace('/checkpoints', '')
        # ckp_folder = ckp_folder.replace('.ckpt', '')
        # ckp_folder = ckp_folder.replace('/', '_')

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        normalizer = dataset.get_normalizer()
        # episode_ends = np.concatenate(([0], dataset.get_episode_ends()))
        episode_index = dataset.get_episode_index()
        rollout_traj_index = np.arange(228,229) #np.arange(200, 232) # np.arange(200, 256) # np.array(sorted(random.sample(range(235, 240), 5))) # np.array([0,235]) 
        print(rollout_traj_index)
        st = []
        end = []
        for idx in rollout_traj_index:
            st.append(episode_index[idx][0])
            end.append(episode_index[idx][-1]+1)
        # st, end = episode_ends[rollout_traj_index], episode_ends[rollout_traj_index+1]
        # cr_index = 0
        rollout_cr_index = 0
        rollout_result = []
        tmp_list = []

        sampler = SamplerForTestClassifier(episode_index, rollout_traj_index)
        train_dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
                                        persistent_workers=cfg.dataloader.persistent_workers)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.model.eval()

        with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                idx = batch[1]
                # classifer_output = self.model.classifier_loss(batch)
                if idx >=st[rollout_cr_index] and idx<end[rollout_cr_index]:  
                    batch = batch[0]
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    classifer_output = self.model.sample_function(batch) 
                    tmp_label = float(classifer_output[-1][0].detach().cpu()-classifer_output[0][0].detach().cpu())*2.5
                    if tmp_label>=0.2:
                        print()
                    tmp_list.append(tmp_label)
                    if idx == end[rollout_cr_index] - 1:
                        print('finish:', rollout_traj_index[rollout_cr_index])
                        print(classifer_output[0])
                        rollout_cr_index = rollout_cr_index + 1
                        rollout_result.append(tmp_list)
                        tmp_list = []
                        if rollout_cr_index >= rollout_traj_index.shape[0]:
                            break
                # cr_index = cr_index+1
                # classifer_output = self.model.classifier_infer(batch)

        # single trajectory comparison
        # test_dataset: BaseImageDataset
        # cfg.task.dataset.zarr_path = '/home/msc-auto/szhao/LeveFD/pick_and_place_test_false.pkl'
        # test_dataset = hydra.utils.instantiate(cfg.task.dataset)
        # episode_index_test = test_dataset.get_episode_index()
        # rollout_traj_index_test = np.array([0]) # np.arange(200, 256) # np.array(sorted(random.sample(range(235, 240), 5))) # np.array([0,235]) 
        # print(rollout_traj_index_test)
        # st_test = []
        # end_test = []
        # for idx in rollout_traj_index_test:
        #     st_test.append(episode_index_test[idx][0])
        #     end_test.append(episode_index_test[idx][-1]+1)
        # # st, end = episode_ends[rollout_traj_index], episode_ends[rollout_traj_index+1]
        # # cr_index = 0
        # rollout_cr_index_test = 0
        # rollout_result_test = []
        # tmp_list_test = []

        # test_sampler = SamplerForTestClassifier(episode_index_test, rollout_traj_index_test)
        # train_dataloader_test = DataLoader(test_dataset, batch_size=1, sampler=test_sampler,num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
        #                                 persistent_workers=cfg.dataloader.persistent_workers)

        # with tqdm.tqdm(train_dataloader_test, desc=f"Training epoch {self.epoch}", 
        #         leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
        #     for batch_idx, batch in enumerate(tepoch):
        #         # device transfer
        #         idx = batch[1]
        #         # classifer_output = self.model.classifier_loss(batch)
        #         if idx >=st_test[rollout_cr_index_test] and idx<end_test[rollout_cr_index_test]:  
        #             batch = batch[0]
        #             batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        #             classifer_output = self.model.sample_function(batch)[:,8]                    
        #             tmp_list_test.append(float(classifer_output[0].detach().cpu()))
        #             if idx == end_test[rollout_cr_index_test] - 1:
        #                 print('finish:', rollout_traj_index_test[rollout_cr_index_test])
        #                 print(classifer_output[0])
        #                 rollout_cr_index_test = rollout_cr_index_test + 1
        #                 rollout_result_test.append(tmp_list_test)
        #                 tmp_list_test = []
        #                 if rollout_cr_index_test >= rollout_traj_index_test.shape[0]:
        #                     break

        # # plt.ylim(0.45, 1.1)
        # plt.figure()
        # plt.plot(rollout_result_test[0], label='false')
        # plt.plot(rollout_result[0], label='true')
        # plt.grid(True)
        # plt.legend()
        # plt.savefig("/home/msc-auto/szhao/LeveFD/tmp_result/pc.png", dpi=300, bbox_inches="tight")  
        
        # save all performance
        count = 0
        # plt.figure(figsize=(10, 5))
        for j in range(len(rollout_result)):
            count = count + 1
            traj_num = len(rollout_result[j])
            step = 100/traj_num
            x = np.arange(traj_num) * step
            plt.plot(x, rollout_result[j], label=str(rollout_traj_index[j]), linewidth=3)
            # ts = datetime.now().strftime("%m%d_%H%M%S")
            # filename = f"plot_{ts}.png"
            # plt.legend()
            # plt.savefig(os.path.join("/home/msc-auto/szhao/LeveFD/tmp_result", filename), dpi=300, bbox_inches="tight")
            # if count%2==0 and count!=0:
            #     plt.ylim(-0.3, 0.3)
            #     plt.grid(True)
            #     ts = datetime.now().strftime("%m%d_%H%M%S")
            #     time.sleep(1) 
            #     filename = f"plot_{ts}.png"
            #     plt.legend()
            #     plt.savefig(os.path.join("/home/shuqi/LeveFD/tmp_result/17_46_01", filename), dpi=300, bbox_inches="tight")
            #     # plt.savefig(os.path.join("/home/shuqi/LeveFD/tmp_result/16_05_02", filename), dpi=300, bbox_inches="tight")
            #     plt.figure(figsize=(10, 5))
            if count%1==0 and count!=0:
                # plt.ylim(-0.3, 0.3)
                # plt.grid(True)
                ts = datetime.now().strftime("%m%d_%H%M%S")
                time.sleep(1) 
                filename = f"plot_{ts}.png"
                # plt.legend()
                plt.savefig("/home/shuqi/LeveFD/tmp_result/fig4.png", bbox_inches="tight", pad_inches=0, transparent=True)
                # plt.savefig(os.path.join("/home/shuqi/LeveFD/tmp_result/17_46_01", filename), dpi=300, bbox_inches="tight")
                # plt.savefig(os.path.join("/home/shuqi/LeveFD/tmp_result/16_05_02", filename), dpi=300, bbox_inches="tight")
                # plt.figure(figsize=(10, 5))
        # plt.savefig("/home/msc-auto/szhao/LeveFD/tmp_result/plot_compare_8.png", dpi=300, bbox_inches="tight")  

    def generate_image_feature_new(self):
        from torchvision import transforms
        cfg = copy.deepcopy(self.cfg)
        device = cfg.training.device

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        data_path = '/mnt/ssd1/szhao/LeveFD/rlbench_dataset/pick_and_lift_pkl/pick_and_place_same_color_all.pkl'  
        try:
            with open(data_path, "rb") as f:
                root_data_original = NumpyCompatUnpickler(f).load()
                # with open(zarr_path, "rb") as f:
                #     root_data_original = pickle.load(f)
        except Exception as e:
            print("!! ERROR in pickle.load:", repr(e))
            traceback.print_exc()
            sys.exit(1)        

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.model.to(device)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # [0,255] -> [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        img_keys_dataset = [k for k in root_data_original['data'] if 'rgb' in k]
        img_keys = dataset.img_key
        for k in img_keys:
            assert k in img_keys_dataset
        img_observation = {}
        for k in img_keys:
            tmp_obs = root_data_original['data'][k]
            # tmp_img = np.moveaxis(tmp_obs,-1,1)/255
            img_observation[k] = tmp_obs
        # img_observation = root_data_original['data']['front_rgb']
        # img_obs = np.moveaxis(img_observation,-1,1)/255
        length = img_observation[img_keys[0]].shape[0]
        saved_feature = np.zeros((length, 768*len(img_keys)))

        weights = 'dinov3_vitb16'
        ckp_path = os.path.join(CKP_DIR, weights + '.pth')
        dinov3_model = torch.hub.load(REPO_DIR, weights, source='local', weights=ckp_path)
        dinov3_model.to(device)
        
        for i in range(length):
            tmp_feature = []
            for k in img_keys:
                # cur_obs = transform(np.moveaxis(img_observation[k][i],-1,0)/255)
                # tmp_feature.append(dinov3_model(torch.tensor(cur_obs, device=device)[None,:].float()).detach().cpu().numpy()[0])
                img = img_observation[k][i] / 255.0
                img = torch.tensor(img).permute(2, 0, 1).float()
                img = img[[2,1,0], :, :]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img = (img - mean) / std
                tmp_feature.append(dinov3_model(img[None,:].to(device)).detach().cpu().numpy()[0])
            saved_feature[i] = np.concatenate(tmp_feature)

        np.save('/home/shuqi/LeveFD/tmp_result/dinov3/dinov3_pick_and_lift_rgb.npy', saved_feature)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    # workspace.generate_image_feature()
    workspace.run()
    # workspace.test_classifier_feature()
    # workspace.test_image_similarity()
    # workspace.generate_image_feature_new()

if __name__ == "__main__":
    main()
