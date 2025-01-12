import cv2
import numpy as np
import torch
from collections import namedtuple
import wandb
from rlpyt.ul.models.ul.encoders import ResEncoderModel, Res18Encoder, DmlabEncoderModelNorm
from rlpyt.ul.models.ul.drnn import DRnnCore
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.mlp import MlpModel
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.ul.algos.utils.dataset_utils import *

OptInfo = namedtuple("OptInfo", ["predLoss", "gradNorm", 'current_lr',
                                 'train_errorVx', 'train_errorVy', 'train_errorVz', 'train_errorVyaw'])
ValInfo = namedtuple("ValInfo", ["ValPredLoss", 'val_errorVx', 'val_errorVy', 'val_errorVz', 'val_errorVyaw'])


class StateVelRegressBc(BaseUlAlgorithm):
    opt_info_fields = tuple(f for f in OptInfo._fields)

    def __init__(
            self,
            delta_T=0,
            batch_T=1,
            batch_B=512,
            clip_grad_norm=10.,
            validation_split=0.0,  # used for calculating num of epoch
            with_validation=True,
            latent_size=64,
            hidden_sizes=512,
            num_stacked_input=3,
            mlp_hidden_layers=None,
            action_dim=4,
            attitude_dim=9,
            state_latent_dim=16,
            TrainReplayCls=OfflineUlReplayBuffer,
            ValReplayCls=OfflineUlReplayBuffer,
            EncoderCls=ResEncoderModel,
            MlpCls=MlpModel,
            state_dict_filename=None,
            sched_kwargs=None,
            optim_kwargs=None,
            encoder_kwargs=None,
            train_replay_kwargs=None,
            val_replay_kwargs=None,
    ):
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        self.batch_size = batch_B * batch_T
        self._replay_T = delta_T + batch_T
        self.pred_loss_fn = torch.nn.MSELoss()

    def initialize(self, epochs, cuda_idx=None):
        self.device = torch.device('cpu') if cuda_idx is None else torch.device('cuda', index=cuda_idx)
        examples = self.load_replay(with_validation=self.with_validation)
        self.img_shape = examples.observation.shape
        self.itrs_per_epoch = self.train_buffer.size // self.batch_size
        self.n_updates = epochs * self.itrs_per_epoch
        print(self.itrs_per_epoch, self.n_updates)
        self.encoder = self.EncoderCls(
            image_shape=self.img_shape,
            hidden_sizes=self.hidden_sizes,
            latent_size=self.latent_size,
            num_stacked_input=self.num_stacked_input,
            **self.encoder_kwargs,
        )
        # self.encoder = self.EncoderCls(
        #     image_shape=self.img_shape,
        #     latent_size=self.latent_size,
        #     hidden_sizes=self.hidden_sizes
        # )
        # self.drnn_model = DRnnCore(
        #     latent_dim=self.latent_size,
        #     embed_dim=self.latent_size,
        #     action_dim=8,
        #     deter_dim=1024,
        #     device=self.device,
        #     warmup_T=0,
        #     hidden_sizes=512,
        #     linear_dyna=True,
        #     num_gru_layers=1,
        #     gru_type='gru',
        #     batch_norm=True
        # )
        # used as a linear projector
        # self.state_projector = torch.nn.Linear(
        #     in_features=self.attitude_dim,
        #     out_features=self.state_latent_dim,
        #     bias=False
        # )

        self.policy = self.MlpCls(
            input_size=self.encoder.output_size,
            # input_size=self.latent_size + self.state_latent_dim,
            hidden_sizes=self.mlp_hidden_layers,
            output_size=self.action_dim,
        )
        if self.state_dict_filename is not None:
            logger.log('models loading state dict ....')
            loaded_state_dict = torch.load(self.state_dict_filename,
                                           map_location=torch.device('cpu'))
            # the conv state dict was stored as encoder
            loaded_state_dict = loaded_state_dict.get('algo_state_dict', loaded_state_dict)
            loaded_encoder_dict = loaded_state_dict.get('encoder', loaded_state_dict)

            # conv_state_dict = OrderedDict([(k, v) for k, v in loaded_state_dict.items() if k.startswith('conv.')])
            self.encoder.load_state_dict(loaded_encoder_dict)
            logger.log('conv encoder has loaded the pretrained model')
        else:
            logger.log('models has not loaded any pretrained model yet!')
        self.encoder.to(self.device)
        # self.state_projector.to(self.device)
        # self.drnn_model.to(self.device)
        self.policy.to(self.device)

        self.optim_initialize(epochs)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.train_buffer.sample_batch(self.batch_B)  # batch b is the batch_size of every single trajectory
        current_epoch = itr // self.itrs_per_epoch
        if self.lr_scheduler is not None and itr % self.itrs_per_epoch == 0:
            self.lr_scheduler.step(current_epoch)
        current_lr = self.lr_scheduler.get_epoch_values(current_epoch)[0]
        self.optimizer.zero_grad()
        pred_loss, vel_errors = self.pred_loss(samples)
        pred_loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.predLoss.append(pred_loss.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.current_lr.append(current_lr)
        opt_info.train_errorVx.append(vel_errors[0])
        opt_info.train_errorVy.append(vel_errors[1])
        opt_info.train_errorVz.append(vel_errors[2])
        opt_info.train_errorVyaw.append(vel_errors[3])
        return opt_info

    def pred_loss(self, samples):
        obs = samples.observations
        if obs.dtype == torch.uint8:
            default_float_type = torch.get_default_dtype()
            obs = obs.to(dtype=default_float_type).div(255.0)

        length, batch_size, f, c, h, w = obs.shape
        obs = obs.view(length, batch_size * f, c, h, w)

        vel_states = samples.velocities
        attitude_states = samples.attitudes

        obs, vel_states, attitude_states = buffer_to((obs, vel_states, attitude_states), device=self.device)
        # proj_feat, conv_out = self.encoder(obs.reshape(length*batch_size*f, c, h, w))
        # with torch.no_grad():
        #     conv_out = self.encoder.conv(obs.reshape(length * batch_size * f, c, h, w))
        conv_out = self.encoder.conv(obs.reshape(length * batch_size * f, c, h, w))
        # state_embedding = self.state_projector(attitude_states)
        # policy_input = torch.cat((conv_out.reshape(length * batch_size * f, -1),
        #                           state_embedding.reshape(length * batch_size, -1)), dim=-1)
        # policy_input = torch.cat((conv_out.detach().reshape(length*batch_size*f, -1),
        #                           state_embedding.reshape(length*batch_size, -1)), dim=-1)
        # policy_input = torch.cat((conv_out.detach().reshape(length*batch_size*f, -1),
        #                           state_embedding.reshape(length*batch_size, -1)), dim=-1)
        pred_vel = self.policy(conv_out.reshape(length * batch_size * f, -1))
        pred_loss = self.pred_loss_fn(pred_vel, vel_states.reshape(length*batch_size, -1))
        # calculate validation metrics
        gt_vels = de_normalize_v(vel_states.reshape(length*batch_size, -1).detach().cpu().numpy())
        pred_vels = de_normalize_v(pred_vel.detach().cpu().numpy())
        vel_errors = np.absolute(pred_vels - gt_vels).mean(axis=0)

        return pred_loss, vel_errors

    def validation(self, itr):
        logger.log('computing validation loss .....')
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.itrs_per_epoch):
            samples = self.val_buffer.sample_batch(self.batch_B)
            with torch.no_grad():
                pred_loss, vel_errors = self.pred_loss(samples)
            val_info.ValPredLoss.append(pred_loss.item())
            val_info.val_errorVx.append(vel_errors[0])
            val_info.val_errorVy.append(vel_errors[1])
            val_info.val_errorVz.append(vel_errors[2])
            val_info.val_errorVyaw.append(vel_errors[3])
        self.optimizer.zero_grad()
        logger.log('.. validation loss completed')
        return val_info

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            # state_projector=self.state_projector.state_dict(),
            mlp_head=self.policy.state_dict()
        )

    def load_state_dict(self, initial_state_dict):
        """use for loading state dict for all the models"""
        pass

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        # self.state_projector.eval()
        self.policy.eval()

    def train(self):
        self.encoder.train()
        # self.state_projector.train()
        self.policy.train()

    def parameters(self):
        yield from self.encoder.parameters()
        # yield from self.state_projector.parameters()
        yield from self.policy.parameters()

    def named_parameters(self):
        yield from self.encoder.named_parameters()
        # yield from self.state_projector.named_parameters()
        yield from self.policy.named_parameters()

    def load_replay(self, with_validation=True):
        logger.log('Loading train replay buffer ...')
        self.train_buffer = self.TrainReplayCls(OfflineDatasets, **self.train_replay_kwargs)
        if with_validation is True:
            logger.log('loading validation replay buffer ...')
            self.val_buffer = self.ValReplayCls(OfflineDatasets, **self.val_replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.train_buffer.get_example()
        return example

    def wandb_log_code(self):
        wandb.save('./rlpyt/ul/algos/downstreams/state_mlp_vel_regressor.py')
