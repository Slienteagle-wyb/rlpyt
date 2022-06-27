import torch
from collections import namedtuple
import copy
import wandb
import numpy as np
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.models.ul.encoders import DmlabEncoderModelNorm, ResEncoderModel
from rlpyt.ul.models.ul.atc_models import ByolMlpModel, DroneStateProj
from rlpyt.ul.algos.utils.data_augs import get_augmentation, random_shift
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.models.ul.drnn import DRnnCore
import torchvision.transforms as Trans
import torch.nn.functional as F


IGNORE_INDEX = -100  # Mask TC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["mstLoss", "spatialLoss", "temporalLoss",
                                 'cos_similarity', 'global_latents_std', "gradNorm", 'current_lr'])
ValInfo = namedtuple("ValInfo", ["mstLoss", "accuracy", "convActivation"])


class DroneMST(BaseUlAlgorithm):
    """
    Spatial and Temporal Contrastive Pretraining with  forward and inverse
    dyna model.
    """
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_T=32,  # the default length of extracted batch
            batch_B=16,  # batch B is the sampled batch size for extraction
            warmup_T=0,
            overshot_horizon=3,
            num_stacked_input=1,  # stacked input num equal to 1 when no stack
            clip_grad_norm=10.,
            target_update_tau=0.01,  # 1 for hard update
            target_update_interval=1,
            latent_size=256,
            hidden_sizes=512,
            deter_dim=1024,
            attitude_dim=9,
            vel_state_dim=4,
            random_shift_prob=1.,
            random_shift_pad=4,
            augmentations=('blur', 'intensity'),  # combined with intensity jit accord to SGI
            spatial_coefficient=1.0,
            temporal_coefficient=1.0,
            validation_split=0.0,
            n_validation_batches=0,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=ResEncoderModel,
            initial_state_dict=None,
            optim_kwargs=None,
            sched_kwargs=None,
            encoder_kwargs=None,
            replay_kwargs=None,
            rssm_kwargs=None
    ):
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.batch_size = batch_B * batch_T  # for logging only
        self._replay_T = warmup_T + batch_T  # self.replay_T == self._replay_T is the len of every sampled trajectory

    def initialize(self, epochs, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)
        torch.backends.cudnn.benchmark = True
        # torch.autograd.set_detect_anomaly(True)
        examples = self.load_replay()
        self.itrs_per_epoch = self.replay_buffer.size // self.batch_size
        self.n_updates = epochs * self.itrs_per_epoch
        print('total number of itrs is:', self.n_updates)
        self.image_shape = image_shape = examples.observation.shape  # [c, h, w]

        trans_dim = self.replay_buffer.translation_dim
        rotate_dim = self.replay_buffer.rotation_dim
        forward_input_size = rotate_dim + trans_dim  # no reward

        # resnet backbone
        self.encoder = self.EncoderCls(
            image_shape=image_shape,
            latent_size=self.latent_size,
            hidden_sizes=self.hidden_sizes,
            num_stacked_input=self.num_stacked_input,
            **self.encoder_kwargs
        )
        # dmlab_norm backbone
        # self.encoder = DmlabEncoderModelNorm(
        #     image_shape=image_shape,
        #     latent_size=self.latent_size,
        #     hidden_sizes=self.hidden_sizes
        # )
        self.target_encoder = copy.deepcopy(self.encoder)  # the target encoder is not tied with online encoder

        self.spatial_predictor = ByolMlpModel(
            input_dim=self.latent_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_sizes
        )

        self.temporal_predictor = ByolMlpModel(
            input_dim=self.deter_dim,
            latent_size=self.latent_size,
            hidden_size=self.hidden_sizes
        )
        # self.temporal_predictor = torch.nn.Linear(in_features=self.deter_dim,
        #                                           out_features=self.latent_size)

        self.drnn_model = DRnnCore(
            latent_dim=self.latent_size,
            embed_dim=self.latent_size,
            action_dim=forward_input_size,
            deter_dim=self.deter_dim,
            device=self.device,
            warmup_T=self.warmup_T,
            **self.rssm_kwargs
        )

        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.spatial_predictor.to(self.device)
        self.temporal_predictor.to(self.device)
        self.drnn_model.to(self.device)

        self.optim_initialize(epochs)

        # load the pretrained models
        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_B)  # batch b is the batch_size of every single trajectory
        current_epoch = itr // self.itrs_per_epoch
        if self.lr_scheduler is not None and itr % self.itrs_per_epoch == 0:
            self.lr_scheduler.step(current_epoch)
        current_lr = self.lr_scheduler.get_epoch_values(current_epoch)[0]

        self.optimizer.zero_grad()
        # calculate the loss func
        spatial_loss, temporal_loss, cos_sim, global_latents_std = self.mst_loss(samples)
        loss = spatial_loss + temporal_loss

        optimize_loss = self.spatial_coefficient * spatial_loss + self.temporal_coefficient * temporal_loss
        optimize_loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # log the optimization info/result
        opt_info.mstLoss.append(loss.item())
        opt_info.spatialLoss.append(spatial_loss.item())
        opt_info.temporalLoss.append(temporal_loss.item())
        # opt_info.predAccuracy.append(pred_accuracy.item())
        opt_info.cos_similarity.append(cos_sim.item())
        opt_info.global_latents_std.append(global_latents_std.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.current_lr.append(current_lr)

        # the update interval for the momentum encoder
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(), self.target_update_tau)
        return opt_info

    def mst_loss(self, samples):
        obs_one = samples.observations
        if obs_one.dtype == torch.uint8:
            default_float_dtype = torch.get_default_dtype()
            obs_one = obs_one.to(dtype=default_float_dtype).div(255)
        length, b, f, c, h, w = obs_one.shape
        obs_one = obs_one.view(length, b * f, c, h, w)
        obs_two = copy.deepcopy(obs_one)

        if self.random_shift_prob > 0.:
            obs_one = random_shift(
                imgs=obs_one,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )
            obs_two = random_shift(
                imgs=obs_two,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )

        cur_vel = samples.velocities
        cur_attitude = samples.attitudes
        cur_drone_state = torch.cat((cur_vel, cur_attitude), dim=-1)
        prev_translation = samples.prev_translations
        prev_rotation = samples.prev_rotations
        prev_action = torch.cat((prev_translation, prev_rotation), dim=-1)

        obs_one, obs_two, prev_action, cur_drone_state = buffer_to((obs_one, obs_two, prev_action, cur_drone_state),
                                                                   device=self.device)

        lead_dim, batch_len, batch_size, shape = infer_leading_dims(obs_one, 3)
        obs_one = obs_one.reshape(batch_len * batch_size, *shape)
        obs_two = obs_two.reshape(batch_len * batch_size, *shape)
        aug_trans = Trans.Compose(get_augmentation(self.augmentations, shape))
        obs_one = aug_trans(obs_one).reshape(batch_len, batch_size, *shape)
        obs_two = aug_trans(obs_two).reshape(batch_len, batch_size, *shape)

        with torch.no_grad():
            target_proj_one, _ = self.target_encoder(obs_one)
            target_proj_two, _ = self.target_encoder(obs_two)
        online_proj_one, _ = self.encoder(obs_one)
        online_proj_two, _ = self.encoder(obs_two)

        # calculate the byol spatial loss
        spatial_loss, cos_sim, global_latents_std = self.spatial_loss(online_proj_one, online_proj_two,
                                                                      target_proj_one, target_proj_two)
        # calculate the temporal loss
        temporal_loss = self.temporal_loss(online_proj_one, target_proj_two,
                                           online_proj_two, target_proj_one, prev_action)

        return spatial_loss, temporal_loss, cos_sim, global_latents_std

    def temporal_loss(self, online_proj_one, target_proj_two,
                      online_proj_two, target_proj_one, prev_actions):
        T, B, latent_dim = online_proj_one.shape
        init_state = self.drnn_model.init_state(B)
        # calculate the close loop rollout
        h_states_one = self.drnn_model.forward(online_proj_one, prev_actions, init_state, forward_pred=False)
        h_states_two = self.drnn_model.forward(online_proj_two, prev_actions, init_state, forward_pred=False)
        # calculate the open loop rollout
        if self.overshot_horizon > 0:
            temporal_overshoot_loss_one = self.overshot_loss(prev_actions, target_proj_two,
                                                             self.overshot_horizon, h_states_one)
            temporal_overshoot_loss_two = self.overshot_loss(prev_actions, target_proj_one,
                                                             self.overshot_horizon, h_states_two)
            temporal_overshoot_loss = temporal_overshoot_loss_one + temporal_overshoot_loss_two
        else:
            temporal_overshoot_loss = torch.zeros(1)
            pred_accuracy = None

        temporal_loss = temporal_overshoot_loss

        return temporal_loss

    def overshot_loss(self, prev_actions, target_proj, overshot_horizon, init_states):
        start_idxs = np.arange(0, self.batch_T - overshot_horizon)
        end_idxs = start_idxs + overshot_horizon
        base_labels = torch.arange(self.batch_T*self.batch_B, dtype=torch.long, device=self.device).view(self.batch_T,
                                                                                                         self.batch_B)
        target_proj_list = []
        sliced_actions = []
        init_state_h = []
        label_lists = []
        for start_idx, end_idx in zip(start_idxs, end_idxs):
            init_state_h.append(init_states[start_idx])
            sliced_actions.append(prev_actions[start_idx+1:end_idx+1])
            target_proj_list.append(target_proj[start_idx+1:end_idx+1])
            label_lists.append(base_labels[start_idx+1:end_idx+1].view(-1))

        init_state_h = torch.cat(init_state_h, dim=0)
        sliced_actions = torch.cat(sliced_actions, dim=1)
        target_projs = torch.cat(target_proj_list, dim=1)
        # open loop rollout for predictions
        states_h = self.drnn_model.forward_imagine(sliced_actions, init_state_h)

        overshot_length, num_batch, hidden_dim = states_h.shape
        assert overshot_length == overshot_horizon
        # calculate byol overshot predictive loss
        temporal_pred = self.temporal_predictor(states_h.view(overshot_length*num_batch, -1))
        target_projs = target_projs.detach().reshape(overshot_length*num_batch, -1)
        temporal_overshot_loss = self.byol_loss_fn(temporal_pred, target_projs)

        # # calculate contrastive loss
        # temporal_pred = self.temporal_predictor(states_h.view(overshot_length*num_batch, -1))
        # target_projs = target_projs.detach().reshape(overshot_length*num_batch, -1)
        # pred_logits = torch.matmul(temporal_pred, target_projs.transpose(1, 0))
        # pred_logits = pred_logits - torch.max(pred_logits, dim=-1, keepdim=True)[0]
        # labels = torch.cat(label_lists)
        # temporal_overshot_loss = self.c_e_loss(pred_logits, labels)
        # pred_accuracy = self.contrastive_metric(pred_logits, labels)

        return temporal_overshot_loss.mean()

    def spatial_loss(self, online_proj_one, online_proj_two,
                     target_proj_one, target_proj_two):
        T, B, latent_dim = online_proj_one.shape

        target_proj_one = target_proj_one.detach().view(-1, latent_dim)
        target_proj_two = target_proj_two.detach().view(-1, latent_dim)  # [T*B, latent_dim]

        spatial_pred_one = self.spatial_predictor(online_proj_one.view(-1, latent_dim))
        spatial_pred_two = self.spatial_predictor(online_proj_two.view(-1, latent_dim))

        loss_one = self.byol_loss_fn(spatial_pred_one, target_proj_two)
        loss_two = self.byol_loss_fn(spatial_pred_two, target_proj_one)
        spatial_loss = loss_one + loss_two
        # calculate byol metric
        cos_sim, global_latents_std = self.byol_metric(online_proj_one)
        return spatial_loss.mean(), cos_sim, global_latents_std

    def byol_loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2, eps=1e-3)
        y = F.normalize(y, dim=-1, p=2, eps=1e-3)
        return 2 - 2 * (x * y).sum(dim=-1)

    def byol_metric(self, proj_logits):
        T, B, latent_dim = proj_logits.shape
        global_latents = proj_logits.detach().view(-1, latent_dim)
        global_latents = F.normalize(global_latents, p=2.0, dim=-1, eps=1e-3)
        global_latents_std = global_latents.std(dim=0).mean()
        cos_sim = torch.matmul(global_latents, global_latents.transpose(1, 0))  # get a matrix [T*B, T*B]
        mask = 1 - torch.eye(T * B, device=self.device, dtype=torch.float)
        cos_sim = cos_sim * mask  # mask the similarity of every self
        offset = cos_sim.shape[-1] / (cos_sim.shape[-1] - 1)  # (T*B)/(T*B-1)
        cos_sim = cos_sim.mean() * offset
        return cos_sim, global_latents_std

    def contrastive_metric(self, pred_logits, labels):
        pred_correct = torch.argmax(pred_logits.detach(), dim=1) == labels
        overshot_pred_accuracy = torch.mean(pred_correct.float())
        return overshot_pred_accuracy

    def validation(self, itr):
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            spatial_predictor=self.spatial_predictor.state_dict(),
            temporal_predictor=self.temporal_predictor.state_dict(),
            drnn_model=self.drnn_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.spatial_predictor.load_state_dict(state_dict['spatial_predictor'])
        self.temporal_predictor.load_state_dict(state_dict['temporal_predictor'])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.target_encoder.parameters()
        yield from self.temporal_predictor.parameters()
        yield from self.spatial_predictor.parameters()
        yield from self.drnn_model.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.target_encoder.named_parameters()
        yield from self.temporal_predictor.named_parameters()
        yield from self.spatial_predictor.named_parameters()
        yield from self.drnn_model.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.target_encoder.eval()
        self.spatial_predictor.eval()
        self.temporal_predictor.eval()
        self.drnn_model.eval()

    def train(self):
        self.encoder.train()
        self.target_encoder.train()
        self.spatial_predictor.train()
        self.temporal_predictor.train()
        self.drnn_model.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example

    def wandb_log_code(self):
        wandb.save('./rlpyt/ul/algos/ul_for_rl/mst.py')
        wandb.save('./rlpyt/ul/models/ul/drnn.py')
