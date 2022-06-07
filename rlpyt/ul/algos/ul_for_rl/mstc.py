import numpy as np
import torch
from collections import namedtuple
import copy
import wandb
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.tensor import valid_mean
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.encoders import FusResEncoderModel, Res18Encoder
from rlpyt.ul.models.ul.atc_models import ByolMlpModel, DroneStateProj
from rlpyt.ul.algos.utils.data_augs import get_augmentation, random_shift
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.models.ul.rssm import RSSMCore
import torchvision.transforms as Trans
import torch.nn.functional as F
import torch.distributions as D


IGNORE_INDEX = -100  # Mask TC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["mstcLoss", "spatialLoss", "temporalLoss", 'klLoss',
                                 'entropy_posts', 'entropy_priors',
                                 'cos_similarity', "gradNorm", 'current_lr'])

ValInfo = namedtuple("ValInfo", ["mstLoss", "accuracy", "convActivation"])


class DroneMSTC(BaseUlAlgorithm):
    """
    Spatial and Temporal Contrastive Pretraining with  forward and inverse
    dyna model, using moco style self-surpervised learning object.
    """
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            warmup_T=16,
            batch_T=32,  # the default length of extracted batch
            batch_B=16,  # batch B is the sampled batch size for extraction
            num_stacked_input=3,
            clip_grad_norm=10.,
            target_update_tau=0.01,  # 1 for hard update
            target_update_interval=1,
            latent_size=256,
            hidden_sizes=512,
            stoch_dim=32,
            stoch_discrete=32,
            attitude_dim=9,
            vel_state_dim=4,
            random_shift_prob=1.,
            random_shift_pad=6,
            augmentations=('blur', 'intensity'),  # combined with intensity jit accord to SGI
            spatial_coefficient=1.0,
            temporal_coefficient=1.0,
            kl_coefficient=1.0,
            kl_balance=0.5,
            validation_split=0.0,
            n_validation_batches=0,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=FusResEncoderModel,
            initial_state_dict=None,
            optim_kwargs=None,
            sched_kwargs=None,
            encoder_kwargs=None,
            replay_kwargs=None,
            rssm_kwargs=None,
    ):
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.batch_size = batch_B * batch_T  # for logging only
        self._replay_T = batch_T  # self.replay_T == self._replay_T is the len of every sampled trajectory

    def initialize(self, epochs, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)
        torch.backends.cudnn.benchmark = True
        examples = self.load_replay()
        self.itrs_per_epoch = self.replay_buffer.size // self.batch_size
        self.n_updates = epochs * self.itrs_per_epoch
        self.image_shape = examples.observation.shape  # [c, h, w]

        trans_dim = self.replay_buffer.translation_dim
        rotate_dim = self.replay_buffer.rotation_dim
        forward_input_size = rotate_dim + trans_dim  # no reward

        self.encoder = self.EncoderCls(
            image_shape=self.image_shape,
            latent_size=self.latent_size,
            hidden_sizes=self.hidden_sizes,
            num_stacked_input=self.num_stacked_input,
            **self.encoder_kwargs
        )
        self.target_encoder = copy.deepcopy(self.encoder)  # the target encoder is not tied with online encoder

        self.drone_state_proj = DroneStateProj(
            input_dim=self.attitude_dim + self.vel_state_dim,
            latent_size=self.latent_size
        )
        self.target_drone_state_proj = copy.deepcopy(self.drone_state_proj)

        # transforms = [None]
        # for _ in range(self.batch_T - self.warmup_T - 1):
        #     transforms.append(
        #         torch.nn.Linear(in_features=self.hidden_sizes, out_features=self.hidden_sizes, bias=False)
        #     )
        # self.transforms = torch.nn.ModuleList(transforms)

        # self.partial_trans = torch.nn.Linear(self.hidden_sizes, self.stoch_dim * (self.stoch_discrete or 1), bias=False)
        # self.full_trans = torch.nn.Linear(self.latent_size, self.hidden_sizes, bias=False)

        self.feature_pred_head = ByolMlpModel(
            input_dim=self.latent_size,
            latent_size=self.hidden_sizes + self.stoch_dim * (self.stoch_discrete or 1),
            hidden_size=self.hidden_sizes
        )
        self.image_embed_pred_head = ByolMlpModel(
            input_dim=self.hidden_sizes + self.stoch_dim * (self.stoch_discrete or 1),
            latent_size=self.latent_size,
            hidden_size=self.hidden_sizes
        )

        self.spatial_predictor = ByolMlpModel(
            input_dim=self.latent_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_sizes
        )
        # self.spatial_temporal_predictor = ByolMlpModel(
        #     input_dim=self.latent_size,
        #     latent_size=self.stoch_dim * (self.stoch_discrete or 1) + self.hidden_sizes,
        #     hidden_size=self.hidden_sizes
        # )

        self.rssm_model = RSSMCore(
            embed_dim=self.latent_size,
            action_dim=forward_input_size,
            deter_dim=self.hidden_sizes,
            device=self.device,
            stoch_dim=self.stoch_dim,
            stoch_discrete=self.stoch_discrete,
            warmup_T=self.warmup_T,
            **self.rssm_kwargs
        )

        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.drone_state_proj.to(self.device)
        self.target_drone_state_proj.to(self.device)
        self.spatial_predictor.to(self.device)
        # self.spatial_temporal_predictor.to(self.device)
        self.rssm_model.to(self.device)
        self.feature_pred_head.to(self.device)
        self.image_embed_pred_head.to(self.device)

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
        spatial_loss, temporal_loss, kl_loss, cos_sim, entropy_posts, entropy_priors = self.mstc_loss(samples)

        optimize_loss = self.spatial_coefficient * spatial_loss + self.temporal_coefficient * temporal_loss + \
                        self.kl_coefficient * kl_loss
        optimize_loss.backward()

        loss = spatial_loss + temporal_loss + kl_loss

        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # log the optimize info/result
        opt_info.mstcLoss.append(loss.item())
        opt_info.spatialLoss.append(spatial_loss.item())
        opt_info.temporalLoss.append(temporal_loss.item())
        opt_info.klLoss.append(kl_loss.item())
        opt_info.cos_similarity.append(cos_sim.item())
        opt_info.entropy_posts.append(entropy_posts.item())
        opt_info.entropy_priors.append(entropy_priors.item())
        # opt_info.predAccuracy.append(accuracies[0].item())
        # opt_info.predAccuracyTm.append(accuracies[1].item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.current_lr.append(current_lr)

        # the update interval for the momentum encoder
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(), self.target_update_tau)
            update_state_dict(self.target_drone_state_proj, self.drone_state_proj.state_dict(), self.target_update_tau)
        return opt_info

    def mstc_loss(self, samples):
        obs_one = samples.observations
        if obs_one.dtype == torch.uint8:
            default_float_dtype = torch.get_default_dtype()
            obs_one = obs_one.to(dtype=default_float_dtype).div(255.0)
        length, b, f, c, h, w = obs_one.shape
        obs_one = obs_one.view(length, b * f, c, h, w)  # Treat all T,B as separate.(reshape the sample)
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
            target_spatial_embed_one, target_temporal_embed_one, _ = self.target_encoder(obs_one)
            target_spatial_embed_two, target_temporal_embed_two, _ = self.target_encoder(obs_two)
            target_temporal_embed_one = target_temporal_embed_one + self.target_drone_state_proj(cur_drone_state)
            # target_temporal_embed_two = target_temporal_embed_two + self.target_drone_state_proj(cur_drone_state)

        online_spatial_embed_one, online_temporal_embed_one, _ = self.encoder(obs_one)
        online_spatial_embed_two, online_temporal_embed_two, _ = self.encoder(obs_two)
        online_temporal_embed_one = online_temporal_embed_one + self.drone_state_proj(cur_drone_state)
        # online_temporal_embed_two = online_temporal_embed_two + self.drone_state_proj(cur_drone_state)

        spatial_loss, cos_sim = self.spatial_loss(online_spatial_embed_one, online_spatial_embed_two,
                                                  target_spatial_embed_one, target_spatial_embed_two)
        temporal_loss, kl_loss, entropy_posts, entropy_prior = self.spatial_temporal_loss(online_temporal_embed_one,
                                                                                          target_temporal_embed_one,
                                                                                          prev_action)

        return spatial_loss, temporal_loss, kl_loss, cos_sim, entropy_posts, entropy_prior

    def spatial_loss(self, online_spatial_embed_one, online_spatial_embed_two,
                     target_spatial_embed_one, target_spatial_embed_two):
        T, B, latent_dim = online_spatial_embed_one.shape

        target_spatial_embed_one = target_spatial_embed_one.detach().view(-1, latent_dim)
        target_spatial_embed_two = target_spatial_embed_two.detach().view(-1, latent_dim)

        pred_one = self.spatial_predictor(online_spatial_embed_one.reshape(-1, latent_dim))
        pred_two = self.spatial_predictor(online_spatial_embed_two.reshape(-1, latent_dim))

        loss_one = self.byol_loss_fn(pred_one, target_spatial_embed_two)
        loss_two = self.byol_loss_fn(pred_two, target_spatial_embed_one)
        spatial_loss = loss_one + loss_two

        # calculate metrix
        global_latents = online_spatial_embed_one.detach().view(-1, latent_dim)
        global_latents = F.normalize(global_latents, p=2.0, dim=-1, eps=1e-3)
        cos_sim = torch.matmul(global_latents, global_latents.transpose(1, 0))  # get a matrix [T*B, T*B]
        mask = 1 - torch.eye(T * B, device=self.device, dtype=torch.float)
        cos_sim = cos_sim * mask  # mask the similarity of every self
        offset = cos_sim.shape[-1] / (cos_sim.shape[-1] - 1)  # (T*B)/(T*B-1)
        cos_sim = cos_sim.mean() * offset

        return spatial_loss, cos_sim

    def spatial_temporal_loss(self, online_temporal_embed_one, target_temporal_embed_one, prev_action):
        init_state = self.rssm_model.init_state(self.batch_B)
        target_temporal_embed_one = target_temporal_embed_one.detach()

        with torch.no_grad():
            # target full branch
            target_temporal_embed_one_rssm = target_temporal_embed_one.clone()
            target_posts, target_h_full, target_z_repre, target_features, target_priors = self.rssm_model(target_temporal_embed_one_rssm,
                                                                                                          prev_action, init_state,
                                                                                                          forward_pred=False)

        T, B, feature_dim = target_features.shape
        # ----- temporal stream ----- #
        # calculate partial pred contrast loss
        online_temporal_embed_one_rssm = online_temporal_embed_one.clone()
        online_posts, online_h_full, online_z_reper, online_features, online_priors = self.rssm_model(online_temporal_embed_one_rssm,
                                                                                                         prev_action, init_state,
                                                                                                         forward_pred=False)
        online_priors_dist = self.rssm_model.zdistr(online_priors)
        online_z_pred = online_priors_dist.rsample().reshape(T, B, -1)
        online_pred_features = torch.cat((online_h_full, online_z_pred), dim=-1)

        # calculate embed to full_feature pred loss and back_propagate to encoder
        pred_features = self.feature_pred_head(online_temporal_embed_one.reshape(-1, self.latent_size))
        target_features = target_features.detach().view(-1, feature_dim)
        feature_pred_loss = self.byol_loss_fn(pred_features, target_features)
        # calculate pred_feature to embed prediction loss and back_propagate to
        pred_embeds = self.image_embed_pred_head(online_pred_features.reshape(-1, feature_dim))
        target_embeds = target_temporal_embed_one.view(-1, self.latent_size)
        embed_pred_loss = self.byol_loss_fn(pred_embeds, target_embeds)
        temporal_loss = feature_pred_loss + embed_pred_loss

        # ----- spatial_temporal stream ----- #
        # feature_dim = target_features.shape[-1]
        # target_features = target_features.detach().view(-1, feature_dim)
        # cross_pred = self.spatial_temporal_predictor(online_spatial_embed_one.reshape(-1, self.latent_size))
        # cross_loss = self.byol_loss_fn(cross_pred, target_features)

        # ----- calculate KL loss ----- #
        dist = self.rssm_model.zdistr
        online_priors = dist(online_priors)
        online_posts = dist(online_posts)
        target_priors = dist(target_priors.detach())
        target_posts = dist(target_posts.detach())
        kl_loss_post = D.kl.kl_divergence(online_posts, target_priors).mean()
        kl_loss_prior = D.kl.kl_divergence(target_posts, online_priors).mean()
        kl_loss = (1 - self.kl_balance) * kl_loss_post + self.kl_balance * kl_loss_prior

        # calculate metrics
        entropy_prior = online_priors.entropy().mean()
        entropy_posts = online_posts.entropy().mean()

        return temporal_loss, kl_loss, entropy_posts, entropy_prior

    def byol_loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1).mean()

    def logavgexp(self, x, dim):
        if x.size(dim) > 1:
            return x.logsumexp(dim=dim) - np.log(x.size(dim))
        else:
            return x.squeeze(dim)

    def validation(self, itr):
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            drone_state_proj=self.drone_state_proj.state_dict(),
            target_drone_state_proj=self.target_drone_state_proj.state_dict(),
            spatial_predictor=self.spatial_predictor.state_dict(),
            image_embed_pred_head=self.image_embed_pred_head.state_dict(),
            feature_pred_head=self.feature_pred_head.state_dict(),
            # spatial_temporal_predictor=self.spatial_temporal_predictor.state_dict(),
            rssm_model=self.rssm_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.drone_state_proj.load_state_dict(state_dict['drone_state_proj'])
        self.target_drone_state_proj.load_state_dict(state_dict['target_drone_state_proj'])
        self.spatial_predictor.load_state_dict(state_dict['spatial_predictor'])
        self.feature_pred_head.load_state_dict(state_dict['feature_pred_head'])
        self.image_embed_pred_head.load_state_dict(state_dict['image_embed_pred_head'])
        # self.spatial_temporal_predictor.load_state_dict(state_dict['spatial_temporal_predictor'])
        self.rssm_model.load_state_dict(state_dict['rssm_model'])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.target_encoder.parameters()
        yield from self.drone_state_proj.parameters()
        yield from self.target_drone_state_proj.parameters()
        yield from self.spatial_predictor.parameters()
        yield from self.image_embed_pred_head.parameters()
        yield from self.feature_pred_head.parameters()
        # yield from self.spatial_temporal_predictor.parameters()
        yield from self.rssm_model.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.target_encoder.named_parameters()
        yield from self.drone_state_proj.named_parameters()
        yield from self.target_drone_state_proj.named_parameters()
        yield from self.spatial_predictor.named_parameters()
        yield from self.image_embed_pred_head.named_parameters()
        yield from self.feature_pred_head.named_parameters()
        # yield from self.spatial_temporal_predictor.named_parameters()
        yield from self.rssm_model.named_parameters()

    def eval(self):
        self.encoder.eval()
        self.target_encoder.eval()
        self.drone_state_proj.eval()
        self.target_drone_state_proj.eval()
        self.spatial_predictor.eval()
        self.feature_pred_head.eval()
        self.image_embed_pred_head.eval()
        # self.spatial_temporal_predictor.eval()
        self.rssm_model.eval()

    def train(self):
        self.encoder.train()
        self.target_encoder.train()
        self.drone_state_proj.train()
        self.target_drone_state_proj.train()
        self.spatial_predictor.train()
        self.feature_pred_head.train()
        self.image_embed_pred_head.train()
        # self.spatial_temporal_predictor.train()
        self.rssm_model.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example

    def wandb_log_code(self):
        wandb.save('./rlpyt/ul/algos/ul_for_rl/mstc.py')
