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
from rlpyt.ul.models.ul.encoders import ResEncoderModel
from rlpyt.ul.models.ul.atc_models import ByolMlpModel, DroneStateProj
from rlpyt.ul.algos.utils.data_augs import get_augmentation, random_shift
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.models.ul.rssm import RSSMCore
import torchvision.transforms as Trans
import torch.nn.functional as F
import torch.distributions as D

IGNORE_INDEX = -100  # Mask TC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["mstcLoss", "temporalLoss", 'klLoss', 'overshot_contrastLoss',
                                 'entropy_posts', 'entropy_priors', 'embedPredAccuracy',
                                 'overshotPredAccuracy', "gradNorm", 'current_lr'])

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
            overshot_horizon=3,
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
            random_shift_pad=4,
            augmentations=('blur', 'intensity'),  # combined with intensity jit accord to SGI
            temporal_coefficient=1.0,
            kl_coefficient=1.0,
            kl_balance=0.5,
            validation_split=0.0,
            n_validation_batches=0,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=ResEncoderModel,
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

        self.embed_pred_head = torch.nn.Linear(self.latent_size, self.rssm_model.stoch_dim, bias=False)

        # self.imagine_pred_head = ByolMlpModel(
        #     input_dim=self.rssm_model.stoch_dim,
        #     latent_size=self.latent_size,
        #     hidden_size=self.hidden_sizes
        # )

        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.drone_state_proj.to(self.device)
        self.target_drone_state_proj.to(self.device)
        self.rssm_model.to(self.device)
        self.embed_pred_head.to(self.device)
        # self.imagine_pred_head.to(self.device)

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
        temporal_loss, full_kl_loss, embed_pred_accuracy, entropy_posts, entropy_priors, overshot_kl_loss, \
            overshot_contrast_loss, overshot_pred_accuracy = self.mstc_loss(samples)

        contrast_loss = temporal_loss + overshot_contrast_loss
        kl_loss = full_kl_loss + overshot_kl_loss
        optimize_loss = self.temporal_coefficient * contrast_loss + self.kl_coefficient * kl_loss
        optimize_loss.backward()

        loss = contrast_loss + kl_loss

        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # log the optimization info/result
        opt_info.mstcLoss.append(loss.item())
        opt_info.temporalLoss.append(temporal_loss.item())
        opt_info.overshot_contrastLoss.append(overshot_contrast_loss.item())
        opt_info.klLoss.append(kl_loss.item())
        opt_info.embedPredAccuracy.append(embed_pred_accuracy.item())
        opt_info.overshotPredAccuracy.append(overshot_pred_accuracy.item())
        opt_info.entropy_posts.append(entropy_posts.item())
        opt_info.entropy_priors.append(entropy_priors.item())
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
            target_temporal_embed_one, _ = self.target_encoder(obs_one)
            target_temporal_embed_two, _ = self.target_encoder(obs_two)
            target_temporal_embed_one = target_temporal_embed_one + self.target_drone_state_proj(cur_drone_state)
            target_temporal_embed_two = target_temporal_embed_two + self.target_drone_state_proj(cur_drone_state)

        online_temporal_embed_one, _ = self.encoder(obs_one)
        online_temporal_embed_two, _ = self.encoder(obs_two)
        online_temporal_embed_one = online_temporal_embed_one + self.drone_state_proj(cur_drone_state)
        online_temporal_embed_two = online_temporal_embed_two + self.drone_state_proj(cur_drone_state)

        temporal_loss, kl_loss, embed_pred_accuracy, entropy_posts, entropy_prior, \
            overshot_kl_loss, overshot_contrast_loss, overshot_pred_accuracy = \
            self.spatial_temporal_loss(online_temporal_embed_one, target_temporal_embed_one,
                                       online_temporal_embed_two, target_temporal_embed_two, prev_action)

        return temporal_loss, kl_loss, embed_pred_accuracy, entropy_posts, entropy_prior, overshot_kl_loss, \
            overshot_contrast_loss, overshot_pred_accuracy

    def spatial_temporal_loss(self, online_temporal_embed_one, target_temporal_embed_one,
                              online_temporal_embed_two, target_temporal_embed_two,
                              prev_action):
        T, B, latent_dim = online_temporal_embed_one.shape
        init_state = self.rssm_model.init_state(self.batch_B)

        # ----- temporal stream ----- #
        # calculate partial pred contrast loss
        online_posts_one, online_h_partial_one, online_z_pred_one, online_features_one, online_priors_one = \
            self.rssm_model(online_temporal_embed_one.clone(), prev_action, init_state, forward_pred=False)
        online_posts_two, online_h_partial_two, online_z_pred_two, online_features_two, online_priors_two = \
            self.rssm_model(online_temporal_embed_two.clone(), prev_action, init_state, forward_pred=False)

        if self.overshot_horizon > 0:
            overshot_kl_loss, overshot_contrast_loss, overshot_pred_accuracy = \
                self.overshot_loss(prev_action, online_h_partial_one.clone(), online_z_pred_one.clone(),
                                   self.overshot_horizon, online_posts_one.clone(), target_temporal_embed_two.clone())
        else:
            overshot_kl_loss = torch.zeros(1)
            overshot_contrast_loss = torch.zeros(1)
            overshot_pred_accuracy = torch.zeros(1)

        online_aux_z_one = self.embed_pred_head(online_temporal_embed_one)

        online_aux_z_two = self.embed_pred_head(online_temporal_embed_two)
        d_aux_z_one = self.rssm_model.zdistr(online_aux_z_one)
        d_aux_z_two = self.rssm_model.zdistr(online_aux_z_two)
        sampled_aux_z_one = d_aux_z_one.rsample().view(-1, self.rssm_model.stoch_dim)
        sampled_aux_z_two = d_aux_z_two.rsample().view(-1, self.rssm_model.stoch_dim)

        online_z_pred_one = online_z_pred_one.view(-1, self.rssm_model.stoch_dim)
        online_z_pred_two = online_z_pred_two.view(-1, self.rssm_model.stoch_dim)

        with torch.no_grad():
            target_temporal_logits_one = self.embed_pred_head(target_temporal_embed_one).view(-1, self.stoch_dim, self.stoch_discrete)
            target_temporal_logits_two = self.embed_pred_head(target_temporal_embed_two).view(-1, self.stoch_dim, self.stoch_discrete)
            target_pmf_one = F.log_softmax(target_temporal_logits_one, dim=-1).view(-1, self.rssm_model.stoch_dim)
            target_pmf_two = F.log_softmax(target_temporal_logits_two, dim=-1).view(-1, self.rssm_model.stoch_dim)

        embed_logits_one = torch.matmul(online_z_pred_two, target_pmf_one.detach().transpose(1, 0))  # ((T-1)*B, (T-1)*B)
        embed_logits_one = embed_logits_one - torch.max(embed_logits_one, dim=1, keepdim=True)[0]
        embed_logits_two = torch.matmul(online_z_pred_one, target_pmf_two.detach().transpose(1, 0))
        embed_logits_two = embed_logits_two - torch.max(embed_logits_two, dim=1, keepdim=True)[0]
        embed_aux_logits_one = torch.matmul(sampled_aux_z_two, target_pmf_one.detach().transpose(1, 0))
        embed_aux_logits_one = embed_aux_logits_one - torch.max(embed_aux_logits_one, dim=1, keepdim=True)[0]
        embed_aux_logits_two = torch.matmul(sampled_aux_z_one, target_pmf_two.detach().transpose(1, 0))
        embed_aux_logits_two = embed_aux_logits_two - torch.max(embed_aux_logits_two, dim=1, keepdim=True)[0]

        # calculate loss
        labels = torch.arange(T * B, dtype=torch.long, device=self.device)
        embed_loss_one = self.c_e_loss(embed_logits_one, labels)
        embed_loss_two = self.c_e_loss(embed_logits_two, labels)
        embed_aux_loss_one = self.c_e_loss(embed_aux_logits_one, labels)
        embed_aux_loss_two = self.c_e_loss(embed_aux_logits_two, labels)
        temporal_loss = 0.5 * (embed_loss_one + embed_loss_two) + 0.5 * (embed_aux_loss_one + embed_aux_loss_two)

        # ----- calculate KL loss ----- #
        dist = self.rssm_model.zdistr
        d_priors = dist(online_priors_one)
        d_posts = dist(online_posts_one)
        if self.kl_balance == 0.5:
            kl_loss = D.kl.kl_divergence(d_posts, d_priors)
        else:
            kl_loss_postgrad = D.kl.kl_divergence(d_posts, dist(online_priors_one.detach())).mean()
            kl_loss_priorgrad = D.kl.kl_divergence(dist(online_posts_one.detach()), d_priors).mean()
            kl_loss = (1 - self.kl_balance) * kl_loss_postgrad + self.kl_balance * kl_loss_priorgrad

        # calculate metrics
        embed_pred_correct_one = torch.argmax(embed_logits_one.detach(), dim=1) == labels
        embed_pred_correct_two = torch.argmax(embed_logits_two.detach(), dim=1) == labels
        embed_pred_correct = torch.cat((embed_pred_correct_one, embed_pred_correct_two), dim=-1)
        embed_pred_accuracy = torch.mean(embed_pred_correct.float())
        entropy_prior = d_priors.entropy().mean()
        entropy_posts = d_posts.entropy().mean()

        return temporal_loss, kl_loss, embed_pred_accuracy, entropy_posts, entropy_prior, \
            overshot_kl_loss, overshot_contrast_loss, overshot_pred_accuracy

    def overshot_loss(self, prev_actions, post_states_h, post_states_z, overshot_horizon, posteriors, target_embeds):
        start_idxs = np.arange(0, self.batch_T - overshot_horizon)
        end_idxs = start_idxs + overshot_horizon
        base_labels = torch.arange(self.batch_T * self.batch_B, dtype=torch.long, device=self.device).view(self.batch_T,
                                                                                                           self.batch_B)
        init_states_h = []
        init_samples_z = []
        sliced_actions = []
        target_posteriors = []
        label_list = []
        for start_idx, end_idx in zip(start_idxs, end_idxs):
            init_states_h.append(post_states_h[start_idx])  # init_states_h list(1, B, h_dim)
            init_samples_z.append(post_states_z[start_idx])  # init_states_z list(1, B, z_dim)
            sliced_actions.append(prev_actions[start_idx + 1:end_idx + 1])
            target_posteriors.append(posteriors[start_idx + 1:end_idx + 1])
            label_list.append(base_labels[start_idx + 1:end_idx + 1].view(-1))

        init_states_h = torch.cat(init_states_h, dim=0)  # init_states_h (B*num_samples, h_dim)
        init_samples_z = torch.cat(init_samples_z, dim=0)  # init_samples_z (B*num_samples, z_dim)
        sliced_actions = torch.cat(sliced_actions, dim=1)
        target_posteriors = torch.cat(target_posteriors, dim=1)
        # calculate forward pred through wm
        states_h, samples, features, priors = self.rssm_model.forward_imagine(sliced_actions,
                                                                              (init_states_h, init_samples_z))
        # overshot_kl_loss
        dist = self.rssm_model.zdistr
        d_target_post = dist(target_posteriors.detach())
        d_overshot_prior = dist(priors)
        overshot_kl_loss = D.kl.kl_divergence(d_target_post, d_overshot_prior).mean()
        # overshot_contrast_loss
        with torch.no_grad():
            target_pred_z_logits = self.embed_pred_head(target_embeds).view(-1, self.stoch_dim, self.stoch_discrete)
            target_pmf = F.log_softmax(target_pred_z_logits, dim=-1).view(-1, self.rssm_model.stoch_dim)
        embed_logits = torch.matmul(samples.view(-1, self.rssm_model.stoch_dim), target_pmf.detach().transpose(1, 0))
        embed_logits = embed_logits - torch.max(embed_logits, dim=-1, keepdim=True)[0]
        labels = torch.cat(label_list)
        overshot_contrast_loss = self.c_e_loss(embed_logits, labels)
        # calculate metrix
        embed_pred_correct = torch.argmax(embed_logits.detach(), dim=1) == labels
        overshot_pred_accuracy = torch.mean(embed_pred_correct.float())
        return overshot_kl_loss, overshot_contrast_loss, overshot_pred_accuracy

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
            spatial_predictor=self.embed_pred_head.state_dict(),
            # imagine_pred_head=self.imagine_pred_head.state_dict(),
            # spatial_temporal_predictor=self.spatial_temporal_predictor.state_dict(),
            rssm_model=self.rssm_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.drone_state_proj.load_state_dict(state_dict['drone_state_proj'])
        self.target_drone_state_proj.load_state_dict(state_dict['target_drone_state_proj'])
        self.embed_pred_head.load_state_dict(state_dict['spatial_predictor'])
        # self.imagine_pred_head.load_state_dict(state_dict['imagine_pred_head'])
        # self.spatial_temporal_predictor.load_state_dict(state_dict['spatial_temporal_predictor'])
        self.rssm_model.load_state_dict(state_dict['rssm_model'])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.target_encoder.parameters()
        yield from self.drone_state_proj.parameters()
        yield from self.target_drone_state_proj.parameters()
        yield from self.embed_pred_head.parameters()
        # yield from self.imagine_pred_head.parameters()
        # yield from self.spatial_temporal_predictor.parameters()
        yield from self.rssm_model.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.target_encoder.named_parameters()
        yield from self.drone_state_proj.named_parameters()
        yield from self.target_drone_state_proj.named_parameters()
        yield from self.embed_pred_head.named_parameters()
        # yield from self.imagine_pred_head.named_parameters()
        # yield from self.spatial_temporal_predictor.named_parameters()
        yield from self.rssm_model.named_parameters()

    def eval(self):
        self.encoder.eval()
        self.target_encoder.eval()
        self.drone_state_proj.eval()
        self.target_drone_state_proj.eval()
        # self.imagine_pred_head.eval()
        # self.spatial_temporal_predictor.eval()
        self.rssm_model.eval()
        self.embed_pred_head.eval()

    def train(self):
        self.encoder.train()
        self.target_encoder.train()
        self.drone_state_proj.train()
        self.target_drone_state_proj.train()
        # self.imagine_pred_head.train()
        # self.spatial_temporal_predictor.train()
        self.rssm_model.train()
        self.embed_pred_head.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example

    def wandb_log_code(self):
        wandb.save('./rlpyt/ul/algos/ul_for_rl/mstc.py')
