import torch
import torch.nn.functional as F
from collections import namedtuple
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.ul.models.ul.vae_models import VaeHeadModel, VaeDecoderModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import valid_mean
from rlpyt.ul.models.ul.encoders import DmlabEncoderModelNorm
from rlpyt.ul.models.ul.forward_models import SkipConnectForwardAggModel
from rlpyt.ul.replays.offline_dataset import OfflineDatasets

IGNORE_INDEX = -100  # Mask action samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["reconLoss", "klLoss", "gradNorm"])
ValInfo = namedtuple("ValInfo", ["reconLoss", "klLoss"])


class VAE(BaseUlAlgorithm):
    """VAE to predict o_t+k from o_t."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_T=32,
            batch_B=16,
            warmup_T=16,
            clip_grad_norm=100.,
            latent_size=256,
            hidden_sizes=512,  # But maybe use for forward prediction
            validation_split=0.0,
            n_validation_batches=0,
            with_validation=True,
            kl_coefficient=1.0,
            TrainReplayCls=OfflineUlReplayBuffer,
            ValReplayCls=OfflineUlReplayBuffer,
            EncoderCls=DmlabEncoderModelNorm,
            VaeHeadCls=VaeHeadModel,
            DecoderCls=VaeDecoderModel,
            decoder_kwargs=None,
            optim_kwargs=None,
            sched_kwargs=None,
            encoder_kwargs=None,
            initial_state_dict=None,
            train_replay_kwargs=None,
            val_replay_kwargs=None,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.batch_size = self.batch_B * self.batch_T

    def initialize(self, epochs, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)

        examples = self.load_replay(with_validation=self.with_validation)
        trans_dim = self.train_buffer.translation_dim
        rotate_dim = self.train_buffer.rotation_dim
        forward_input_size = rotate_dim + trans_dim  # no reward

        self.itrs_per_epoch = self.train_buffer.size // self.batch_size
        self.n_updates = epochs * self.itrs_per_epoch
        print('{0} iters per epoch, {1} epochs , {2} n updates', self.itrs_per_epoch, epochs, self.n_updates)

        self.encoder = self.EncoderCls(
            image_shape=examples.observation.shape,
            latent_size=self.latent_size,  # UNUSED
            hidden_sizes=self.hidden_sizes,
            **self.encoder_kwargs
        )

        self.forward_agg_rnn = SkipConnectForwardAggModel(
            input_size=int(self.latent_size + forward_input_size),
            hidden_sizes=self.hidden_sizes,
            latent_size=self.latent_size
        )

        self.forward_pred_rnn = SkipConnectForwardAggModel(
            input_size=int(forward_input_size),
            hidden_sizes=self.hidden_sizes,
            latent_size=self.latent_size
        )

        self.vae_head = self.VaeHeadCls(
            latent_size=self.latent_size,
            action_size=0,
            hidden_sizes=self.hidden_sizes,
        )

        self.decoder = self.DecoderCls(
            latent_size=self.latent_size,
            **self.decoder_kwargs
        )

        self.optim_initialize(epochs)

        if self.initial_state_dict is not None:
            logger.log('models loading state dict ....')
            loaded_state_dict = torch.load(self.initial_state_dict,
                                           map_location=torch.device('cpu'))
            loaded_state_dict = loaded_state_dict.get('algo_state_dict', loaded_state_dict)
            loaded_encoder_dict = loaded_state_dict.get('encoder', loaded_state_dict)
            loaded_forward_agg_dict = loaded_state_dict.get('forward_agg_rnn', loaded_state_dict)
            loaded_forward_pred_dict = loaded_state_dict.get('forward_pred_rnn', loaded_state_dict)
            self.encoder.load_state_dict(loaded_encoder_dict)
            self.forward_agg_rnn.load_state_dict(loaded_forward_agg_dict)
            self.forward_pred_rnn.load_state_dict(loaded_forward_pred_dict)
            logger.log('finished loaded all the relevant pretrained model.....')

        self.encoder.to(self.device)
        self.forward_agg_rnn.to(self.device)
        self.forward_pred_rnn.to(self.device)
        self.vae_head.to(self.device)
        self.decoder.to(self.device)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.train_buffer.sample_batch(self.batch_size)
        current_epoch = itr // self.itrs_per_epoch
        if self.lr_scheduler is not None and itr % self.itrs_per_epoch == 0:
            self.lr_scheduler.step(current_epoch)
        self.optimizer.zero_grad()

        recon_loss, kl_loss, conv_output = self.vae_loss(samples)
        loss = recon_loss + kl_loss * self.kl_coefficient
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)

        self.optimizer.step()

        opt_info.reconLoss.append(recon_loss.item())
        opt_info.klLoss.append(kl_loss.item())
        opt_info.gradNorm.append(grad_norm.item())
        return opt_info

    def vae_loss(self, samples):
        observation = samples.observations[:self.warmup_T]
        target_observation = samples.observations[self.warmup_T + 1:]
        length_o, b, f, c, h, w = observation.shape
        observation = observation.reshape(length_o, b * f, c, h, w)
        length_t, b, f, c, h, w = target_observation.shape
        target_observation = target_observation.reshape(length_t, b * f, c, h, w)
        prev_translation = samples.prev_translations
        prev_rotation = samples.prev_rotations
        current_translation = samples.translations
        current_rotation = samples.rotations
        prev_action = torch.cat((prev_translation, prev_rotation), dim=-1)[:self.warmup_T]
        current_action = torch.cat((current_translation, current_rotation), dim=-1)[self.warmup_T:]

        observation, target_observation, prev_action, current_action = buffer_to((observation, target_observation,
                                                                                 prev_action, current_action),
                                                                                 device=self.device)

        t, b, c, h, w = target_observation.shape
        with torch.no_grad():
            h, conv_out = self.encoder(observation)  # observation [batch_T/2, batch_B, *image_shape]
            agg_input = torch.cat((h, prev_action), dim=-1)
            _, hidden_states = self.forward_agg_rnn(agg_input)
            pred_next_states, _ = self.forward_pred_rnn(current_action, init_hiddens=hidden_states[-1])

        pred_next_states = pred_next_states[:-1]
        pred_next_states = pred_next_states.reshape(t*b, -1).detach()

        z, mu, logvar = self.vae_head(pred_next_states)
        recon_z = self.decoder(z)  # recon_z [t*b, c, h, w]
        if target_observation.dtype == torch.uint8:
            target_observation = target_observation.type(torch.float)
            target_observation = target_observation.mul_(1 / 255.)

        recon_losses = F.binary_cross_entropy(
            input=recon_z.reshape(t * b * c, h, w),
            target=target_observation.reshape(t * b * c, h, w),
            reduction="none",
        )

        recon_losses = recon_losses.view(b, c, h, w).sum(dim=(2, 3))  # sum over H,W
        recon_losses = recon_losses.mean(dim=1)  # mean over C (o/w loss is HUGE)
        recon_loss = valid_mean(recon_losses, valid=None)  # mean over batch

        kl_losses = 1 + logvar - mu.pow(2) - logvar.exp()
        kl_losses = kl_losses.sum(dim=-1)  # sum over latent dimension
        kl_loss = -0.5 * valid_mean(kl_losses, valid=None)  # mean over batch

        return recon_loss, kl_loss, conv_out

    def validation(self, itr):
        logger.log("Computing validation loss...")
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.itrs_per_epoch):
            samples = self.val_buffer.sample_batch(self.batch_size)
            with torch.no_grad():
                recon_loss, kl_loss, conv_output = self.vae_loss(samples)
            val_info.reconLoss.append(recon_loss.item())
            val_info.klLoss.append(kl_loss.item())
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            forward_agg_rnn=self.forward_agg_rnn.state_dict(),
            forward_pred_rnn=self.forward_pred_rnn.state_dict(),
            vae_head=self.vae_head.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.forward_agg_rnn.load_state_dict(state_dict['forward_agg_rnn'])
        self.forward_pred_rnn.load_state_dict(state_dict['forward_pred_rnn'])
        self.vae_head.load_state_dict(state_dict["vae_head"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.forward_pred_rnn.parameters()
        yield from self.forward_agg_rnn.parameters()
        yield from self.vae_head.parameters()
        yield from self.decoder.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.forward_agg_rnn.named_parameters()
        yield from self.forward_pred_rnn.named_parameters()
        yield from self.vae_head.named_parameters()
        yield from self.decoder.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.forward_agg_rnn.eval()
        self.forward_pred_rnn.eval()
        self.vae_head.eval()
        self.decoder.eval()

    def train(self):
        self.encoder.train()
        self.forward_agg_rnn.train()
        self.forward_pred_rnn.train()
        self.vae_head.train()
        self.decoder.train()

    def load_replay(self, with_validation=True):
        logger.log('Loading train replay buffer ...')
        self.train_buffer = self.TrainReplayCls(OfflineDatasets, **self.train_replay_kwargs)
        if with_validation is True:
            logger.log('loading validation replay buffer ...')
            self.val_buffer = self.ValReplayCls(OfflineDatasets, **self.val_replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.train_buffer.get_example()
        return example
