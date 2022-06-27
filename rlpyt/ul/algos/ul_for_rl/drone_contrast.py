import torch
from collections import namedtuple
import copy
import wandb
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.encoders import ResEncoderModel
from rlpyt.ul.models.ul.atc_models import ContrastModel
from rlpyt.ul.algos.utils.data_augs import random_shift
from rlpyt.ul.replays.offline_dataset import OfflineDatasets

IGNORE_INDEX = -100  # Mask TC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["atcLoss", "accuracy", "gradNorm"])
ValInfo = namedtuple("ValInfo", ["atcLoss", "accuracy", "convActivation"])


class DroneContrast(BaseUlAlgorithm):
    """Contrastive loss against one future time step, using a momentum encoder
	for the target."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            delta_T=3,  # delta_T is the forward step
            batch_T=1,  # the default length of extracted batch
            batch_B=512,  # batch B is the sampled batch size for extraction
            clip_grad_norm=10.,
            learning_rate=1e-3,
            num_stacked_input=1,
            target_update_tau=0.01,  # 1 for hard update
            target_update_interval=1,
            latent_size=256,
            hidden_sizes=512,
            random_shift_prob=1.,
            random_shift_pad=4,
            validation_split=0.0,
            n_validation_batches=0,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=ResEncoderModel,
            ContrastCls=ContrastModel,
            initial_state_dict=None,
            optim_kwargs=None,
            sched_kwargs=None,
            encoder_kwargs=None,
            replay_kwargs=None,
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
        print('total number of itrs is:', self.n_updates)
        self.image_shape = image_shape = examples.observation.shape  # [c, h, w]

        self.encoder = self.EncoderCls(
            image_shape=image_shape,
            latent_size=self.latent_size,
            hidden_sizes=self.hidden_sizes,
            num_stacked_input=self.num_stacked_input,
            **self.encoder_kwargs
        )
        self.target_encoder = copy.deepcopy(self.encoder)  # the target encoder is not tied with online encoder

        self.contrast = self.ContrastCls(
            latent_size=self.latent_size,
            anchor_hidden_sizes=self.hidden_sizes,
        )
        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.contrast.to(self.device)

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
        atc_loss, accuracy, conv_output = self.atc_loss(samples)
        atc_loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # log the optimize info/result
        opt_info.atcLoss.append(atc_loss.item())
        opt_info.accuracy.append(accuracy.item())
        opt_info.gradNorm.append(grad_norm.item())

        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(), self.target_update_tau)
        return opt_info

    def atc_loss(self, samples):
        # the dim of T for samples.observation is 1+delta_T
        # TODO the postive and anchor is just for one forward step, and needed be changed for multi-step
        anchor = (samples.observations if self.delta_T == 0 else samples.observations[:-self.delta_T])[0]
        positive = samples.observations[self.delta_T:][0]
        b, f, c, h, w = anchor.shape
        anchor = anchor.view(b * f, c, h, w)  # Treat all T,B as separate.(reshape the sample)
        positive = positive.view(b * f, c, h, w)

        if self.random_shift_prob > 0.:
            anchor = random_shift(
                imgs=anchor,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )
            positive = random_shift(
                imgs=positive,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )

        # sned a tuple of tensor (anchor, postive) to device and return a tuple
        anchor, positive = buffer_to((anchor, positive), device=self.device)

        with torch.no_grad():
            c_positive, _ = self.target_encoder(positive)
            c_positive.detach_()

        c_anchor, conv_output = self.encoder(anchor)

        logits = self.contrast(anchor=c_anchor, positive=c_positive.detach())

        labels = torch.arange(c_anchor.shape[0], dtype=torch.long, device=self.device)
        # valid = valid_from_done(samples.done).type(torch.bool) # length of sampels.done is T+1 and if done, valid is false
        # valid = valid[self.delta_T:].reshape(-1)  # at location of positive
        # labels[~valid] = IGNORE_INDEX # idx by bool type and mask the invalued sample by making the lable==-100
        atc_loss = self.c_e_loss(logits, labels)

        correct = torch.argmax(logits.detach(), dim=1) == labels
        # accuracy = torch.mean(correct[valid].float())
        accuracy = torch.mean(correct.float())

        return atc_loss, accuracy, conv_output

    def validation(self, itr):
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            contrast=self.contrast.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.contrast.load_state_dict(state_dict["contrast"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.contrast.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.contrast.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.contrast.eval()

    def train(self):
        self.encoder.train()
        self.contrast.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example

    def wandb_log_code(self):
        wandb.save('./rlpyt/ul/algos/ul_for_rl/drone_contrast.py')
