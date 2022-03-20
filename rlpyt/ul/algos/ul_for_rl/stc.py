import torch
from collections import namedtuple
import copy
from rlpyt.utils.tensor import valid_mean
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.encoders import DmlabEncoderModel
from rlpyt.ul.algos.utils.data_augs import random_shift
from rlpyt.ul.replays.offline_dataset import OfflineDatasets

IGNORE_INDEX = -100  # Mask TC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["stcLoss", "sprLoss", "contrastLoss",
                                 "cpcAccuracy1", "cpcAccuracy2", "cpcAccuracyTm1", "cpcAccuracyTm2",
                                 'contrast_accuracy', "gradNorm", 'current_lr'])
ValInfo = namedtuple("ValInfo", ["atcLoss", "accuracy", "convActivation"])


class DroneSTC(BaseUlAlgorithm):
    """
    Spatial and Temporal Contrastive Pretraining with  ConvGru features
    """
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_T=32,  # the default length of extracted batch
            batch_B=16,  # batch B is the sampled batch size for extraction
            warmup_T=0,
            clip_grad_norm=10.,
            target_update_tau=0.01,  # 1 for hard update
            target_update_interval=1,
            latent_size=256,
            hidden_sizes=512,
            rnn_size=256,
            random_shift_prob=1.,
            random_shift_pad=4,
            spr_loss_coefficient=1.0,
            contrast_loss_coefficient=1.0,
            validation_split=0.0,
            n_validation_batches=0,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=DmlabEncoderModel,
            initial_state_dict=None,
            optim_kwargs=None,
            sched_kwargs=None,
            encoder_kwargs=None,
            replay_kwargs=None,
    ):
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self.batch_size = batch_B * batch_T  # batch_size for calculating update params
        self._replay_T = warmup_T + batch_T  # self.replay_T == self._replay_T is the len of every sampled trajectory


    def initialize(self, epochs, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)

        examples = self.load_replay()
        self.itrs_per_epoch = self.replay_buffer.size // self.batch_size
        self.n_updates = epochs * self.itrs_per_epoch
        self.image_shape = image_shape = examples.observation.shape  # [c, h, w]

        self.encoder = self.EncoderCls(
            image_shape=image_shape,
            latent_size=self.latent_size,
            hidden_sizes=self.hidden_sizes,
            **self.encoder_kwargs
        )
        self.target_encoder = copy.deepcopy(self.encoder)  # the target encoder is not tied with online encoder

        trans_dim = self.replay_buffer.translation_dim
        rotate_dim = self.replay_buffer.rotation_dim
        ar_input_size = rotate_dim + trans_dim  # no reward

        self.aggregator_rnn = torch.nn.LSTM(
            input_size=int(self.latent_size+ar_input_size),
            hidden_size=self.rnn_size,
        )

        # linear transforms from one step prediction to T-1 forward steps
        transforms = [None]
        for _ in range(self.batch_T - 1):
            transforms.append(
                torch.nn.Linear(in_features=self.latent_size, out_features=self.latent_size, bias=False)
            )
        self.transforms = torch.nn.ModuleList(transforms)
        self.byol_linear_trans = torch.nn.Linear(self.latent_size, self.latent_size, bias=False)


        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.aggregator_rnn.to(self.device)
        self.transforms.to(self.device)
        self.byol_linear_trans.to(self.device)

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
        loss, spr_loss, contrast_loss, pred_accuracies, contrast_accuracy = self.stc_loss(samples)
        optimize_loss = spr_loss * self.spr_loss_coefficient + contrast_loss * self.contrast_loss_coefficient
        optimize_loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # log the optimize info/result
        opt_info.stcLoss.append(loss.item())
        opt_info.sprLoss.append(spr_loss.item())
        opt_info.contrastLoss.append(contrast_loss.item())
        opt_info.cpcAccuracy1.append(pred_accuracies[0].item())
        opt_info.cpcAccuracy2.append(pred_accuracies[1].item())
        opt_info.cpcAccuracyTm1.append(pred_accuracies[2].item())
        opt_info.cpcAccuracyTm2.append(pred_accuracies[3].item())
        opt_info.contrast_accuracy.append(contrast_accuracy.item())
        opt_info.current_lr.append(current_lr)
        opt_info.gradNorm.append(grad_norm.item())
        # opt_info.convActivation.append(
        #     conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        # the update interval for the momentun encoder
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder,
                              self.encoder.state_dict(),
                              self.target_update_tau)
        return opt_info

    def stc_loss(self, samples):
        # the dim of T for samples.observation is 1+delta_T
        # TODO the postive and anchor is just for one forward step, and needed be changed for multi-step
        anchor = samples.observations
        length, b, f, c, h, w = anchor.shape
        anchor = anchor.view(length, b * f, c, h, w)  # Treat all T,B as separate.(reshape the sample)
        positive = copy.deepcopy(anchor)
        translation = samples.translations
        rotation = samples.rotations
        action = torch.cat((translation, rotation), dim=-1)

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
        anchor, positive, action = buffer_to((anchor, positive, action), device=self.device)

        with torch.no_grad():
            z_positive, z_conv_positive = self.target_encoder(positive)
            # shape of c_positive [batch_T, batch_B, latent_dim]

        z_anchor, z_conv_anchor = self.encoder(anchor)

        spr_loss, pred_accuracies = self.spr_loss(z_anchor, z_positive, action)
        contrast_loss, contrast_accuracy = self.contrast_loss(z_anchor, z_positive)
        loss = spr_loss  + contrast_loss

        return loss, spr_loss, contrast_loss, pred_accuracies, contrast_accuracy

    def spr_loss(self, z_anchor, z_positive, action):
        rnn_input = torch.cat((z_anchor, action), dim=-1)
        c_anchor, _ = self.aggregator_rnn(rnn_input)
        # Extract only the ones to train (all were needed to compute).
        c_anchor = c_anchor[self.warmup_T:]
        z_positive = z_positive.detach()[self.warmup_T:]

        T, B, Z = z_positive.shape
        target_trans = z_positive.view(-1, Z).transpose(1, 0)
        base_labels = torch.arange(T * B, dtype=torch.long, device=self.device).view(T, B)
        prediction_list = list()
        label_list = list()
        # from 1 forward step to T-1 forward step
        for delta_t in range(1, T):
            prediction_list.append(self.transforms[delta_t](c_anchor[:-delta_t].view(-1, Z)))
            label_list.append(base_labels[delta_t:].view(-1))

        forward_steps = [0] + [len(label) for label in label_list]
        cumulative_forward_steps = torch.cumsum(torch.tensor(forward_steps), dim=0)

        # total predictions pairs: P=T*(T-1)/2*B
        prdictions = torch.cat(prediction_list)  # [P, z_dim]
        labels = torch.cat(label_list)  # [P]

        logits = torch.matmul(prdictions, target_trans)  # [P, z_dim] * [z_dim, T*B]
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        spr_loss = self.c_e_loss(logits, labels)

        ################################################
        # calculate the prediction metrix
        logits_d = logits.detach()
        # begin, end, step (downsample==sample from one step forward predictions every 4 steps):
        b, e, s = cumulative_forward_steps[0], cumulative_forward_steps[1], 4  # delta_t = 1
        logits1, labels1 = logits_d[b:e:s], labels[b:e:s]
        correct1 = torch.argmax(logits1, dim=1) == labels1
        accuracy1 = valid_mean(correct1.float(), valid=labels1 >= 0)  # IGNORE=-100

        b, e, s = cumulative_forward_steps[1], cumulative_forward_steps[2], 4  # delta_t = 2
        logits2, labels2 = logits_d[b:e:s], labels[b:e:s]
        correct2 = torch.argmax(logits2, dim=1) == labels2
        accuracy2 = valid_mean(correct2.float(), valid=labels2 >= 0)

        b, e, s = cumulative_forward_steps[-2], cumulative_forward_steps[-1], 1  # delta_t = T - 1
        logitsT1, labelsT1 = logits_d[b:e:s], labels[b:e:s]
        correctT1 = torch.argmax(logitsT1, dim=1) == labelsT1
        accuracyT1 = valid_mean(correctT1.float(), valid=labelsT1 >= 0)

        b, e, s = cumulative_forward_steps[-3], cumulative_forward_steps[-2], 1  # delta_t = T - 2
        logitsT2, labelsT2 = logits_d[b:e:s], labels[b:e:s]
        correctT2 = torch.argmax(logitsT2, dim=1) == labelsT2
        accuracyT2 = valid_mean(correctT2.float(), valid=labelsT2 >= 0)
        accuracies = (accuracy1, accuracy2, accuracyT1, accuracyT2)
        return spr_loss, accuracies

    def contrast_loss(self, z_anchor, z_positive):
        T, B, latent_dim = z_anchor.shape
        z_positive_trans = z_positive.detach().view(-1, latent_dim).transpose(1, 0)  # [latent_dim, T*B]
        labels = torch.arange(T * B, dtype=torch.long, device=self.device)
        pred = self.byol_linear_trans(z_anchor).view(-1, latent_dim)
        logits = torch.matmul(pred, z_positive_trans)  # [T*B, T*B]
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        contrast_loss = self.c_e_loss(logits, labels)

        correct = torch.argmax(logits.detach(), dim=1) == labels
        contrast_accuracy = torch.mean(correct.float())

        return contrast_loss, contrast_accuracy

    def validation(self, itr):
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            aggregator_rnn=self.aggregator_rnn.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.aggregator_rnn.load_state_dict(state_dict['aggregator_rnn'])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.aggregator_rnn.parameters()
        yield from self.transforms.parameters()
        yield from self.byol_linear_trans.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.aggregator_rnn.named_parameters()
        yield from self.transforms.named_parameters()
        yield from self.byol_linear_trans.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.aggregator_rnn.eval()
        self.transforms.eval()
        self.byol_linear_trans.eval()

    def train(self):
        self.encoder.train()
        self.aggregator_rnn.train()
        self.transforms.train()
        self.byol_linear_trans.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example
