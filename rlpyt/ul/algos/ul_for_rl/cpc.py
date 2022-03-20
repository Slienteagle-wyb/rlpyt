import torch
import numpy as np
from collections import namedtuple
from rlpyt.utils.tensor import valid_mean
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import to_onehot
from rlpyt.ul.models.ul.encoders import DmlabEncoderModel
from rlpyt.ul.replays.offline_dataset import OfflineDatasets


IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["cpcLoss", "cpcAccuracy1", "cpcAccuracy2", "cpcAccuracyTm1", "cpcAccuracyTm2",
                                 "gradNorm"])
ValInfo = namedtuple("ValInfo", ["cpcLoss", "cpcAccuracy1", "cpcAccuracy2",
                                 "cpcAccuracyTm1", "cpcAccuracyTm2", "convActivation"])


class CPC(BaseUlAlgorithm):
    """Contrastive Predictive Coding with recurrent network."""
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_B,
            batch_T,
            warmup_T=0,
            learning_rate=5e-4,
            rnn_size=256,
            latent_size=256,
            clip_grad_norm=1000.,
            onehot_actions=False,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            OptimCls=torch.optim.Adam,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=DmlabEncoderModel,
            validation_split=0.0,
            optim_kwargs=None,
            initial_state_dict=None,
            encoder_kwargs=None,
            replay_kwargs=None,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        self.batch_size = batch_B * batch_T  # for logging only
        self._replay_T = batch_T + warmup_T

    def initialize(self, n_updates, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)
        examples = self.load_replay()
        self.encoder = self.EncoderCls(
            image_shape=examples.observation.shape,
            latent_size=self.latent_size,
            **self.encoder_kwargs
        )
        self.encoder.to(self.device)

        if self.onehot_actions:
            max_act = self.replay_buffer.samples.action.max()
            self._act_dim = max_act + 1  # To use for 1-hot encoding
            ar_input_size = self._act_dim + 1  # for 1 step, + 1 reward
        else:
            assert len(self.replay_buffer.samples.translation.shape) == 3  # [T,B,A]
            trans_dim = self.replay_buffer.translation_dim
            rotate_dim = self.replay_buffer.rotation_dim
            ar_input_size = rotate_dim + trans_dim  # no reward

        self.prediction_rnn = torch.nn.LSTM(
            input_size=int(self.latent_size + ar_input_size),
            hidden_size=self.rnn_size,
        )
        self.prediction_rnn.to(self.device)

        transforms = [None]
        for _ in range(self.batch_T - 1):
            transforms.append(
                torch.nn.Linear(in_features=self.rnn_size, out_features=self.latent_size)
            )
        self.transforms = torch.nn.ModuleList(transforms)
        self.transforms.to(self.device)

        self.optim_initialize(n_updates)

        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_B)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        cpc_loss, cpc_accuracies, conv_output = self.cpc_loss(samples)

        cpc_loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            # return the norm of parameters` gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.cpcLoss.append(cpc_loss.item())
        opt_info.cpcAccuracy1.append(cpc_accuracies[0].item())
        opt_info.cpcAccuracy2.append(cpc_accuracies[1].item())
        opt_info.cpcAccuracyTm1.append(cpc_accuracies[2].item())
        opt_info.cpcAccuracyTm2.append(cpc_accuracies[3].item())
        opt_info.gradNorm.append(grad_norm.item())
        return opt_info

    def cpc_loss(self, samples):
        observation = samples.observations  # shape of observation [batch_T, batch_B, frame_stack, 3, 84, 84]
        length, b, f, c, h, w = observation.shape
        observation = observation.view(length, b*f, c, h, w)
        translation = samples.translations
        rotation = samples.rotations
        action = torch.cat((translation, rotation), dim=-1)
        if self.onehot_actions:
            action = to_onehot(action, self._act_dim, dtype=torch.float)
        observation, action = buffer_to((observation, action), device=self.device)
        # encoder all the observation into latent space and the latent variable is passed by a nolinear projector
        z_latent, conv_output = self.encoder(observation)  # [T,B,z_dim]
        rnn_input = torch.cat([z_latent, action], dim=-1)  # [T,B,z_dim+act_dim]

        context, _ = self.prediction_rnn(rnn_input)

        # Extract only the ones to train (all were needed to compute).
        z_latent = z_latent[self.warmup_T:]  # warmup for lstm
        conv_output = conv_output[self.warmup_T:]
        context = context[self.warmup_T:]
        ###############################
        # Contrast the network outputs:
        # Should have T,B,C=context.shape, T,B=valid.shape, T,B,Z=z_latent.shape
        T, B, Z = z_latent.shape
        target_trans = z_latent.view(-1, Z).transpose(1, 0)  # [T,B,z]->[T*B,z]->[z,T*B]
        # Draw from base_labels according to the location of the corresponding
        # positive latent for contrast, using [T,B]; will give the location
        # within T*B.
        base_labels = torch.arange(T * B, dtype=torch.long, device=self.device).view(T, B)

        # All predictions and labels into one tensor for efficient contrasting.
        prediction_list = list()
        label_list = list()
        for delta_t in range(1, T):
            # Predictions based on context starting from t=0 up to the point where
            # there isn't a future latent within the timesteps of the minibatch.
            # [T-delta_t,B,c_dim] -> [T-delta_t,B,z_dim] -> [(T-delta_t)*B,z_dim]
            # context is chosen from 0:T-delta_T and the target is
            prediction_list.append(self.transforms[delta_t](context[:-delta_t]).view(-1, Z))
            # The correct latent is delta_t time steps ahead:
            # [T-delta_t,B] -> [(T-delta_t)*B]
            label_list.append(base_labels[delta_t:].view(-1))

        # Before cat, to isolate delta_t for diagnostic accuracy check later:
        dt_lengths = [0] + [len(label) for label in label_list]
        dtb = torch.cumsum(torch.tensor(dt_lengths), dim=0)  # delta_t_boundaries

        # Total number of predictions: P = T*(T-1)/2*B
        predictions = torch.cat(prediction_list)  # [P,z_dim]
        labels = torch.cat(label_list)  # [P]
        # contrast against ALL latents, not just the "future" ones:
        logits = torch.matmul(predictions, target_trans)  # [P,H]*[H,T*B] -> [P,T*B]
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # [P,T*B] normalize
        cpc_loss = self.c_e_loss(logits, labels)  # every logit weighted equally

        ##################################################
        # Compute some downsampled accuracies for diagnostics:

        logits_d = logits.detach()
        # begin, end, step (downsample):
        b, e, s = dtb[0], dtb[1], 4  # delta_t = 1
        logits1, labels1 = logits_d[b:e:s], labels[b:e:s]
        correct1 = torch.argmax(logits1, dim=1) == labels1
        accuracy1 = valid_mean(correct1.float(), valid=labels1 >= 0)  # IGNORE=-100

        b, e, s = dtb[1], dtb[2], 4  # delta_t = 2
        logits2, labels2 = logits_d[b:e:s], labels[b:e:s]
        correct2 = torch.argmax(logits2, dim=1) == labels2
        accuracy2 = valid_mean(correct2.float(), valid=labels2 >= 0)

        b, e, s = dtb[-2], dtb[-1], 1  # delta_t = T - 1
        logitsT1, labelsT1 = logits_d[b:e:s], labels[b:e:s]
        correctT1 = torch.argmax(logitsT1, dim=1) == labelsT1
        accuracyT1 = valid_mean(correctT1.float(), valid=labelsT1 >= 0)

        b, e, s = dtb[-3], dtb[-2], 1  # delta_t = T - 2
        logitsT2, labelsT2 = logits_d[b:e:s], labels[b:e:s]
        correctT2 = torch.argmax(logitsT2, dim=1) == labelsT2
        accuracyT2 = valid_mean(correctT2.float(), valid=labelsT2 >= 0)

        accuracies = (accuracy1, accuracy2, accuracyT1, accuracyT2)

        return cpc_loss, accuracies, conv_output

    def validation(self, itr):
        # logger.log("Computing validation loss...")
        # val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        # self.optimizer.zero_grad()
        # for _ in range(self.n_validation_batches):
        #     samples = self.replay_buffer.sample_batch(self.validation_batch_B,
        #         validation=True)
        #     with torch.no_grad():
        #         cpc_loss, cpc_accuracies, conv_output = self.cpc_loss(samples)
        #     val_info.cpcLoss.append(cpc_loss.item())
        #     val_info.cpcAccuracy1.append(cpc_accuracies[0].item())
        #     val_info.cpcAccuracy2.append(cpc_accuracies[1].item())
        #     val_info.cpcAccuracyTm1.append(cpc_accuracies[2].item())
        #     val_info.cpcAccuracyTm2.append(cpc_accuracies[3].item())
        #     val_info.convActivation.append(
        #         conv_output[0, 0].detach().cpu().view(-1).numpy())
        # self.optimizer.zero_grad()
        # logger.log("...validation loss completed.")
        # return val_info
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            prediction_rnn=self.prediction_rnn.state_dict(),
            transforms=self.transforms.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.prediction_rnn.load_state_dict(state_dict["prediction_rnn"])
        self.transforms.load_state_dict(state_dict["transforms"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.prediction_rnn.parameters()
        yield from self.transforms.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.prediction_rnn.named_parameters()
        yield from self.transforms.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.prediction_rnn.eval()
        self.transforms.eval()

    def train(self):
        self.encoder.train()
        self.prediction_rnn.train()
        self.transforms.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example