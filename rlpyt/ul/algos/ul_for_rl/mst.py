import torch
from collections import namedtuple
import copy
from rlpyt.utils.tensor import valid_mean
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import to_onehot
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.encoders import DmlabEncoderModel
from rlpyt.ul.algos.utils.data_augs import random_shift
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.models.ul.inverse_models import InverseModelHead
from rlpyt.ul.models.ul.forward_models import ForwardAggRnnModel

IGNORE_INDEX = -100  # Mask TC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["mstLoss", "sprLoss", "contrastLoss", 'inverseDynaLoss',
                                 "cpcAccuracy1", "cpcAccuracy2", "cpcAccuracyTm1", "cpcAccuracyTm2",
                                 'contrast_accuracy', 'inverse_pred_accuracy', "gradNorm"])
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
            learning_rate=1e-3,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            clip_grad_norm=10.,
            target_update_tau=0.01,  # 1 for hard update
            target_update_interval=1,
            latent_size=256,
            hidden_sizes=512,
            random_shift_prob=1.,
            random_shift_pad=4,
            spr_loss_coefficient=1.0,
            contrast_loss_coefficient=1.0,
            inverse_dyna_loss_coefficient=1.0,
            validation_split=0.0,
            n_validation_batches=0,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=DmlabEncoderModel,
            OptimCls=torch.optim.Adam,
            initial_state_dict=None,
            optim_kwargs=None,
            encoder_kwargs=None,
            replay_kwargs=None,
    ):
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]

        self.batch_size = batch_B * batch_T  # for logging only
        self._replay_T = warmup_T + batch_T  # self.replay_T == self._replay_T is the len of every sampled trajectory

    def initialize(self, n_updates, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)

        examples = self.load_replay()
        self.image_shape = image_shape = examples.observation.shape  # [c, h, w]
        trans_dim = self.replay_buffer.translation_dim
        rotate_dim = self.replay_buffer.rotation_dim
        command_dim = self.replay_buffer.command_dim
        forward_input_size = rotate_dim + trans_dim  # no reward
        inverse_pred_dim = command_dim

        self.encoder = self.EncoderCls(
            image_shape=image_shape,
            latent_size=self.latent_size,
            hidden_sizes=self.hidden_sizes,
            **self.encoder_kwargs
        )
        self.target_encoder = copy.deepcopy(self.encoder)  # the target encoder is not tied with online encoder

        self.forward_agg_rnn = ForwardAggRnnModel(
            input_size=int(self.latent_size + forward_input_size),
            hidden_sizes=self.hidden_sizes,
            latent_size=self.latent_size
        )

        self.inverse_pred_head = InverseModelHead(
            input_dim=2 * self.latent_size,
            hidden_size=self.hidden_sizes,
            num_actions=inverse_pred_dim
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
        self.forward_agg_rnn.to(self.device)
        self.inverse_pred_head.to(self.device)
        self.transforms.to(self.device)
        self.byol_linear_trans.to(self.device)

        self.optim_initialize(n_updates)

        # load the pretrained models
        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_B)  # batch b is the batch_size of every single trajectory
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch

        self.optimizer.zero_grad()
        # calculate the loss func
        loss, spr_loss, contrast_loss, inverse_dyna_loss,  pred_accuracies, contrast_accuracy, inverse_pred_accuracy = self.mst_loss(samples)
        optimize_loss = spr_loss * self.spr_loss_coefficient + contrast_loss * self.contrast_loss_coefficient + inverse_dyna_loss * self.inverse_dyna_loss_coefficient
        optimize_loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # log the optimize info/result
        opt_info.mstLoss.append(loss.item())
        opt_info.sprLoss.append(spr_loss.item())
        opt_info.inverseDynaLoss.append(inverse_dyna_loss.item())
        opt_info.contrastLoss.append(contrast_loss.item())
        opt_info.cpcAccuracy1.append(pred_accuracies[0].item())
        opt_info.cpcAccuracy2.append(pred_accuracies[1].item())
        opt_info.cpcAccuracyTm1.append(pred_accuracies[2].item())
        opt_info.cpcAccuracyTm2.append(pred_accuracies[3].item())
        opt_info.contrast_accuracy.append(contrast_accuracy.item())
        opt_info.inverse_pred_accuracy.append(inverse_pred_accuracy.item())
        opt_info.gradNorm.append(grad_norm.item())

        # the update interval for the momentun encoder
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder,
                              self.encoder.state_dict(),
                              self.target_update_tau)
        return opt_info

    def mst_loss(self, samples):
        obs_one = samples.observations
        length, b, f, c, h, w = obs_one.shape
        obs_one = obs_one.view(length, b * f, c, h, w)  # Treat all T,B as separate.(reshape the sample)
        obs_two = copy.deepcopy(obs_one)
        translation = samples.translations
        rotation = samples.rotations
        direction_label = samples.directions
        onehot_direction = to_onehot(direction_label, num=8)
        action = torch.cat((translation, rotation), dim=-1)

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

        obs_one, obs_two, action, onehot_direction = buffer_to((obs_one, obs_two, action, onehot_direction),
                                                               device=self.device)

        with torch.no_grad():
            obs_one_target_proj, _ = self.target_encoder(obs_one)
            obs_two_target_proj, _ = self.target_encoder(obs_two)
            obs_one_target_proj.detach_()
            obs_two_target_proj.detach_()

        obs_one_online_proj, _ = self.encoder(obs_one)
        obs_two_online_proj, _ = self.encoder(obs_two)

        spr_loss, pred_accuracies = self.spr_loss(obs_one_online_proj, obs_two_target_proj, action)
        contrast_loss, contrast_accuracy = self.contrast_loss(obs_one_online_proj, obs_two_target_proj,
                                                              obs_two_online_proj, obs_one_target_proj)
        inverse_dyna_loss, inverse_pred_accuracy,  = self.inverse_dyna_loss(obs_two_online_proj,
                                                                            obs_one_target_proj, onehot_direction)

        loss = spr_loss + contrast_loss + inverse_dyna_loss

        return loss, spr_loss, contrast_loss, inverse_dyna_loss, \
            pred_accuracies, contrast_accuracy, inverse_pred_accuracy

    def spr_loss(self, obs_one_online_proj, obs_two_target_proj, action):
        rnn_input = torch.cat((obs_one_online_proj, action), dim=-1)
        context, _ = self.forward_agg_rnn(rnn_input)

        # Extract only the ones to train (all were needed to compute).
        context = context[self.warmup_T:]
        z_positive = obs_two_target_proj.detach()[self.warmup_T:]

        T, B, Z = z_positive.shape
        target_trans = z_positive.view(-1, Z).transpose(1, 0)
        base_labels = torch.arange(T * B, dtype=torch.long, device=self.device).view(T, B)
        prediction_list = list()
        label_list = list()
        # from 1 forward step to T-1 forward step
        for delta_t in range(1, T):
            prediction_list.append(self.transforms[delta_t](context[:-delta_t].view(-1, Z)))
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

    def contrast_loss(self, obs_one_online_proj, obs_two_target_proj,
                      obs_two_online_proj, obs_one_target_proj):
        T, B, latent_dim = obs_one_online_proj.shape

        obs_one_target_trans = obs_one_target_proj.detach().view(-1, latent_dim).transpose(1, 0)
        obs_two_target_trans = obs_two_target_proj.detach().view(-1, latent_dim).transpose(1, 0)  # [latent_dim, T*B]
        labels = torch.arange(T * B, dtype=torch.long, device=self.device)
        pred_one = self.byol_linear_trans(obs_one_online_proj).view(-1, latent_dim)
        pred_two = self.byol_linear_trans(obs_two_online_proj).view(-1, latent_dim)
        logits_one = torch.matmul(pred_one, obs_two_target_trans)  # [T*B, T*B]
        logits_two = torch.matmul(pred_two, obs_one_target_trans)
        logits_one = logits_one - torch.max(logits_one, dim=1, keepdim=True)[0]
        logits_two = logits_two - torch.max(logits_two, dim=1, keepdim=True)[0]
        contrast_loss_one = self.c_e_loss(logits_one, labels)
        contrast_loss_two = self.c_e_loss(logits_two, labels)
        contrast_loss = (contrast_loss_one + contrast_loss_two)
        correct = torch.argmax(logits_one.detach(), dim=1) == labels
        contrast_accuracy = torch.mean(correct.float())

        return contrast_loss, contrast_accuracy

    def inverse_dyna_loss(self, obs_two_online_proj, obs_one_target_proj, direction_label):
        T, B, latent_dim = obs_two_online_proj.shape
        next_state, _ = self.forward_agg_rnn(obs_two_online_proj)
        current_state = obs_one_target_proj.detach()
        x = torch.cat((next_state.reshape(T * B, latent_dim), current_state.reshape(T * B, latent_dim)), dim=-1)
        logits = self.inverse_pred_head(x)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        labels = direction_label.reshape(T * B, -1)
        inverse_loss = self.c_e_loss(logits, labels)
        correct = torch.argmax(logits.detach(), dim=1) == labels
        inverse_pred_accuracy = torch.mean(correct.float())
        return inverse_loss, inverse_pred_accuracy

    def validation(self, itr):
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            forward_agg_rnn=self.forward_agg_rnn.state_dict(),
            inverse_pred_head=self.inverse_pred_head.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.forward_agg_rnn.load_state_dict(state_dict['forward_agg_rnn'])
        self.inverse_pred_head.load_state_dict(state_dict['inverse_pred_head'])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.forward_agg_rnn.parameters()
        yield from self.inverse_pred_head.parameters()
        yield from self.transforms.parameters()
        yield from self.byol_linear_trans.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.forward_agg_rnn.named_parameters()
        yield from self.forward_agg_rnn.named_parameters()
        yield from self.transforms.named_parameters()
        yield from self.byol_linear_trans.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.forward_agg_rnn.eval()
        self.inverse_pred_head.eval()
        self.transforms.eval()
        self.byol_linear_trans.eval()

    def train(self):
        self.encoder.train()
        self.forward_agg_rnn.train()
        self.inverse_pred_head.train()
        self.transforms.train()
        self.byol_linear_trans.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example
