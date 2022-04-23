import torch
from collections import namedtuple
import copy
import wandb
from rlpyt.utils.tensor import valid_mean
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.encoders import DmlabEncoderModel, DmlabEncoderModelNorm
from rlpyt.ul.models.ul.atc_models import ByolMlpModel
from rlpyt.ul.algos.utils.data_augs import get_augmentation, random_shift
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.models.ul.inverse_models import InverseModelHead
from rlpyt.ul.models.ul.forward_models import ForwardAggRnnModel, SkipConnectForwardAggModel
import torchvision.transforms as Trans
import torch.nn.functional as F


IGNORE_INDEX = -100  # Mask TC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["mstLoss", "sprLoss", "contrastLoss", 'inverseDynaLoss',
                                 "cpcAccuracy1", "cpcAccuracy2", "cpcAccuracyTm1", "cpcAccuracyTm2",
                                 'inverse_pred_accuracy', 'cos_similarity', "gradNorm", 'current_lr'])
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
            clip_grad_norm=10.,
            target_update_tau=0.01,  # 1 for hard update
            target_update_interval=1,
            latent_size=256,
            hidden_sizes=512,
            random_shift_prob=1.,
            random_shift_pad=4,
            augmentations=('intensity',),  # combined with intensity jit accord to SGI
            spr_loss_coefficient=1.0,
            contrast_loss_coefficient=1.0,
            inverse_dyna_loss_coefficient=1.0,
            validation_split=0.0,
            n_validation_batches=0,
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=DmlabEncoderModelNorm,
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
        self._replay_T = warmup_T + batch_T  # self.replay_T == self._replay_T is the len of every sampled trajectory


    def initialize(self, epochs, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)

        examples = self.load_replay()
        self.itrs_per_epoch = self.replay_buffer.size // self.batch_size
        self.n_updates = epochs * self.itrs_per_epoch
        self.image_shape = image_shape = examples.observation.shape  # [c, h, w]

        trans_dim = self.replay_buffer.translation_dim
        rotate_dim = self.replay_buffer.rotation_dim
        command_catgorical = self.replay_buffer.command_catgorical
        forward_input_size = rotate_dim + trans_dim  # no reward
        inverse_pred_dim = command_catgorical

        self.encoder = self.EncoderCls(
            image_shape=image_shape,
            latent_size=self.latent_size,
            hidden_sizes=self.hidden_sizes,
            **self.encoder_kwargs
        )
        self.target_encoder = copy.deepcopy(self.encoder)  # the target encoder is not tied with online encoder

        self.online_predictor = ByolMlpModel(
            input_dim=self.latent_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_sizes
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

        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.online_predictor.to(self.device)
        self.forward_agg_rnn.to(self.device)
        self.forward_pred_rnn.to(self.device)
        self.inverse_pred_head.to(self.device)
        self.transforms.to(self.device)

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
        loss, spr_loss, contrast_loss, inverse_dyna_loss,  pred_accuracies, inverse_pred_accuracy, cos_similarity = self.mst_loss(samples)
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
        opt_info.inverse_pred_accuracy.append(inverse_pred_accuracy.item())
        opt_info.cos_similarity.append(cos_similarity.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.current_lr.append(current_lr)

        # the update interval for the momentum encoder
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
        prev_translation = samples.prev_translations
        prev_rotation = samples.prev_rotations
        current_translation = samples.translations
        current_rotation = samples.rotations
        direction_label = samples.directions
        prev_action = torch.cat((prev_translation, prev_rotation), dim=-1)
        current_action = torch.cat((current_translation, current_rotation), dim=-1)

        lead_dim, batch_len, batch_size, shape = infer_leading_dims(obs_one, 3)
        obs_one = obs_one.reshape(batch_len*batch_size, *shape)
        obs_two = obs_two.reshape(batch_len*batch_size, *shape)
        aug_trans = Trans.Compose(get_augmentation(self.augmentations, 84))
        obs_one = aug_trans(obs_one)
        obs_two = aug_trans(obs_two)
        obs_one = restore_leading_dims(obs_one, lead_dim, batch_len, batch_size)
        obs_two = restore_leading_dims(obs_two, lead_dim, batch_len, batch_size)

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

        obs_one, obs_two, prev_action, current_action, direction_label = buffer_to((obs_one, obs_two, prev_action,
                                                                                    current_action, direction_label),
                                                                                   device=self.device)

        with torch.no_grad():
            obs_one_target_proj, _ = self.target_encoder(obs_one)
            obs_two_target_proj, _ = self.target_encoder(obs_two)

        obs_one_online_proj, _ = self.encoder(obs_one)
        obs_two_online_proj, _ = self.encoder(obs_two)

        spr_loss, pred_accuracies = self.spr_loss(obs_one_online_proj, obs_two_target_proj, prev_action, current_action)
        contrast_loss, cos_similarity = self.contrast_loss(obs_one_online_proj, obs_two_target_proj,
                                                           obs_two_online_proj, obs_one_target_proj)
        inverse_dyna_loss, inverse_pred_accuracy,  = self.inverse_dyna_loss(obs_two_online_proj,
                                                                            obs_one_target_proj,
                                                                            direction_label,
                                                                            prev_action)

        loss = spr_loss + contrast_loss + inverse_dyna_loss

        return loss, spr_loss, contrast_loss, inverse_dyna_loss, \
            pred_accuracies, inverse_pred_accuracy, cos_similarity

    def spr_loss(self, obs_one_online_proj, obs_two_target_proj, prev_action, current_action):
        agg_input = torch.cat((obs_one_online_proj[:self.warmup_T], prev_action[:self.warmup_T]), dim=-1)
        _, hidden_states = self.forward_agg_rnn(agg_input)

        current_action = current_action[self.warmup_T:]
        pred_next_states, _ = self.forward_pred_rnn(current_action, init_hiddens=hidden_states[-1])  # pred_next_states [16, 16, 256]
        z_positive = obs_two_target_proj.detach()[self.warmup_T:]

        T, B, Z = z_positive.shape
        target_trans = z_positive.view(-1, Z).transpose(1, 0)
        base_labels = torch.arange(T * B, dtype=torch.long, device=self.device).view(T, B)
        prediction_list = list()
        label_list = list()
        # from 1 forward step to T-1 forward step
        for delta_t in range(1, T):
            prediction_list.append(self.transforms[delta_t](pred_next_states[:-delta_t].view(-1, Z)))
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

        obs_one_target_proj = obs_one_target_proj.detach().view(-1, latent_dim)
        obs_two_target_proj = obs_two_target_proj.detach().view(-1, latent_dim)  # [T*B, latent_dim]

        pred_one = self.online_predictor(obs_one_online_proj.view(-1, latent_dim))
        pred_two = self.online_predictor(obs_two_online_proj.view(-1, latent_dim))

        loss_one = self.byol_loss_fn(pred_one, obs_one_target_proj)
        loss_two = self.byol_loss_fn(pred_two, obs_two_target_proj)
        contrast_loss = loss_one + loss_two

        global_latents = obs_one_online_proj.detach().view(-1, latent_dim)
        global_latents = F.normalize(global_latents, p=2.0, dim=-1, eps=1e-3)
        cos_sim = torch.matmul(global_latents, global_latents.transpose(1, 0))  # get a matrix [T*B, T*B]
        mask = 1 - torch.eye(T*B, device=self.device, dtype=torch.float)
        cos_sim = cos_sim*mask  # mask the similarity of every self
        offset = cos_sim.shape[-1] / (cos_sim.shape[-1] - 1)  # (T*B)/(T*B-1)
        cos_sim = cos_sim.mean() * offset
        return contrast_loss.mean(), cos_sim

    def inverse_dyna_loss(self, obs_two_online_proj, obs_one_target_proj, direction_label, action):
        rnn_input = torch.cat((obs_two_online_proj, action), dim=-1)
        agg_states, _ = self.forward_agg_rnn(rnn_input)

        agg_states = agg_states[self.warmup_T:]
        direction_label = direction_label[self.warmup_T:]
        obs_one_target_proj = obs_one_target_proj[self.warmup_T:]

        next_states = agg_states[1:]
        current_states = obs_one_target_proj.detach()[:-1]
        T, B, latent_dim = current_states.shape
        x = torch.cat((next_states.reshape(T * B, latent_dim), current_states.reshape(T * B, latent_dim)), dim=-1)
        logits = self.inverse_pred_head(x)
        direction_label = direction_label[:-1]
        labels = direction_label.reshape(T * B, -1).squeeze()
        inverse_loss = self.c_e_loss(logits, labels.long())

        correct = torch.argmax(logits.detach(), dim=1) == labels
        inverse_pred_accuracy = torch.mean(correct.float())
        return inverse_loss, inverse_pred_accuracy

    def byol_loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def validation(self, itr):
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            forward_agg_rnn=self.forward_agg_rnn.state_dict(),
            forward_pred_rnn=self.forward_pred_rnn.state_dict(),
            inverse_pred_head=self.inverse_pred_head.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.online_predictor.load_state_dict(state_dict["online_predictor"])
        self.forward_agg_rnn.load_state_dict(state_dict['forward_agg_rnn'])
        self.forward_pred_rnn.load_state_dict(state_dict['forward_pred_rnn'])
        self.inverse_pred_head.load_state_dict(state_dict['inverse_pred_head'])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.forward_agg_rnn.parameters()
        yield from self.forward_pred_rnn.parameters()
        yield from self.online_predictor.parameters()
        yield from self.inverse_pred_head.parameters()
        yield from self.transforms.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.online_predictor.named_parameters()
        yield from self.forward_agg_rnn.named_parameters()
        yield from self.forward_pred_rnn.named_parameters()
        yield from self.inverse_pred_head.named_parameters()
        yield from self.transforms.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.online_predictor.eval()
        self.forward_agg_rnn.eval()
        self.forward_pred_rnn.eval()
        self.inverse_pred_head.eval()
        self.transforms.eval()

    def train(self):
        self.encoder.train()
        self.online_predictor.train()
        self.forward_agg_rnn.train()
        self.forward_pred_rnn.train()
        self.inverse_pred_head.train()
        self.transforms.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example

    def wandb_log_code(self):
        wandb.save('./rlpyt/ul/algos/ul_for_rl/mst.py')
