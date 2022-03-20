import torch
from collections import namedtuple
import copy
import torch.nn.functional as F
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.encoders import ByolEncoderModel
from rlpyt.ul.models.ul.atc_models import ByolMlpModel
from rlpyt.ul.algos.utils.data_augs import byol_aug
from rlpyt.ul.replays.offline_dataset import OfflineDatasets

OptInfo = namedtuple("OptInfo", ["byolLoss", "gradNorm"])
ValInfo = namedtuple("ValInfo", ["atcLoss", "accuracy", "convActivation"])


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class ByolContrast(BaseUlAlgorithm):
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            delta_T=0,  # delta_T is the forward step
            batch_T=1,  # batch_T is the default len of T_axis for every batch
            batch_B=512,  # batch B is the sampled batch size for extraction
            learning_rate=1e-3,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            clip_grad_norm=10.,
            target_update_tau=0.01,  # 1 for hard update
            target_update_interval=1,
            proj_latent_size=256,
            predict_hidden_size=512,
            validation_split=0.0,
            n_validation_batches=0,  # usually don't do it.
            ReplayCls=OfflineUlReplayBuffer,
            EncoderCls=ByolEncoderModel,
            PredictCls=ByolMlpModel,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            encoder_kwargs=None,
            replay_kwargs=None,
            initial_state_dict=None,
            ):
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        assert learning_rate_anneal in [None, "cosine"]

        self.batch_size = batch_B * batch_T  # for logging only
        self._replay_T = delta_T + batch_T  # self.replay_T == self._replay_T is the len of every sampled trajectory

    def initialize(self, n_updates, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device("cuda", index=cuda_idx)

        examples = self.load_replay()
        self.image_shape = image_shape = examples.observation.shape  # [c, h, w]

        self.encoder = self.EncoderCls(
            image_shape=image_shape,
            latent_size=self.proj_latent_size,
            **self.encoder_kwargs
        )
        self.target_encoder = copy.deepcopy(self.encoder)  # the target encoder is not tied with online encoder

        self.online_predictor = self.PredictCls(
            input_dim=self.proj_latent_size,
            latent_size=self.proj_latent_size,
            hidden_size=self.predict_hidden_size
        )

        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.online_predictor.to(self.device)

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
        loss = self.byol_loss(samples)
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        # log the optimize info/result
        opt_info.byolLoss.append(loss.item())
        opt_info.gradNorm.append(grad_norm.item())

        # the update interval for the momentun encoder
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder,
                              self.encoder.state_dict(),
                              self.target_update_tau)
        return opt_info

    def byol_loss(self, samples):
        x = samples.observations[0]  # [forward_step+1, batch_size, frame_stack, c, h, w]
        b, f, c, h, w = x.shape
        x = x.view(b * f, c, h, w)
        img_one, img_two = byol_aug(x), byol_aug(x)  # one img with different aug operation
        img_one, img_two = buffer_to((img_one, img_two), device=self.device)
        online_proj_one, online_repre_one = self.encoder(img_one)
        online_proj_two, online_repre_two = self.encoder(img_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_proj_one, target_repre_one = self.target_encoder(img_one)
            target_proj_two, target_repre_two = self.target_encoder(img_two)
            target_proj_one.detach_()
            target_proj_two.detach_()
        loss_one = loss_fn(online_pred_one, target_proj_one.detach())
        loss_two = loss_fn(online_pred_two, target_proj_two.detach())

        loss = loss_one + loss_two
        byol_loss = loss.mean()
        return byol_loss

    def validation(self, itr):
        pass

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            online_predictor=self.online_predictor.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.online_predictor.load_state_dict(state_dict["online_predictor"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.online_predictor.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.online_predictor.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.online_predictor.eval()

    def train(self):
        self.encoder.train()
        self.online_predictor.train()

    def load_replay(self, pixel_control_buffer=None):
        logger.log('Loading replay buffer ...')
        self.replay_buffer = self.ReplayCls(OfflineDatasets, **self.replay_kwargs)
        logger.log("Replay buffer loaded")
        example = self.replay_buffer.get_example()
        return example
