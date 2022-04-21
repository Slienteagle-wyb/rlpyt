import torch
import pickle
from rlpyt.utils.logging import logger
from rlpyt.ul.algos.utils.optim_factory import create_optimizer
from rlpyt.ul.algos.utils.scheduler_factory import create_scheduler


class UlAlgorithm:

    opt_info_fields = ()

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def load_replay(self):
        raise NotImplementedError

    def optimize(self, itr):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def eval(self):
        """Call this on NN modules."""
        raise NotImplementedError

    def train(self):
        """Call this on NN modules."""
        raise NotImplementedError

    def validation(self, *args, **kwargs):
        raise NotImplementedError


class BaseUlAlgorithm(UlAlgorithm):
    """A few common methods."""

    def load_replay(self, pixel_control_buffer=None):
        """Loads either one or multiple replay buffer files,
        must assign self.ReplayCls accordingly."""
        if isinstance(self.replay_filepath, (list, tuple)):
            logger.log("Loading multiple replay buffers...")
            replay_buffer = list()
            for rep_file in self.replay_filepath:
                with open(rep_file, "rb") as fh:
                    replay_buffer.append(pickle.load(fh))
            logger.log("Replay buffers loaded; combining...")
        else:
            logger.log("Loading replay buffer...")
            with open(self.replay_filepath, "rb") as fh:
                replay_buffer = pickle.load(fh)
            logger.log("Replay buffer loaded")
        self.replay_buffer = self.ReplayCls(
            replay_buffer=replay_buffer,
            replay_T=self.replay_T,
            validation_split=self.validation_split,
            pixel_control_buffer=pixel_control_buffer,
        )
        if isinstance(replay_buffer, list):
            logger.log("Replay buffers combined")
        examples = self.replay_buffer.get_examples()
        return examples

    @property
    def replay_T(self):
        """Set this in each algo's init."""
        return self._replay_T

    def optim_initialize(self, epochs):
        self.optimizer = create_optimizer(
            model=self,
            **self.optim_kwargs,
        )

        self.lr_scheduler = None
        sched_slice = self.sched_kwargs.pop('epoch_slice', 1)
        self.lr_scheduler, _ = create_scheduler(
            optimizer=self.optimizer,
            num_epochs=epochs // sched_slice,
            sched_kwargs=self.sched_kwargs,
        )
        if self.lr_scheduler is not None:
            self.optimizer.zero_grad()
            self.optimizer.step()  # needed to initialize the scheduler

    def activation_loss(self, conv_output):
        """Rarely if ever used this."""
        if getattr(self, "activation_loss_coefficient", 0.) == 0.:
            return torch.tensor(0., device=self.device)
        # Only penalize above 1 (conv_output is after ReLU).
        large_x = torch.clamp(conv_output.view(-1) - 1, min=0.)
        # Gentle squared-magnitude loss, l2-like
        act_loss = large_x.pow(2).mean()
        return self.activation_loss_coefficient * act_loss
