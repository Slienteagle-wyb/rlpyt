import numpy as np
from rlpyt.ul.replays.offline_dataset import OfflineSamples, OfflineDatasets
from rlpyt.utils.buffer import torchify_buffer, buffer_func
from rlpyt.utils.misc import extract_sequences
from rlpyt.utils.collections import namedarraytuple

BatchSamples = namedarraytuple('BatchSamples', ['observations', 'translations', 'rotations', 'prev_translations',
                                                'prev_rotations', 'velocities', 'directions', 'attitudes'])


class OfflineUlReplayBuffer:
    def __init__(self,
                 replay_buffer,
                 img_size=84,
                 frame_stacks=1,
                 data_path=None,
                 translation_dim=3,
                 rotation_dim=6,
                 command_catgorical=8,
                 episode_length=2000,
                 num_runs=1,
                 forward_step=0,
                 ):
        self.frame_stacks = frame_stacks
        self.forward_step = forward_step
        self.data_path = data_path
        self.episode_length = episode_length
        self.num_runs = num_runs
        self.img_size = img_size
        self.translation_dim = translation_dim
        self.rotation_dim = rotation_dim
        self.command_catgorical = command_catgorical
        # initialize the namedarray with the shape of [T, B, C, H, W] and [T, B, act_dim]
        self.example = OfflineSamples(observation=np.zeros((img_size, img_size, 3), dtype=np.uint8),
                                      translation=np.zeros(translation_dim, dtype=np.float32),
                                      rotation=np.zeros(rotation_dim, dtype=np.float32),
                                      velocity=np.zeros(4, dtype=np.float32),
                                      direction=np.zeros(1, dtype=np.float32),
                                      attitude=np.zeros(9, dtype=np.float32))
        self._load_replay(replay_buffer)

    def _load_replay(self, replay_buffer):
        self.loaded_buffer = replay_buffer(data_path=self.data_path,
                                           episode_length=self.episode_length,
                                           num_runs=self.num_runs,
                                           buffer_example=self.example,
                                           img_size=self.img_size,
                                           )
        self._samples = self.loaded_buffer.samples  # the loaded sample datas
        self.T = self.loaded_buffer.T
        self.B = self.loaded_buffer.B
        self.size = self.T * self.B

    @property
    def samples(self):
        return self._samples

    def get_example(self):
        return self.example

    def sample_batch(self, batch_size):
        T_idxs, B_idxs = self.sample_idxs(batch_size)
        return self.extract_batch(T_idxs, B_idxs)

    def sample_idxs(self, batch_size):
        B_idxs = np.random.randint(low=0, high=self.B, size=batch_size)
        T_idxs = np.random.randint(low=self.frame_stacks - 1, high=self.T - self.forward_step, size=batch_size)
        return T_idxs, B_idxs

    def extract_batch(self, T_idxs, B_idxs):
        # extract one more axis than obs
        translations = buffer_func(self.samples.translation,
                                   extract_sequences,
                                   T_idxs - 1, B_idxs, self.forward_step + 2)
        rotations = buffer_func(self.samples.rotation,
                                extract_sequences,
                                T_idxs - 1, B_idxs, self.forward_step + 2)
        velocities = buffer_func(self.samples.velocity,
                                 extract_sequences,
                                 T_idxs - 1, B_idxs, self.forward_step + 2)
        directions = buffer_func(self.samples.direction, extract_sequences,
                                 T_idxs - 1, B_idxs, self.forward_step + 2)
        attitudes = buffer_func(self.samples.attitude,
                                extract_sequences,
                                T_idxs - 1, B_idxs, self.forward_step + 2)
        batch = BatchSamples(
            observations=self.extract_observation(T_idxs, B_idxs, self.forward_step),
            translations=translations[1:],
            prev_translations=translations[:-1],
            rotations=rotations[1:],
            prev_rotations=rotations[:-1],
            velocities=velocities[1:],
            directions=directions[1:],
            attitudes=attitudes[1:]
        )
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs, forward_step):
        # return observatons as [T, B, F, C, H, W]
        observation = np.empty(shape=(self.forward_step + 1, len(B_idxs), self.frame_stacks) +
                                      self.samples.observation.shape[2:], dtype=self.samples.observation.dtype)

        for i, (t_idx, b_idx) in enumerate(zip(T_idxs, B_idxs)):
            if t_idx + forward_step > self.T:
                for frame in range(self.frame_stacks):
                    observation[0:self.T - t_idx, i, frame] = self.samples.observation[t_idx + frame:self.T + frame, b_idx]
                    observation[self.T - t_idx:, i, frame] = self.samples.observation[
                                                             frame:frame + self.forward_step + t_idx - self.T, b_idx]
            else:
                for frame in range(self.frame_stacks):
                    observation[:, i, frame] = self.samples.observation[t_idx + frame:t_idx + frame + self.forward_step + 1, b_idx]
        return observation


if __name__ == '__main__':
    replay = OfflineUlReplayBuffer(
        replay_buffer=OfflineDatasets,
        img_size=84,
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/il_val_datasets',
        episode_length=4096,
        num_runs=1,
        forward_step=0,
    )
    batchs = replay.sample_batch(16)
    x = batchs.observations
    print(x.shape)
    print(x.dtype)
    print(x[0].device)
