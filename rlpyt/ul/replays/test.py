from ul.replays.offline_dataset import OfflineDatasets
from ul.replays.offline_dataset import OfflineSamples
import numpy as np
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
import os.path as osp
from rlpyt.utils.launching.variant import make_variants, VariantLevel

# example = OfflineSamples(observation=np.zeros((3, 84, 84), dtype=np.float32),
#                          translation=np.zeros(3, dtype=np.float32),
#                          rotation=np.zeros(6, dtype=np.float32),
#                          velocity=np.zeros(4, dtype=np.float32))

# params = {
#     'img_size': 84,
#     'frame_stacks': 1,
#     'data_path': f'/home/yibo/spaces/drone_repr',
#     'episode_length': 4000,
#     'num_runs': 1,
# }

# dataset = OfflineDatasets(data_path=f'/home/yibo/spaces/drone_repr',
#                           episode_length=4000,
#                           num_runs=20,
#                           buffer_example=example,
#                           img_size=84,
#                           )

replay_buffer = OfflineUlReplayBuffer(OfflineDatasets,
                                      data_path=f'/home/yibo/spaces/datasets/drone_repr',
                                      forward_step=31,
                                      num_runs=50
                                      )
batchs = replay_buffer.sample_batch(8)
print(batchs.observations.shape)
print(batchs.translations[:, 0])
print(batchs.rotations[:, 0])


