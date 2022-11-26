import torch
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt
import rlpyt.ul.algos.utils.tsne_utils as utils
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.models.ul.encoders import DmlabEncoderModelNorm, ResEncoderModel, DmlabEncoderModel
import pylustrator

def main():
    gpu_device = 'cuda:0'
    batch_size = 80
    encoder = DmlabEncoderModelNorm(
        image_shape=(3, 84, 84),
        latent_size=256,
        hidden_sizes=512
    )
    # stc_model: f'/home/yibo/Documents/rlpyt/data/local/20220507/143654/mst_state_mlp_vel_regressor/state_latent_dim_256/itr_47999.pkl'
    # e2e_model: f'/home/yibo/Documents/rlpyt/data/local/20220825/231501/mst_state_mlp_vel_regressor/state_latent_dim_256_nostack/itr_23999.pkl'
    state_dict_path = f'/home/yibo/Documents/rlpyt/data/local/20220507/143654/mst_state_mlp_vel_regressor/state_latent_dim_256/itr_47999.pkl'
    loaded_state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
    loaded_state_dict = loaded_state_dict.get('algo_state_dict', loaded_state_dict)
    loaded_state_dict = loaded_state_dict.get('encoder', loaded_state_dict)
    for keys in loaded_state_dict:
        print(keys)
    # conv_state_dict = OrderedDict([(k, v) for k, v in loaded_state_dict.items() if k.startswith('conv.')])
    encoder.load_state_dict(loaded_state_dict)
    encoder.to(gpu_device)
    print('has successfully loaded the pretrianed model')
    val_dataset = OfflineUlReplayBuffer(
        replay_buffer=OfflineDatasets,
        img_size=84,
        frame_stacks=1,
        data_path=f'/home/yibo/spaces/datasets/tsne_data',
        episode_length=640,  # 4096
        num_runs=1,
        forward_step=0,
    )

    features = []


    images = val_dataset.samples.observation.reshape(-1, 3, 84, 84)
    images = np.multiply(images, 1 / 255.0, dtype=np.float32)
    images = torch.tensor(images).to(gpu_device).type(torch.float)

    num_samples = len(images)
    y_idx = np.arange(num_samples)
    y_label = np.multiply(y_idx, 1/80.0).astype(np.int32)
    num_iters = int(val_dataset.size // batch_size)

    encoder.eval()
    for i in range(num_iters):
        batch_data = images[i*batch_size: (i+1)*batch_size]
        observation = batch_data
        with torch.no_grad():
            proj_feat, _ = encoder(observation)
            features.append(proj_feat.squeeze())
    encoder.train()

    embeddings = torch.cat(features, dim=0).cpu().numpy()  # (num_samples, latent_dim)

    embedding_annealing = TSNE(
        perplexity=30,  # 30 is proper for local struct
        metric='euclidean',  # 'cosine'
        n_jobs=8,
        random_state=42,
        verbose=True
    ).fit(embeddings)

    embedding_annealing.affinities.set_perplexities(10)  # 10
    embedding_annealing = embedding_annealing.optimize(250, momentum=0.8)

    pylustrator.start()
    utils.plot(embedding_annealing, y_label, colors=utils.MOUSE_10X_COLORS, draw_legend=False,
               title='Navigation-STC-Embeddings')
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).set_size_inches(12.230000/2.54, 9.230000/2.54, forward=True)
    plt.figure(1).axes[0].set_facecolor("#ffffffff")
    plt.figure(1).axes[0].set_position([0.092424, 0.109600, 0.806462, 0.806101])
    plt.figure(1).axes[0].spines['bottom'].set_color('#cccccc')
    plt.figure(1).axes[0].spines['left'].set_color('#cccccc')
    plt.figure(1).axes[0].spines['right'].set_color('#cccccc')
    plt.figure(1).axes[0].spines['top'].set_color('#cccccc')
    plt.figure(1).axes[0].title.set_fontname("Times New Roman")
    plt.figure(1).axes[0].get_legend()._set_loc((0.05, 0.5))
    #% end: automatic generated code from pylustrator
    # plt.show()
    plt.savefig(f'/home/yibo/Pictures/stc_tsne_30_10_no_center.pdf', dpi=300)


if __name__ == '__main__':
    main()
