import torch
import umap
import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.ul.replays.offline_dataset import OfflineDatasets
from rlpyt.ul.models.ul.encoders import DmlabEncoderModelNorm, ResEncoderModel, DmlabEncoderModel


class OfflineUMAP:
    def __init__(self, color_palette: str = 'hls'):
        self.color_palette = color_palette

    def plot(self,
             device: str,
             model: torch.nn.modules,
             dataloader: OfflineUlReplayBuffer,
             batch_size: int,
             plot_path: str):
        conv_feats = []
        directions = []
        num_itrs = int(dataloader.size // batch_size)

        model.eval()
        for i in range(num_itrs):
            batch_data = dataloader.sample_batch(batch_size=batch_size)
            observation = batch_data.observations.to(device)
            t, b, f, c, h, w = observation.shape
            observation = observation.reshape(t, b*f, c, h, w)
            direction = batch_data.directions
            with torch.no_grad():
                _, conv_feat = model(observation)
                conv_feat = conv_feat.reshape(t, b*f, -1)
                conv_feats.append(conv_feat.squeeze().cpu())
                directions.append(direction.squeeze())
        model.train()

        conv_feats = torch.cat(conv_feats, dim=0).numpy()  # tensor_shape:(batch_size*itrs, latent_dim)
        directions = torch.cat(directions, dim=0)
        num_classes = len(torch.unique(directions))
        print(num_classes)
        directions = directions.numpy()

        print("Creating UMAP")
        data = umap.UMAP(n_components=2).fit_transform(conv_feats)

        # passing to dataframe
        df = pd.DataFrame()
        df["feat_1"] = data[:, 0]
        df["feat_2"] = data[:, 1]
        df["Y"] = directions
        plt.figure(figsize=(9, 9))
        ax = sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=df,
            legend="full",
            alpha=0.3,
        )
        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, right=False, bottom=False, top=False)

        # manually improve quality of imagenet umaps
        if num_classes > 100:
            anchor = (0.5, 1.8)
        else:
            anchor = (0.5, 1.35)

        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
        plt.tight_layout()

        # save plot locally as well
        plt.savefig(plot_path)
        plt.close()


def main():
    device = 'cuda:0'
    model = DmlabEncoderModel(
        image_shape=(3, 84, 84),
        latent_size=256,
        hidden_sizes=None
    )
    # model = ResEncoderModel(
    #     image_shape=(3, 84, 84),
    #     latent_size=256,
    #     hidden_sizes=512,
    #     num_stacked_input=1
    # )
    state_dict_path = f'/home/yibo/spaces/snap_shots/rlpyt_drone_representation/20220315/144748/cpc_pretrain/itr_199999.pkl'
    print('models loading state dict ....')
    loaded_state_dict = torch.load(state_dict_path,
                                   map_location=torch.device('cpu'))
    # the conv state dict was stored as encoder
    loaded_state_dict = loaded_state_dict.get('algo_state_dict', loaded_state_dict)
    loaded_state_dict = loaded_state_dict.get('encoder', loaded_state_dict)
    for keys in loaded_state_dict:
        print(keys)
    # conv_state_dict = OrderedDict([(k, v) for k, v in loaded_state_dict.items() if k.startswith('conv.')])
    model.load_state_dict(loaded_state_dict)
    print('conv encoder has loaded the pretrained model')
    model.to(device)
    dataloader = OfflineUlReplayBuffer(
        replay_buffer=OfflineDatasets,
        img_size=84,
        frame_stacks=1,
        data_path=f'/home/yibo/spaces/datasets/drone_repr_body',
        episode_length=1792,
        num_runs=60,
        forward_step=0,
    )
    batch = 64
    plot_path = 'in_domain_cpc_umap.pdf'

    umap = OfflineUMAP()
    umap.plot(device, model, dataloader, batch, plot_path)


if __name__ == '__main__':
    main()
