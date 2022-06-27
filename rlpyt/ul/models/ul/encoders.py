import torch
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.ul.models.dmlab_conv2d import DmlabConv2dModel, DmlabConv2dModelBn
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.models.ul.atc_models import ByolMlpModel
from rlpyt.ul.models.ul.residual_networks import ResnetCNN
from rlpyt.models.utils import conv2d_output_shape
from torchvision.models import resnet18


def weight_init(m):
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class EncoderModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            latent_size,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            hidden_sizes=None,  # usually None; NOT the same as anchor MLP
            kaiming_init=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=False,
        )
        self._output_size = self.conv.conv_out_size(h, w)
        # self._output_shape = self.conv.conv_out_shape(h, w)
        self.head = MlpModel(
            input_size=self._output_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )
        if kaiming_init:
            self.apply(weight_init)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.view(T * B, *img_shape))
        c = self.head(conv.contiguous().view(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


class DmlabEncoderModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            latent_size,
            use_fourth_layer=True,
            skip_connections=True,
            hidden_sizes=None,
            kaiming_init=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = DmlabConv2dModel(
            in_channels=c,
            use_fourth_layer=use_fourth_layer,
            skip_connections=skip_connections,
            use_maxpool=False,
        )
        self._output_size = self.conv.output_size(h, w)
        self._output_shape = self.conv.output_shape(h, w)

        self.head = MlpModel(  # gets to z_t, not necessarily c_t
            input_size=self._output_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )
        if kaiming_init:
            self.apply(weight_init)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.reshape(T * B, *img_shape))
        c = self.head(conv.reshape(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


class ByolEncoderModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            latent_size,
            use_fourth_layer=True,
            skip_connections=True,
            hidden_sizes=None,
            kaiming_init=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = DmlabConv2dModel(
            in_channels=c,
            use_fourth_layer=use_fourth_layer,
            skip_connections=skip_connections,
            use_maxpool=False,
        )
        self._output_size = self.conv.output_size(h, w)
        self._output_shape = self.conv.output_shape(h, w)

        self.head = ByolMlpModel(
            input_dim=self._output_size,
            latent_size=latent_size,
            hidden_size=hidden_sizes
        )
        if kaiming_init:
            self.apply(weight_init)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.view(T * B, *img_shape))
        c = self.head(conv.contiguous().view(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


class DmlabEncoderModelNorm(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            latent_size,
            use_fourth_layer=True,
            skip_connections=True,
            hidden_sizes=None,
            kaiming_init=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = DmlabConv2dModelBn(
            in_channels=c,
            use_fourth_layer=use_fourth_layer,
            skip_connections=skip_connections,
            use_maxpool=False,
            norm_type='bn'
        )
        self._output_size = self.conv.output_size(h, w)
        self._output_shape = self.conv.output_shape(h, w)
        self.head = ByolMlpModel(
            input_dim=self._output_size,
            latent_size=latent_size,
            hidden_size=hidden_sizes
        )
        if kaiming_init:
            self.apply(weight_init)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.reshape(T * B, *img_shape))
        c = self.head(conv.reshape(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


class ResEncoderModel(torch.nn.Module):
    def __init__(self,
                 image_shape,
                 latent_size,
                 hidden_sizes,
                 num_stacked_input=3,
                 res_depths=(32, 64, 64),
                 downsampling_strides=(3, 2, 2),
                 blocks_per_group=3,
                 expand_ratio=2
                 ):
        super(ResEncoderModel, self).__init__()
        self.num_stacked_input = num_stacked_input
        c, h, w = image_shape
        c = c * num_stacked_input
        self.conv = ResnetCNN(input_channels=c,
                              depths=res_depths,
                              strides=downsampling_strides,
                              blocks_per_group=blocks_per_group,
                              expand_ratio=expand_ratio)
        self._output_size = self.conv.output_size(h, w)
        self._output_shape = self.conv.output_shape(h, w)

        self.head = ByolMlpModel(
            input_dim=res_depths[-1],
            latent_size=latent_size,
            hidden_size=hidden_sizes
        )

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.reshape(T * B, *img_shape))
        conv = conv.mean(dim=(2, 3))
        c = self.head(conv.reshape(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


class Res18Encoder(torch.nn.Module):
    def __init__(self,
                 latent_size,
                 hidden_sizes,
                 num_stacked_input=1,
                 state_dict_path=None,
                 image_shape=None,
                 ):
        super().__init__()
        self.num_stacked_input = num_stacked_input
        self.conv = resnet18()
        self.conv.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, bias=False)
        self.conv.maxpool = torch.nn.Identity()
        self.conv.fc = torch.nn.Identity()
        # self.conv.avgpool = torch.nn.AdaptiveMaxPool2d((2, 2))
        # self.conv.avgpool = torch.nn.Identity()
        if state_dict_path is not None:
            print('loading the off-shelf pretrained model....')
            state = torch.load(state_dict_path)
            state_dict = state['state_dict']
            for k in list(state_dict.keys()):
                if 'backbone' in k:
                    state_dict[k.replace('backbone.', '')] = state_dict[k]
                del state_dict[k]
            self.conv.load_state_dict(state_dict, strict=False)
            print('had successfully loaded the pretrained model params!!!')
        self.output_shape = self.conv(torch.randn(1, 3, 144, 144)).shape
        self.head = ByolMlpModel(
            input_dim=self.output_shape[-1] * self.num_stacked_input,
            latent_size=latent_size,
            hidden_size=hidden_sizes
        )

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.reshape(T * B, *img_shape))

        if self.num_stacked_input > 1:
            conv_feature_list = []
            conv = restore_leading_dims(conv, lead_dim, T, B)
            assert T % self.num_stacked_input == 0
            for i in range(self.num_stacked_input):
                conv_feature_list.append(conv[i::self.num_stacked_input])
            # shape of the conv is (T/self.num_stacked_input, B, conv_feature * self.num_stacked_input)
            conv = torch.cat(conv_feature_list, dim=-1)
            assert conv.shape[0] == int(T / self.num_stacked_input)

        img_embedding = conv.reshape(int(T * B / self.num_stacked_input), -1)
        c = self.head(img_embedding)
        c = restore_leading_dims(c, lead_dim, int(T // self.num_stacked_input), B)
        return c, conv

    @property
    def output_size(self):
        return self.output_shape[-1]


class FusResEncoderModel(torch.nn.Module):
    def __init__(self,
                 image_shape,
                 latent_size,
                 hidden_sizes,
                 num_stacked_input=3,
                 res_depths=(32, 64, 64),
                 downsampling_strides=(3, 2, 2),
                 blocks_per_group=3,
                 expand_ratio=2
                 ):
        super(FusResEncoderModel, self).__init__()
        self.num_stacked_input = num_stacked_input
        c, h, w = image_shape
        c = c * num_stacked_input
        self.conv = ResnetCNN(input_channels=c,
                              depths=res_depths,
                              strides=downsampling_strides,
                              blocks_per_group=blocks_per_group,
                              expand_ratio=expand_ratio)
        self._output_size = self.conv.output_size(h, w)
        self._output_shape = self.conv.output_shape(h, w)

        self.spatial_head = ByolMlpModel(
            input_dim=self._output_size,
            hidden_size=hidden_sizes,
            latent_size=latent_size
        )
        self.temporal_head = ByolMlpModel(
            input_dim=self._output_size,
            hidden_size=hidden_sizes,
            latent_size=latent_size
        )

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        z = self.conv(img.reshape(T * B, *img_shape))
        # spatial feature
        z_spatial = self.spatial_head(z.reshape(T * B, -1))
        # temporal feature
        z_temporal = self.temporal_head(z.reshape(T * B, -1))
        z_spatial, z_temporal, z = restore_leading_dims((z_spatial, z_temporal, z), lead_dim, T, B)

        return z_spatial, z_temporal, z  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


if __name__ == '__main__':
    img = torch.randn((9, 1, 3, 144, 144))
    model = Res18Encoder(
        latent_size=256,
        hidden_sizes=512,
        state_dict_path=f'/home/yibo/Documents/solo-learn/pretrain_models/byol/byol-400ep-imagenet100-ep=399.ckpt'
    )
    c, conv = model(img)
    print(c.shape)
    print(conv.shape)
    # model = resnet18()
    # nodes, _ = get_graph_node_names(model)
    # features = {'layer4.0.relu_1': 'out'}
    # feature_extractor = create_feature_extractor(model, return_nodes=features)
    # out = feature_extractor(img)
    # print(out['out'].shape)
    # test

