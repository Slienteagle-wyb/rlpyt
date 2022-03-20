import torch
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.ul.models.dmlab_conv2d import DmlabConv2dModel, StcConv2dModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.models.ul.atc_models import ByolMlpModel


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


class MstEncoderModel(torch.nn.Module):
    def __init__(
            self,
            image_shape,
            latent_size,
            action_dim,
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

        self.forward_dyna_head = MlpModel(  # gets to z_t, not necessarily c_t
            input_size=self._output_size+action_dim,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )
        if kaiming_init:
            self.apply(weight_init)

    def forward(self, observation, action):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.reshape(T * B, *img_shape))
        latent_z = torch.cat((conv.reshape(T * B, -1), action.reshape(T * B, -1)), dim=-1)
        c = self.forward_dyna_head(latent_z)

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


if __name__ == '__main__':
    model = ByolEncoderModel(image_shape=(3, 84, 84),
                             latent_size=256,
                             hidden_sizes=4096)
    for params in model.named_parameters():
        print(params[0])
