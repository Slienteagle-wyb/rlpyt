import torch
import numpy as np
from rlpyt.models.utils import scale_grad, conv2d_output_shape
import torch.nn.functional as F
from rlpyt.ul.models.utils import init_normalization


def fixup_init(layer, num_layers):
    torch.nn.init.normal_(layer.weight, mean=0, std=np.sqrt(
        2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * num_layers ** (-0.25))


# a block of an inverted group conv with normalization and params initialization
class InvertedResidual(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 norm_type,
                 num_layers=1,
                 groups=-1,  # use inverted group layers
                 drop_prob=0.,
                 bias=True):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2, 3]
        self.drop_prob = drop_prob
        # returns a floating point number with specified number of decimals
        hidden_dim = round(in_channels * expand_ratio)
        if groups <= 0:  # use group kernel
            groups = hidden_dim

        conv = torch.nn.Conv2d

        # define the conv down_sampling layer
        if stride != 1:
            self.downsample = torch.nn.Conv2d(in_channels, out_channels, stride, stride)  # kernel_size==stride
            torch.nn.init.normal_(self.downsample.weight, mean=0, std=
            np.sqrt(2 / (self.downsample.weight.shape[0] *
                         np.prod(self.downsample.weight.shape[2:]))))
        else:
            self.downsample = False

        if expand_ratio == 1:
            conv1 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)
            conv2 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            self.conv = torch.nn.Sequential(
                # depth wise convolution layer
                conv1,
                init_normalization(hidden_dim, norm_type),
                torch.nn.ReLU(inplace=True),
                # point wise convolution layer with 1x1 kernel
                conv2,
                init_normalization(out_channels, norm_type),
            )
            torch.nn.init.constant_(self.conv[-1].weight, 0)
        else:
            conv1 = conv(in_channels, hidden_dim, 1, 1, 0, bias=bias)  # point wise 1x1 kernel for channel expand
            conv2 = conv(hidden_dim, hidden_dim, 3, stride, 1, groups=groups, bias=bias)  # depth wise layer
            conv3 = conv(hidden_dim, out_channels, 1, 1, 0, bias=bias)  # point wise conv layer for dim reduction
            fixup_init(conv1, num_layers)
            fixup_init(conv2, num_layers)
            fixup_init(conv3, num_layers)
            self.conv = torch.nn.Sequential(
                # pw
                conv1,
                init_normalization(hidden_dim, norm_type),
                torch.nn.ReLU(inplace=True),
                # dw
                conv2,
                init_normalization(hidden_dim, norm_type),
                torch.nn.ReLU(inplace=True),
                # pw-linear
                conv3,
                init_normalization(out_channels, norm_type)
            )
            if norm_type != "none":
                torch.nn.init.constant_(self.conv[-1].weight, 0)

    def forward(self, x):
        if self.downsample:
            identity = self.downsample(x)
        else:
            identity = x
        if self.training and np.random.uniform() < self.drop_prob:  # dropout the out of conv layer
            return identity
        else:
            return identity + self.conv(x)


# residual block without depth wise group kernel
class Residual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, groups=1)


# a residual conv network using inverted residual layer
# without pooling and global avg pooling
# it could become deeper by adjust the blocks_per_grop or expand_ratio param
class ResnetCNN(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 depths=(32, 64, 64),  # (48, 96, 96) for large scale
                 strides=(3, 2, 2),  # down_sampling stride for the down_sampling layer at the input every group
                 blocks_per_group=3,  # equal to res18 block_per_group*len(depths)*2=18, 5 blocks for large
                 expand_ratio=2,
                 norm_type="bn",
                 resblock=InvertedResidual,
                 ):  # 4 expand_ratio for large
        super(ResnetCNN, self).__init__()
        self.strides = strides
        self.depths = (input_channels, ) + depths  # output channels after every group
        self.resblock = resblock
        self.expand_ratio = expand_ratio
        self.blocks_per_group = blocks_per_group
        self.layers = []
        self.norm_type = norm_type
        self.num_layers = self.blocks_per_group*len(depths)
        # make grop consist of blocks
        for i in range(len(depths)):
            self.layers.append(self._make_layer(self.depths[i],
                                                self.depths[i+1],
                                                strides[i],
                                                ))
        self.layers = torch.nn.Sequential(*self.layers)

    def _make_layer(self, in_channels, depth, stride,):
        # make the first block if is needed for down_sampling
        blocks = [self.resblock(in_channels, depth,
                                expand_ratio=self.expand_ratio,
                                stride=stride,
                                norm_type=self.norm_type,
                                num_layers=self.num_layers,)]

        for i in range(1, self.blocks_per_group):
            blocks.append(self.resblock(depth, depth,
                                        expand_ratio=self.expand_ratio,
                                        stride=1,
                                        norm_type=self.norm_type,
                                        num_layers=self.num_layers,))

        return torch.nn.Sequential(*blocks)

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs):
        return self.layers(inputs)

    def output_shape(self, h, w, c=None):
        for stride in self.strides:
            try:
                h, w = conv2d_output_shape(h, w, stride, stride, 0)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = self.depths[-1]
            except AttributeError:
                pass  # Not a conv layer.
        return c, h, w

    def output_size(self, h, w, c=None):
        c, h, w = self.output_shape(h, w, c)
        return c * h * w


if __name__ == '__main__':
    convmodel = ResnetCNN(input_channels=3)
    x = torch.randn((1, 3, 84, 84))
    out_put = convmodel(x)
    print(out_put.shape)
    output_shape = convmodel.output_shape(84, 84, 3)
    print(output_shape)
