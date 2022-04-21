import torch.nn
from rlpyt.ul.models.ul.encoders import DmlabConv2dModelBn
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class DroneSacConvEncoder(torch.nn.Module):
    def __init__(self,
                 image_shape=None,
                 use_fourth_layer=True,
                 skip_connections=True,
                 use_maxpool=False,
                 norm_type='bn',
                 ):
        super(DroneSacConvEncoder, self).__init__()
        self.conv = DmlabConv2dModelBn(
            image_shape[0],
            use_fourth_layer=use_fourth_layer,
            skip_connections=skip_connections,
            use_maxpool=use_maxpool,
            norm_type=norm_type,
        )
        c, h, w = image_shape
        self._output_shape = self.conv.output_shape(h=h, w=w, c=c)
        self._output_size = self.conv.output_size(h=h, w=w, c=c)

    def forward(self, observation):
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation

        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        conv = self.conv(img.view(T * B, *img_shape))
        conv = restore_leading_dims(conv, lead_dim, T, B)
        return conv

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape

