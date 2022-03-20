import torch
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class ForwardAggRnnModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 latent_size):
        super(ForwardAggRnnModel, self).__init__()
        self.forward_agg_rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=latent_size,
        )
        self.forward_dyna_head = MlpModel(
            input_size=latent_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )

    def forward(self, x):
        lead_dim, T, B, img_shape = infer_leading_dims(x, 1)
        hiddens, _ = self.forward_agg_rnn(x)
        context = self.forward_dyna_head(hiddens.reshape(T * B, -1))
        context = restore_leading_dims(context, lead_dim, T, B)
        return context
