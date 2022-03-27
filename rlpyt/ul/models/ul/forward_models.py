import torch
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.ul.models.ul.atc_models import ByolMlpModel


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
        lead_dim, T, B, tensor_shape = infer_leading_dims(x, 1)
        hiddens, _ = self.forward_agg_rnn(x)
        context = self.forward_dyna_head(hiddens.reshape(T * B, -1))
        context = restore_leading_dims(context, lead_dim, T, B)
        return context


class SkipConnectForwardAggModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 latent_size,
                 num_layers=2,
                 skip_connect=True,
                 ):
        super(SkipConnectForwardAggModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        self.skip_connect = skip_connect
        cell_list = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_sizes
            cell = torch.nn.LSTMCell(input_size=input_dim,
                                     hidden_size=hidden_sizes)
            cell_list.append(cell)
        self.cell_list = torch.nn.ModuleList(cell_list)
        self.forward_pred_head = ByolMlpModel(
            input_dim=hidden_sizes,
            latent_size=latent_size,
            hidden_size=hidden_sizes
        )

    def forward(self, x, init_hiddens=None):
        batch_T, batch_B, *spatial_dim = x.size()
        current_layer_input = x
        del x
        init_hiddens_list = []
        for i in range(self.num_layers):
            if i == 0 and init_hiddens is not None:
                init_hiddens_list.append(init_hiddens)
            else:
                init_hiddens_list.append(torch.zeros([batch_B, self.hidden_sizes]).cuda())

        layer_output_list = []
        for idx in range(self.num_layers):
            cell_hidden = init_hiddens_list[idx]
            cell_context = torch.zeros([batch_B, self.hidden_sizes]).cuda()
            layer_hiddens = []
            for t in range(batch_T):
                cell_hidden, cell_context = self.cell_list[idx](current_layer_input[t, :, :],
                                                                (cell_hidden, cell_context))
                layer_hiddens.append(cell_hidden)
            layer_output = torch.stack(layer_hiddens, dim=0)
            current_layer_input = layer_output
            layer_output_list.append(layer_output)
        if self.skip_connect:
            connected_hiddens = torch.stack(layer_output_list, dim=0)
            agg_states = torch.sum(connected_hiddens, dim=0)
        else:
            agg_states = layer_output_list[-1]
        preds = self.forward_pred_head(agg_states.reshape(batch_T*batch_B, -1))
        preds = restore_leading_dims(preds, 2, batch_T, batch_B)
        return preds, agg_states
