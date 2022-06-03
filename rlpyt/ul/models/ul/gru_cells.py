import torch
import torch.nn as nn
from torch import Tensor


class GRUCellStack(nn.Module):
    """Multi-layer stack of GRU cells"""

    def __init__(self, input_size, hidden_size, num_layers, cell_type):
        super().__init__()
        self.num_layers = num_layers
        layer_size = hidden_size // num_layers
        assert layer_size * num_layers == hidden_size, "Must be divisible"
        if cell_type == 'gru':
            cell = nn.GRUCell
        elif cell_type == 'gru_layernorm':
            cell = NormGRUCell
        elif cell_type == 'gru_layernorm_dv2':
            cell = NormGRUCellLateReset
        else:
            assert False, f'Unknown cell type {cell_type}'
        layers = [cell(input_size, layer_size)]
        layers.extend([cell(layer_size, layer_size) for _ in range(num_layers - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        input_states = state.chunk(self.num_layers, -1)  # split the tensor into n chunks at dim=-1
        output_states = []
        x = input
        for i in range(self.num_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)


class NormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_update = nn.LayerNorm(hidden_size, eps=1e-3)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=1e-3)

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)

        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h


class NormGRUCellLateReset(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.lnorm = nn.LayerNorm(3 * hidden_size, eps=1e-3)
        self.update_bias = -1

    def forward(self, input: Tensor, state: Tensor) -> Tensor:
        gates = self.weight_ih(input) + self.weight_hh(state)
        gates = self.lnorm(gates)
        reset, update, newval = gates.chunk(3, 1)

        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update + self.update_bias)
        newval = torch.tanh(reset * newval)  # late reset, diff from normal GRU
        h = update * newval + (1 - update) * state
        return h