import torch
import torch.nn as nn
from rlpyt.ul.models.ul.gru_cells import GRUCellStack
from rlpyt.models.mlp import MlpModel
import torch.nn.functional as F


class DRnnCell(nn.Module):
    def __init__(self, embed_dim, action_dim, deter_dim, latent_dim,
                 device, num_gru_layers, gru_type, batch_norm,
                 hidden_sizes, linear_dyna=True):
        super(DRnnCell, self).__init__()
        self.device = device
        self.deter_dim = deter_dim
        norm = nn.LayerNorm if batch_norm else NoNorm

        self.embed_proj = nn.Linear(embed_dim, latent_dim, bias=False)
        self.a_proj = nn.Linear(action_dim, latent_dim)
        if linear_dyna:
            # linear latent forward dyna
            self.h_proj = nn.Linear(deter_dim, deter_dim)
        else:
            # no_linear latent forward dyna
            self.h_proj = MlpModel(
                input_size=self.deter_dim,
                hidden_sizes=hidden_sizes,
                output_size=deter_dim
            )
        self.in_norm = norm(latent_dim, eps=1e-3)
        self.h_norm = norm(latent_dim, eps=1e-3)

        self.gru_cell = GRUCellStack(latent_dim, deter_dim, num_gru_layers, gru_type)

    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.deter_dim), device=self.device)

    def forward(self, embed, action, init_state):
        h_prev = self.h_proj(init_state)
        x = self.embed_proj(embed) + self.a_proj(action)
        x = self.in_norm(x)
        x = F.elu(x)
        h = self.gru_cell(x, h_prev)
        return h

    def forward_pred(self, action, init_state):
        h_prev = self.h_proj(init_state)
        action = self.a_proj(action)
        a = self.in_norm(action)
        a = F.elu(a)
        h = self.gru_cell(a, h_prev)
        return h


class DRnnCore(nn.Module):
    def __init__(self, embed_dim, action_dim, deter_dim, latent_dim,
                 device, num_gru_layers, gru_type, warmup_T, batch_norm,
                 hidden_sizes, linear_dyna=True):
        super(DRnnCore, self).__init__()
        self.open_loop_cell = DRnnCell(embed_dim, action_dim, deter_dim, latent_dim,
                                       device, num_gru_layers, gru_type, batch_norm,
                                       hidden_sizes, linear_dyna)
        self.closed_loop_cell = DRnnCell(embed_dim, action_dim, deter_dim, latent_dim,
                                         device, num_gru_layers, gru_type, batch_norm,
                                         hidden_sizes, linear_dyna)
        self.warmup_T = warmup_T
        self.deter_dim = deter_dim

    def forward(self, embeds, actions, in_states, forward_pred=False):
        T, B = embeds.shape[:2]
        h_prev = in_states
        states_h = []
        for i in range(self.warmup_T):
            h_prev = self.closed_loop_cell.forward(embeds[i].squeeze(), actions[i].squeeze(), h_prev)
            states_h.append(h_prev)
        for i in range(self.warmup_T, T):
            if not forward_pred:
                h_prev = self.closed_loop_cell.forward(embeds[i].squeeze(), actions[i].squeeze(), h_prev)
                states_h.append(h_prev)
            else:
                h_prev = self.open_loop_cell.forward_pred(actions[i].squeeze(), in_states)
        states_h = torch.stack(states_h)
        return states_h

    def forward_imagine(self, actions, init_states):
        T, B = actions.shape[:2]
        h_prev = init_states
        states_h = []
        for i in range(T):
            h_prev = self.open_loop_cell.forward_pred(actions[i].squeeze(), h_prev)
            states_h.append(h_prev)
        states_h = torch.stack(states_h)
        return states_h

    def init_state(self, batch_size):
        return self.closed_loop_cell.init_state(batch_size)


class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
