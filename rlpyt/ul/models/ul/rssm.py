import torch
import torch.nn as nn
from torch import Tensor
from rlpyt.ul.models.ul.gru_cells import GRUCellStack
import torch.nn.functional as F
import torch.distributions as D


class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, device,
                 stoch_dim, stoch_discrete, latent_dim,
                 num_gru_layers, gru_type, layer_norm=True,
                 ):
        super().__init__()
        self.device = device
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.deter_dim = deter_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        # recurrent model input preprocess
        self.z_proj = nn.Linear(stoch_dim * (stoch_discrete or 1), latent_dim)
        self.a_proj = nn.Linear(action_dim, latent_dim)
        self.in_norm = norm(latent_dim, eps=1e-3)

        self.gru_cell = GRUCellStack(latent_dim, deter_dim, num_gru_layers, gru_type)

        # posterior model (representation model)
        self.post_proj_h = nn.Linear(deter_dim, latent_dim)
        self.post_proj_embed = nn.Linear(embed_dim, latent_dim, bias=False)
        self.post_norm = NoNorm(latent_dim, eps=1e-3)
        self.post_encoder = nn.Linear(latent_dim, stoch_dim * (stoch_discrete or 2))

        # prior model (transition predictor model)
        self.prior_proj_h = nn.Linear(deter_dim, latent_dim)
        self.prior_norm = norm(latent_dim, eps=1e-3)
        self.prior_predictor = nn.Linear(latent_dim, stoch_dim * (stoch_discrete or 2))

    def init_state(self, batch_size):
        return (
            torch.zeros((batch_size, self.deter_dim), device=self.device),
            torch.zeros((batch_size, self.stoch_dim * (self.stoch_discrete or 1)), device=self.device)
        )

    def forward(self,
                embed,
                action,
                in_state):
        h_prev, z_prev = in_state
        B = action.shape[0]

        # recurrent model forward
        x = self.z_proj(z_prev) + self.a_proj(action)
        x = self.in_norm(x)
        z_a = F.elu(x)
        h = self.gru_cell(z_a, h_prev)
        # representation model forward
        x = self.post_proj_h(h) + self.post_proj_embed(embed)
        x = self.post_norm(x)
        x = F.elu(x)
        post = self.post_encoder(x)  # posterior logits
        post_distr = self.zdistr(post)
        z_repre = post_distr.rsample().reshape(B, -1)  # sample as z then feed into model

        return post, (h, z_repre)

    def forward_pred(self,
                     action,
                     in_state):
        h_prev, z_prev = in_state
        B = action.shape[0]
        # recurrent deterministic path
        x = self.z_proj(z_prev) + self.a_proj(action)
        x = self.in_norm(x)
        z_a = F.elu(x)
        h = self.gru_cell(z_a, h_prev)
        # transition predictor path (priors path)
        x = self.prior_proj_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_predictor(x)  # reparameter trick and sample
        prior_distr = self.zdistr(prior)
        z_pred = prior_distr.rsample().reshape(B, -1)

        return prior, (h, z_pred)

    # transition predictor model
    def batch_prior(self, h_prev):
        x = self.prior_proj_h(h_prev)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_predictor(x)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP
            distr = D.independent.Independent(distr, 1)  # This makes d.entropy() and d.kl() sum over stoch_dim
            return distr
        else:
            return diag_normal(pp)


class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


def diag_normal(x, min_std=0.1, max_std=2.0):
    mean, std = x.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


class RSSMCore(nn.Module):
    def __init__(self, embed_dim, action_dim, deter_dim, device,
                 stoch_dim, stoch_discrete, latent_dim, warmup_T,
                 num_gru_layers, gru_type, layer_norm=True):
        super().__init__()
        self.cell = RSSMCell(embed_dim, action_dim, deter_dim, device,
                             stoch_dim, stoch_discrete, latent_dim,
                             num_gru_layers, gru_type, layer_norm)
        self.warmup_T = warmup_T
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim * (stoch_discrete or 1)
        self.feat_dim = self.deter_dim + self.stoch_dim

    def forward(self, embeds, actions, in_state, forward_pred=False):
        T, B = embeds.shape[:2]
        posts = []
        states_h = []
        samples = []
        (h_prev, z_prev) = in_state
        # warmup the forward model
        for i in range(self.warmup_T):
            post, (h_prev, z_prev) = self.cell.forward(embeds[i].squeeze(), actions[i].squeeze(), (h_prev, z_prev))
            posts.append(post)  # real posterior of the representation model

            states_h.append(h_prev)
            samples.append(z_prev)
        # calculate the forward pred or full agg
        for i in range(self.warmup_T, T):
            if not forward_pred:
                post, (h_prev, z_prev) = self.cell.forward(embeds[i].squeeze(), actions[i].squeeze(), (h_prev, z_prev))
                posts.append(post)  # real posterior of the representation model
            else:
                prior, (h_prev, z_prev) = self.cell.forward_pred(actions[i].squeeze(), (h_prev, z_prev))
                posts.append(prior)  # fake posterior of transition predictor model

            states_h.append(h_prev)  # h_full for not forward pred else h_partial
            samples.append(z_prev)  # z_repre for not forward pred else z_pred

        posts = torch.stack(posts)
        states_h = torch.stack(states_h)
        priors = self.cell.batch_prior(states_h)
        samples = torch.stack(samples)
        features = torch.cat((states_h, samples), -1)
        # states = (states_h, samples)

        return posts, states_h, samples, features, priors

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)
