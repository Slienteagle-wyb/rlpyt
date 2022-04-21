import torch
from rlpyt.ul.models.ul.atc_models import ByolMlpModel


class LstmPolicyModel(torch.nn.Module):
    def __init__(self,
                 conv_output_size,
                 latent_size,
                 action_dim,
                 hidden_sizes,
                 state_latent_size,
                 rnn_horizon,
                 train_batch,
                 num_layers=2,
                 ):
        super(LstmPolicyModel, self).__init__()
        self.feature_head = ByolMlpModel(
            input_dim=conv_output_size,
            latent_size=latent_size,
            hidden_size=hidden_sizes
        )
        self.lstm = torch.nn.LSTM(
            input_size=latent_size + state_latent_size,
            hidden_size=hidden_sizes,
            num_layers=num_layers
        )
        self.conv_output_size = conv_output_size
        self.policy_head = torch.nn.Linear(hidden_sizes, action_dim)
        self.rnn_horizon = rnn_horizon
        self.num_layers = num_layers
        self.train_batch = train_batch
        self.rnn_hidden_size = hidden_sizes
        self.rnn_hiddens_states = None
        self.rnn_counter = 0

    def forward(self, conv_features, state_embedding):
        T, B, feature_dim = state_embedding.shape
        conv_features = conv_features.reshape(T*B, -1)
        conv_latent = self.feature_head(conv_features)
        conv_latent = conv_latent.reshape(T, B, -1)
        rnn_input = torch.cat([conv_latent, state_embedding], dim=2)
        context, (hn, cn) = self.lstm(rnn_input, self.rnn_hiddens_states)  # context:[T, B, latent_dim], hn, cn:[num_layers, B, latent_dim]
        acts = self.policy_head(context)  # return act shape [T, B, act_dim]
        return acts, (hn, cn)

    def get_action(self, conv_features, state_embedding):
        assert not self.feature_head.training
        device = conv_features.device
        if self.rnn_hiddens_states is None or self.rnn_counter % self.rnn_horizon == 0:
            self.rnn_hiddens_states = self.get_rnn_init_hiddens(self.train_batch, device)
            self.rnn_counter = 0
        self.rnn_counter += 1
        assert len(state_embedding.shape) == 2
        conv_latent = self.feature_head(conv_features.reshape(1, -1))
        rnn_input = torch.cat([conv_latent, state_embedding], dim=-1)  # rnn_input:[B, latent_size + state_embedding_dim]
        context, self.rnn_hiddens_states = self.lstm(rnn_input.unsqueeze(0))
        act = self.policy_head(context).squeeze()
        return act

    def get_rnn_init_hiddens(self, batch_size, device):
        h_0 = torch.zeros((self.num_layers, batch_size, self.rnn_hidden_size), device=device)
        c_0 = torch.zeros((self.num_layers, batch_size, self.rnn_hidden_size), device=device)
        return h_0, c_0


if __name__ == '__main__':
    conv_features = torch.randn((16, 32, 64, 7, 7))
    prev_actions = torch.randn((16, 32, 256))
    agent = LstmPolicyModel(
        conv_output_size=int(64*7*7),
        latent_size=256,
        action_dim=4,
        hidden_sizes=512,
        state_latent_size=256,
        rnn_horizon=32,
        train_batch=16,
    )
    action_logits, hidden_state = agent(conv_features, prev_actions)
    print(action_logits.shape)

