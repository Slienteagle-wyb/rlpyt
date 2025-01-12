import torch
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims


def weight_init(m):
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class ContrastModel(torch.nn.Module):

    def __init__(self, latent_size, anchor_hidden_sizes):
        super().__init__()
        if anchor_hidden_sizes is not None:
            self.anchor_mlp = MlpModel(
                input_size=latent_size,
                hidden_sizes=anchor_hidden_sizes,
                output_size=latent_size,
            )
        else:
            self.anchor_mlp = None
        self.W = torch.nn.Linear(latent_size, latent_size, bias=False)

    def forward(self, anchor, positive):
        lead_dim, T, B, _ = infer_leading_dims(anchor, 1)
        assert lead_dim == 1  # Assume [B,C] shape
        if self.anchor_mlp is not None:
            anchor = anchor + self.anchor_mlp(anchor)  # skip probably helps
        pred = self.W(anchor)
        logits = torch.matmul(pred, positive.T)  # logit * length of batch
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # normalize: for every logit: logit-max(logit feature)
        return logits


class ByolMlpModel(torch.nn.Module):
    def __init__(self, input_dim, latent_size, hidden_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, latent_size)
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.net(x)


class DroneStateProj(torch.nn.Module):
    def __init__(self, input_dim, latent_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, latent_size),
            torch.nn.LayerNorm(latent_size, eps=1e-3),
            torch.nn.ELU()
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = ByolMlpModel(2048, 256, 6096)
    for params in model.named_parameters():
        print(params[0])
