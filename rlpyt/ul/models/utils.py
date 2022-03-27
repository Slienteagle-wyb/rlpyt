import torch


def init_normalization(channels, type_id="bn", affine=True, one_d=False):
    assert type_id in ["bn", "ln", "in", "gn", "max", "none", None]
    if type_id == "bn":
        if one_d:
            return torch.nn.BatchNorm1d(channels, affine=affine)
        else:
            return torch.nn.BatchNorm2d(channels, affine=affine)
    elif type_id == "ln":
        if one_d:
            return torch.nn.LayerNorm(channels, elementwise_affine=affine)
        else:
            return torch.nn.GroupNorm(1, channels, affine=affine)
    elif type_id == "in":
        return torch.nn.GroupNorm(channels, channels, affine=affine)
    elif type_id == "gn":
        groups = max(min(32, channels//4), 1)
        return torch.nn.GroupNorm(groups, channels, affine=affine)
    elif type_id == "max":
        if not one_d:
            return renormalize
        else:
            return lambda x: renormalize(x, -1)
    elif type_id == "none" or type_id is None:
        return torch.nn.Identity()


def renormalize(tensor, first_dim=-3):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)