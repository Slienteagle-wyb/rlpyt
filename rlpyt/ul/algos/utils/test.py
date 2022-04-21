import torch.optim.optimizer
from timm.scheduler.cosine_lr import CosineLRScheduler
from rlpyt.ul.models.dmlab_conv2d import DmlabConv2dModelBn
from matplotlib import pyplot as plt


def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch))
    return lr_per_epoch


if __name__ == '__main__':
    model = DmlabConv2dModelBn(in_channels=3,)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-3,
                                )
    num_epoch = 50
    scheduler = CosineLRScheduler(optimizer, t_initial=num_epoch, lr_min=1e-5,
                                  cycle_decay=0.5, cycle_limit=2, cycle_mul=1.0,
                                  warmup_t=5)

    lr_per_epoch = get_lr_per_epoch(scheduler, num_epoch*2)

    plt.plot([i for i in range(num_epoch*2)], lr_per_epoch)
    plt.show()


