""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler


def create_scheduler(optimizer, num_epochs, sched_kwargs):

    if getattr(sched_kwargs, 'lr_noise', None) is not None:
        lr_noise = getattr(sched_kwargs, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(sched_kwargs, 'lr_noise_pct', 0.67),
        noise_std=getattr(sched_kwargs, 'lr_noise_std', 1.),
        noise_seed=getattr(sched_kwargs, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(sched_kwargs, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(sched_kwargs, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(sched_kwargs, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    sched_id = sched_kwargs['sched_id']
    if sched_id == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=sched_kwargs['min_lr'],
            warmup_lr_init=sched_kwargs['warmup_lr_init'],
            warmup_t=sched_kwargs['warmup_epochs'],
            k_decay=getattr(sched_kwargs, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        # num_epochs = lr_scheduler.get_cycle_length() + sched_kwargs['cooldown_epochs']
    elif sched_id == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=sched_kwargs['min_lr'],
            warmup_lr_init=sched_kwargs['warmup_lr_init'],
            warmup_t=sched_kwargs['warmup_epochs'],
            t_in_epochs=True,
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + sched_kwargs['cooldown_epochs']
    elif sched_id == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=sched_kwargs['decay_epochs'],
            decay_rate=sched_kwargs['decay_rate'],
            warmup_lr_init=sched_kwargs['warmup_lr_init'],
            warmup_t=sched_kwargs['warmup_epochs'],
            **noise_args,
        )
    elif sched_id == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=sched_kwargs['decay_epochs'],
            decay_rate=sched_kwargs['decay_rate'],
            warmup_lr_init=sched_kwargs['warmup_lr_init'],
            warmup_t=sched_kwargs['warmup_epochs'],
            **noise_args,
        )
    elif sched_id == 'plateau':
        mode = 'min' if 'loss' in getattr(sched_kwargs, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=sched_kwargs['decay_rate'],
            patience_t=sched_kwargs['patience_epochs'],
            lr_min=sched_kwargs['min_lr'],
            mode=mode,
            warmup_lr_init=sched_kwargs['warmup_lr_init'],
            warmup_t=sched_kwargs['warmup_epochs'],
            cooldown_t=0,
            **noise_args,
        )
    elif sched_id == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=sched_kwargs['decay_rate'],  # overloading 'decay_rate' as polynomial power
            t_initial=num_epochs,
            lr_min=sched_kwargs['min_lr'],
            warmup_lr_init=sched_kwargs['warmup_lr_init'],
            warmup_t=sched_kwargs['warmup_epochs'],
            k_decay=getattr(sched_kwargs, 'lr_k_decay', 1.0),
            **cycle_args,
            **noise_args,
        )
        num_epochs = lr_scheduler.get_cycle_length() + sched_kwargs['cooldown_epochs']

    return lr_scheduler, num_epochs