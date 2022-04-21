configs = dict()

config = dict(
    algo=dict(
        batch_B=16,  # batch_size X time (16, 32)
        batch_T=32,  # batch_T = warm_up(16) + contrast_rollout(16)
        warmup_T=16,
        latent_size=256,
        hidden_sizes=512,
        target_update_tau=0.01,
        augmentations=('intensity',),
        spr_loss_coefficient=2.0,
        contrast_loss_coefficient=1.0,
        inverse_dyna_loss_coefficient=1.0,
        clip_grad_norm=10.,
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        kaiming_init=True,
    ),
    optim=dict(
        optim_id='adamw',
        lr=1e-3,
        weight_decay=1e-6,
        skip_list=None,
        eps=1e-8,
        betas=[0.9, 0.99],
        momentum=0.9,  # default, factor of the history update step
    ),
    sched=dict(
            sched_id='cosine',
            min_lr=1e-6,
            sched_slice=2,
            warmup_lr_init=1e-5,
            warmup_epochs=50,
            lr_k_decay=1.0,
            lr_cycle_decay=0.2,
            lr_cycle_limit=2,
        ),
    runner=dict(
        n_epochs=int(1000),  # epoch==200 updates now
        log_interval_updates=int(1e3),
        wandb_log=True,
        wandb_log_name=None,
        snapshot_gap_intervals=40,  # the save interval factor
    ),
    replay=dict(
            img_size=84,
            frame_stacks=1,  # the dim of F channel for the extracted batch
            data_path=f'/home/yibo/spaces/datasets/drone_repr_body',
            episode_length=1800,  # the length of T idx for the dataset replay
            num_runs=60,  # the dim of batch_idx for dataset replay
            forward_step=31,  # the forward step for extracting batch, total extracted batch_T = 1 + forward_step
            translation_dim=3,
            rotation_dim=6,
            command_catgorical=8,
        ),
    name="drone_mst",  # probably change this with the filepath
)

configs["drone_mst"] = config
