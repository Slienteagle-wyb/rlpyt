import copy
configs = dict()

config = dict(
    algo=dict(
        batch_B=16,  # batch_size X time (16, 32)
        batch_T=32,  # batch_T = warm_up(16) + contrast_rollout(16)
        warmup_T=16,
        rnn_size=256,
        latent_size=256,
        hidden_sizes=512,
        spr_loss_coefficient=1.0,
        contrast_loss_coefficient=0.2,
        clip_grad_norm=100.0,
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        kaiming_init=True,
    ),
    optim=dict(
        optim_id='adamw',
        lr=1e-3,
        weight_decay=0.05,
        skip_list=None,
        eps=1e-8,
        betas=[0.9, 0.99],
        momentum=0.9,
    ),
    sched=dict(
        sched_id='cosine',
        min_lr=1e-6,
        warmup_lr_init=1e-5,
        warmup_epochs=150,
        lr_k_decay=1.0,
    ),
    runner=dict(
        n_epochs=int(1500),  # epoch==200 updates now
        log_interval_updates=int(1e3),
        wandb_log=False,
        wandb_log_name=None,
        snapshot_gap_intervals=40,  # the save interval factor
    ),
    replay=dict(
            img_size=84,
            frame_stacks=1,  # the dim of F channel for the extracted batch
            # f'/root/data/drone_repr' for cloud
            data_path=f'/home/yibo/spaces/datasets/drone_repr',
            episode_length=2000,  # the length of T idx for the dataset replay
            num_runs=50,  # the dim of batch_idx for dataset replay
            forward_step=31,  # the forward step for extracting batch, total extracted batch_T = 1 + forward_step
            translation_dim=3,
            rotation_dim=6,
        ),
    name="drone_stc",  # probably change this with the filepath
)

configs["drone_stc"] = config
