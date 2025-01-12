configs = dict()

config = dict(
    algo=dict(
        batch_B=24,  # batch_size X time (16, 32)
        batch_T=40,  # batch_T = warm_up(16) + contrast_rollout(16)
        warmup_T=0,  # attention that the warmup_T is not influenced by num of stacked img input
        overshot_horizon=15,
        latent_size=256,
        hidden_sizes=512,
        # stoch_dim=32,
        # stoch_discrete=32,
        deter_dim=512,
        num_stacked_input=1,
        target_update_tau=0.01,
        augmentations=('blur', 'intensity'),
        spatial_coefficient=1.0,
        temporal_coefficient=1.0,
        clip_grad_norm=5.,
    ),
    # encoder configs of dm_lab encoder
    # encoder=dict(
    #     use_fourth_layer=True,
    #     skip_connections=True,
    #     kaiming_init=True,
    # ),
    # encoder config of resnet style encoder
    encoder=dict(
        res_depths=(64, 128, 256),
        downsampling_strides=(3, 2, 2),
        blocks_per_group=3,
        expand_ratio=2
    ),
    optim=dict(
        optim_id='adamw',
        lr=5e-4,
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
            warmup_epochs=100,
            lr_k_decay=1.0,
            lr_cycle_decay=0.2,
            lr_cycle_limit=2,
        ),
    rssm=dict(
        hidden_sizes=512,
        linear_dyna=True,
        num_gru_layers=1,
        gru_type='gru',
        batch_norm=True
    ),
    runner=dict(
        n_epochs=int(1000),  # base_n_epoch=1000
        log_interval_updates=int(1e3),
        wandb_log=True,
        wandb_log_name=None,
        snapshot_gap_intervals=40,  # the save interval factor(40 * 1k)
    ),
    replay=dict(
            img_size=84,
            frame_stacks=1,  # the dim of F channel for the extracted batch
            data_path=f'/root/autodl-tmp/cross_domain',
            episode_length=496,  # the length of T idx for the dataset replay
            num_runs=250,  # the dim of batch_idx for dataset replay (250) is full
            forward_step=39,  # the forward step for extracting batch, total extracted batch_T = 1 + forward_step
            translation_dim=3,
            rotation_dim=6,
            command_catgorical=8,
            normalized_img=True
        ),
    name="drone_mst",  # probably change this with the filepath
)

configs["drone_mst"] = config
