configs = dict()

config = dict(
    algo=dict(
        delta_T=3,  # forward predict step
        batch_T=1,
        batch_B=512,
        latent_size=256,
        hidden_sizes=512,
        num_stacked_input=1,
        target_update_tau=0.01,   # 1 for hard update
        target_update_interval=1,
        random_shift_prob=1.,
        random_shift_pad=4,
        clip_grad_norm=10.,
    ),
    encoder=dict(
        res_depths=(64, 128, 256),
        downsampling_strides=(3, 2, 2),
        blocks_per_group=3,
        expand_ratio=2
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
        n_epochs=int(500),  # 10k iters counted by num of batch
        log_interval_updates=int(1e3),
        wandb_log=False,
        wandb_log_name=None,
        snapshot_gap_intervals=40,
    ),
    replay=dict(
        img_size=84,
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/cross_domain',
        episode_length=496,
        num_runs=5,
        forward_step=3,
        translation_dim=3,
        rotation_dim=6,
        command_catgorical=8,
        normalized_img=True
    ),
    name="drone_atc",  # probably change this with the filepath
)

configs["drone_atc"] = config
