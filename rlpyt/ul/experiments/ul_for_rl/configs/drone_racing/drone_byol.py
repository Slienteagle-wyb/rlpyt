
configs = dict()

config = dict(
    algo=dict(
        delta_T=0,
        batch_T=1,
        batch_B=512,
        learning_rate=0.4,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=1000,  # number of updates about 10 epoch
        clip_grad_norm=10.,
        target_update_tau=0.004,   # 1 for hard update
        target_update_interval=1,  # 1 means that using moving average update
        proj_latent_size=256,
        predict_hidden_size=2048,
        validation_split=0.0,
        n_validation_batches=0,  # usually don't do it.
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        hidden_sizes=2048,
        kaiming_init=True,
    ),
    optim=dict(
        weight_decay=1.5e-6,
        skip_list=['head.net.1.weight',
                   'head.net.1.bias',
                   'net.1.weight',
                   'net.1.bias',
                   ]
    ),
    runner=dict(
        n_updates=int(1e5),  # 100k
        log_interval_updates=int(1e4),
        wandb_log=True,
    ),
    replay=dict(
        img_size=84,
        frame_stacks=1,  # the dim of F channel for the extracted batch
        data_path='/home/yibo/Downloads/airsim_datasets/soccer_50k/soccer_close_50k',
        episode_length=50000,  # the length of T idx for the dataset replay
        num_runs=1,  # the dim of batch_idx for dataset replay
        forward_step=0,  # the forward step for extracting batch
        translation_dim=1,
        rotation_dim=3,
    ),
    name="drone_byol",  # probably change this with the filepath
)

configs["drone_byol"] = config
