import copy

configs = dict()

config = dict(
    algo=dict(
        batch_B=16,  # batch_size X time (16, 32)
        batch_T=32,  # batch_T = warm_up(16) + contrast_rollout(16)
        warmup_T=16,
        learning_rate=2e-3,
        rnn_size=256,  # dim of context
        latent_size=256,
        clip_grad_norm=10.,
        onehot_actions=False,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=2000,  # number of updates
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        hidden_sizes=512,
        kaiming_init=True,
    ),
    optim=dict(
        weight_decay=0,
    ),
    runner=dict(
        n_updates=int(2e5),  # 40k Usually sufficient for one level?
        log_interval_updates=int(2e4),
        wandb_log=True,
    ),
    replay=dict(
            img_size=84,
            frame_stacks=1,  # the dim of F channel for the extracted batch
            data_path=f'/home/yibo/spaces/datasets/drone_repr',
            episode_length=2000,  # the length of T idx for the dataset replay
            num_runs=50,  # the dim of batch_idx for dataset replay
            forward_step=31,  # the forward step for extracting batch, total extracted batch_T = 1 + forward_step
            translation_dim=3,
            rotation_dim=6,
        ),
    name="drone_cpc",  # probably change this with the filepath
)

configs["drone_cpc"] = config
