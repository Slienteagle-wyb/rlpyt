
import copy

configs = dict()

config = dict(
    algo=dict(
        delta_T=3,  # forward predict step
        batch_T=1,
        batch_B=512,
        learning_rate=5e-3,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=2000,  # number of updates
        clip_grad_norm=10.,
        target_update_tau=0.01,   # 1 for hard update
        target_update_interval=1,
        latent_size=256,
        anchor_hidden_sizes=512,
        random_shift_prob=1.,
        random_shift_pad=4,
        activation_loss_coefficient=0.,  # rarely if ever use
        validation_split=0.0,
        n_validation_batches=0,  # usually don't do it.
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
        n_updates=int(2e5),  # 10k iters counted by num of batch
        log_interval_updates=int(2e4),
        wandb_log=False,
    ),
    replay=dict(
        img_size=84,
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/drone_repr',
        episode_length=2000,
        num_runs=50,
        forward_step=3,
        translation_dim=3,
        rotation_dim=6,
    ),
    name="drone_atc",  # probably change this with the filepath
)

configs["drone_atc"] = config
