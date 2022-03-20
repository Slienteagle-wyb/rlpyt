import copy

configs = dict()

config = dict(
    algo=dict(
        delta_T=0,  # forward predict step
        batch_T=1,
        batch_B=64,
        learning_rate=5e-4,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=200,  # number of itrs
        latent_size=256,
        mlp_hidden_layers=[128, 64, 16],
        action_dim=4,
        clip_grad_norm=10.,
        validation_split=0.0,
        with_validation=True,
        n_validation_itrs=10,
        state_dict_filename=f'/home/yibo/Documents/rlpyt/data/local/20220316/194957/stc_pretrain/itr_199999.pkl',
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        hidden_sizes=512,  # 512 for atc and 2048 for byol
        kaiming_init=True,
    ),
    optim=dict(
        weight_decay=1.5e-6,
    ),
    runner=dict(
        n_updates=int(2e4),  # 10k iters counted by num of batch
        log_interval_updates=int(2e3),
        wandb_log=True,
        wandb_log_name=None,
    ),
    train_replay=dict(
        img_size=84,
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/il_datasets',
        episode_length=4000,
        num_runs=3,
        forward_step=0,
        translation_dim=3,
        rotation_dim=6,
    ),
    val_replay=dict(
            img_size=84,
            frame_stacks=1,
            data_path='/home/yibo/spaces/datasets/il_val_datasets',
            episode_length=4000,
            num_runs=1,
            forward_step=0,
            translation_dim=3,
            rotation_dim=6,
        ),
    name="vel_regressor",  # probably change this with the filepath
)

configs["vel_regressor"] = config
