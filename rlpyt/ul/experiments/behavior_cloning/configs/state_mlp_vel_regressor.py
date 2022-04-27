import copy
configs = dict()

config = dict(
    algo=dict(
        delta_T=0,  # forward predict step
        batch_T=1,
        batch_B=512,
        latent_size=256,
        hidden_sizes=512,
        mlp_hidden_layers=[128, 64, 16],
        action_dim=4,
        attitude_dim=9,
        state_latent_dim=64,
        clip_grad_norm=10.,
        validation_split=0.0,
        with_validation=True,
        # f'/home/yibo/Documents/rlpyt/data/local/20220423/221519/mst_pretrain/mst_0423_run1/params.pkl'
        state_dict_filename=None,
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
            warmup_epochs=50,
            lr_k_decay=1.0,
        ),
    runner=dict(
        n_epochs=int(1000),
        log_interval_updates=int(2e3),
        wandb_log=True,
        wandb_log_name=None,
    ),
    train_replay=dict(
        img_size=84,
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/il_datasets',
        episode_length=4096,
        num_runs=3,
        forward_step=0,
        translation_dim=3,
        rotation_dim=6,
    ),
    val_replay=dict(
            img_size=84,
            frame_stacks=1,
            data_path='/home/yibo/spaces/datasets/il_val_datasets',
            episode_length=4096,
            num_runs=1,
            forward_step=0,
            translation_dim=3,
            rotation_dim=6,
        ),
    name="state_mlp_vel_regressor",  # probably change this with the filepath
)

configs["state_mlp_vel_regressor"] = config
