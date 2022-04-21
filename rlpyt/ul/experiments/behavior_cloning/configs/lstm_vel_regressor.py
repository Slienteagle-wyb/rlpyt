import copy
configs = dict()

config = dict(
    algo=dict(
        delta_T=0,  # forward predict step
        batch_T=32,
        batch_B=16,
        warmup_T=0,
        latent_size=256,
        hidden_sizes=512,
        lstm_layers=2,
        action_dim=4,
        attitude_dim=9,
        clip_grad_norm=10.,
        validation_split=0.0,
        with_validation=True,
        state_dict_filename=f'/home/yibo/spaces/snap_shots/rlpyt_drone_representation/20220326/192341/mst_pretrain/params.pkl',
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
        n_epochs=int(1000),  # 10k iters counted by num of batch
        log_interval_updates=int(2e3),  # also apply validate every log_interval_update
        wandb_log=True,
        wandb_log_name=None,
    ),
    train_replay=dict(
        img_size=84,
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/il_datasets',
        episode_length=4096,
        num_runs=3,
        forward_step=63,
        translation_dim=3,
        rotation_dim=6,
    ),
    val_replay=dict(
            img_size=84,
            frame_stacks=1,
            data_path='/home/yibo/spaces/datasets/il_val_datasets',
            episode_length=4096,
            num_runs=1,
            forward_step=63,
            translation_dim=3,
            rotation_dim=6,
        ),
    name="lstm_vel_regressor",  # probably change this with the filepath
)

configs["lstm_vel_regressor"] = config
