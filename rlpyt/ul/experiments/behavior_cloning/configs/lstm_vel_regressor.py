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
        num_stacked_input=1,
        lstm_layers=1,
        action_dim=4,
        attitude_dim=9,
        clip_grad_norm=10.,
        validation_split=0.0,
        with_validation=True,
        state_dict_filename=f'/home/yibo/Documents/rlpyt/data/local/20220501/210122/mst_pretrain/mst_0501_run1/params.pkl',
    ),
    # encoder=dict(
    #     use_fourth_layer=True,
    #     skip_connections=True,
    #     kaiming_init=True,
    # ),
    encoder=dict(
        res_depths=(32, 64, 64),
        downsampling_strides=(3, 2, 2),
        blocks_per_group=3,
        expand_ratio=2
    ),
    optim=dict(
        optim_id='adamw',
        lr=5e-4,
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
        forward_step=31,
        translation_dim=3,
        rotation_dim=6,
        normalized_img=True,
    ),
    val_replay=dict(
            img_size=84,
            frame_stacks=1,
            data_path='/home/yibo/spaces/datasets/il_val_datasets',
            episode_length=4096,
            num_runs=1,
            forward_step=31,
            translation_dim=3,
            rotation_dim=6,
            normalized_img=True,
        ),
    name="lstm_vel_regressor",  # probably change this with the filepath
)

configs["lstm_vel_regressor"] = config
