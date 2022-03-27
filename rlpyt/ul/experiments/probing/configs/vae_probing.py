
configs = dict()

config = dict(
    algo=dict(
        batch_T=32,
        batch_B=16,
        latent_size=256,
        hidden_sizes=512,
        clip_grad_norm=10.,
        kl_coefficient=1.0,
        validation_split=0.0,
        with_validation=True,
        initial_state_dict=f'/home/yibo/Documents/rlpyt/data/local/20220326/192341/mst_pretrain/params.pkl',
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        kaiming_init=True,
    ),
    decoder=dict(
            reshape=(64, 9, 9),
            channels=(64, 64, 32, 3),
            kernel_sizes=(3, 3, 4, 8),
            strides=(1, 1, 2, 4),
            paddings=(1, 1, 0, 0),
            output_paddings=(0, 0, 0, 0),
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
        wandb_log=False,
        wandb_log_name=None,
    ),
    train_replay=dict(
        img_size=84,
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/drone_repr_body',
        episode_length=1800,
        num_runs=6,
        forward_step=31,
        translation_dim=3,
        rotation_dim=6,
    ),
    val_replay=dict(
            img_size=84,
            frame_stacks=1,
            data_path='/home/yibo/spaces/datasets/drone_repr_body',
            episode_length=1800,
            num_runs=2,
            forward_step=31,
            translation_dim=3,
            rotation_dim=6,
        ),
    name="vae_probing",  # probably change this with the filepath
)

configs["vae_probing"] = config
