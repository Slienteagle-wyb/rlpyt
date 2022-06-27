import copy
configs = dict()

config = dict(
    algo=dict(
        delta_T=0,  # forward predict step(of no usage)
        batch_T=1,
        batch_B=32,
        latent_size=256,
        hidden_sizes=512,
        num_stacked_input=1,
        mlp_hidden_layers=[128, 64],
        action_dim=4,
        attitude_dim=9,
        state_latent_dim=64,
        clip_grad_norm=10.,
        validation_split=0.0,
        with_validation=True,
        # f'/home/yibo/Documents/rlpyt/data/local/20220623/232057/mst_pretrain/mst_0623_run1/params.pkl'
        state_dict_filename=f'/home/yibo/Documents/rlpyt/data/local/20220623/232057/mst_pretrain/mst_0623_run1/params.pkl'
    ),
    encoder=dict(
        use_fourth_layer=True,
        skip_connections=True,
        kaiming_init=True,
    ),
    # # convnext style encoder params
    # encoder=dict(
    #     res_depths=(64, 128, 256),
    #     downsampling_strides=(3, 2, 2),
    #     blocks_per_group=3,
    #     expand_ratio=2
    # ),
    optim=dict(
        optim_id='adamw',
        lr=1e-4,
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
            warmup_epochs=2,
            lr_k_decay=1.0,
        ),
    runner=dict(
        n_epochs=int(15),
        snapshot_gap_intervals=2,
        wandb_log=True,
        wandb_log_name=None,
    ),
    train_replay=dict(
        img_size=84,  # 96 for res18
        frame_stacks=1,
        data_path='/home/yibo/spaces/datasets/il_datasets',
        episode_length=4096,
        num_runs=3,
        forward_step=0,
        translation_dim=3,
        rotation_dim=6,
        normalized_img=True,
    ),
    val_replay=dict(
            img_size=84,  # 96 for res18
            frame_stacks=1,
            data_path='/home/yibo/spaces/datasets/il_val_datasets',
            episode_length=4096,
            num_runs=1,
            forward_step=0,
            translation_dim=3,
            rotation_dim=6,
            normalized_img=True,
        ),
    name="state_mlp_vel_regressor",  # probably change this with the filepath
)

configs["state_mlp_vel_regressor"] = config
