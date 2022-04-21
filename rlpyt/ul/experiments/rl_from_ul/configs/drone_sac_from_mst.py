configs = dict()

config = dict(
    agent=dict(
        action_squash=1.0,
        pretrain_std=0.75,  # 0.75 gets pretty uniform actions
        load_conv=False,  # load the pretrained encoder head
        load_all=False,  # Just for replay saving.
        state_dict_filename=f'/home/yibo/spaces/snap_shots/rlpyt_drone_representation/20220326/192341/mst_pretrain/params.pkl',
        store_latent=False  # store conv latent when forward inference
    ),
    # conv=dict(
    #     image_shape=(3, 84, 84),
    #     use_fourth_layer=True,
    #     skip_connections=True,
    #     use_maxpool=False,
    #     norm_type='bn'
    # ),
    conv=dict(
        channels=[32, 32, 32, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 1],
        paddings=None,
    ),
    fc1=dict(
        latent_size=50,
        layer_norm=True,
    ),
    pi_model=dict(
        hidden_sizes=[1024, 1024],
        min_log_std=-10,
        max_log_std=2,
    ),
    q_model=dict(hidden_sizes=[1024, 1024]),

    algo=dict(
        discount=0.99,
        batch_size=512,
        # replay_ratio=512,  # data_consumption / data_generation
        min_steps_learn=int(1e4),
        replay_size=int(1e5),
        target_update_tau=0.01,  # tau=1 for hard update.
        target_update_interval=2,
        actor_update_interval=2,
        # OptimCls=torch.optim.Adam,
        initial_optim_state_dict=None,  # for all of them.
        action_prior="uniform",  # or "gaussian"
        reward_scale=1,
        target_entropy="auto",  # "auto", float, or None
        reparameterize=True,
        clip_grad_norm=1e6,
        n_step_return=1,  # just use td-0 for bootstrap.
        # updates_per_sync=1,  # For async mode only.
        bootstrap_timelimit=True,
        # crop_size=84,  # Get from agent.
        q_lr=2e-4,  # lr for q_net and encoder(if stop_conv_grad==False)
        pi_lr=2e-4,  # lr for pi_net
        alpha_lr=1e-4,
        q_beta=0.9,
        pi_beta=0.9,
        alpha_beta=0.5,
        alpha_init=0.1,
        encoder_update_tau=0.05,
        augmentation="random_shift",  # [None, "random_shift", "sub_pixel_shift"]
        random_shift_pad=4,  # how much to pad on each direction (like DrQ style)
        random_shift_prob=1.,
        stop_conv_grad=False,
        max_pixel_shift=1.,
    ),
    # Will use same args for eval env.
    env=dict(
        domain_name="cheetah",
        task_name="run",
        from_pixels=True,
        frame_stack=3,
        frame_skip=4,
        height=84,
        width=84,
    ),
    eval_env=dict(
        domain_name="cheetah",
        task_name="run",
        from_pixels=True,
        frame_stack=3,
        frame_skip=4,
        height=84,
        width=84,
        ),
    optim=dict(),
    runner=dict(
        # Total number of interaction steps to run in training loop. and the env steps is frame_skip * n_steps
        n_steps=1e5,
        log_interval_steps=1e3,
        save_snapshot_factor=10,
        with_wandb_log=True,
        wandb_log_name=None
    ),
    sampler=dict(
        batch_T=1,  # num of transitions sampled every interaction/itr
        batch_B=1,  # num of the serial envs
        max_decorrelation_steps=0,  # max random warm_up steps at the start of training(with a uniform factor)
        eval_n_envs=1,  # same to the definition of batch_B == num of serial eval env
        eval_max_steps=int(10000),
        eval_max_trajectories=10,
    ),
    pretrain=dict(  # Just for logging purposes.
        name=None,
        algo='sac_from_mst',
        n_updates=None,
        log_interval_updates=None,
        learning_rate=None,
        target_update_tau=None,
        batch_B=None,
        batch_T=None,
        warmup_T=None,
        delta_T=None,
        hidden_sizes=None,
        latent_size=None,
        batch_size=None,
        validation_batch_size=None,
        activation_loss_coefficient=None,
        replay=None,
        model_dir=None,
        learning_rate_anneal=None,
        learning_rate_warmup=None,
        weight_decay=None,
        anchor_hidden_sizes=None,
        action_condition=False,
        transform_hidden_sizes=None,
        kiaming_init=True,
        data_aug=None,
        random_shift_prob=None,
        use_global_global=None,
        use_global_local=None,
        use_local_local=None,
        local_conv_layer=None,
    ),
)

configs["sac_mst_pretrain"] = config
