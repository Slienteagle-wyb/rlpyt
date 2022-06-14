from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.utils.launching.exp_launcher import run_experiments

# we just have one gpu, so 6cpus assigned to gpu, would set one concurrent experiment
affinity_code = encode_affinity(
    n_cpu_core=6,
    n_gpu=1,
    contexts_per_gpu=2,  # How many experiment to share each GPU
)

print(affinity_code)
script = "rlpyt/ul/experiments/behavior_cloning/scripts/train_bc/train_state_mlp_vel_regressor.py"
runs_per_setting = 1
experiment_title = "mst_state_mlp_vel_regressor"
default_config_key = "state_mlp_vel_regressor"
variant_levles = list()

keys = [('optim', 'lr'), ('runner', 'wandb_log_name'), ('algo', 'state_latent_dim')]
values = [[5e-4, 'mst_0612_mix_res_rssm_dreamer_v2', 256]]
dir_name = ['state_latent_dim_256_nostack']
variant_levles.append(VariantLevel(keys, values, dir_name))

variants, log_dirs = make_variants(*variant_levles)
print(variants, log_dirs)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
