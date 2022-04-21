from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=6,  # 6 for local
    n_gpu=1,
    contexts_per_gpu=1,  # How many experiment to share each GPU
)
print(affinity_code)
script = "rlpyt/ul/experiments/rl_from_ul/scripts/drone_racing/train/drone_sac_from_mst_serial.py"
runs_per_setting = 1
experiment_title = "drone_sac_from_mst"
variant_levels = list()

# make a variant of runs
keys = [('algo', 'reward_scale'), ('algo', 'target_entropy'),
        ('algo', 'alpha_init'), ('runner', 'wandb_log_name')]
values = [[1, 'auto', 0.1, 'sac_rad_fstack3_0414_lr2e-4'], ]
dir_names = ['sac_rad_fstack3_0414_lr2e-4']
variant_levels.append(VariantLevel(keys, values, dir_names))


variants, log_dirs = make_variants(*variant_levels)
default_config_key = "sac_mst_pretrain"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key, experiment_title),
)