from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=16,  # 6 for locals
    n_gpu=1,
    contexts_per_gpu=1,  # How many experiment to share each GPU
)
print(affinity_code)
script = 'rlpyt/ul/experiments/ul_for_rl/scripts/drone_racing/train_ul/drone_mstc.py'
runs_per_setting = 1
experiment_title = "mstc_pretrain"
variant_levels = list()

# make a variant of runs
keys = [('algo', 'kl_coefficient'), ('algo', 'spatial_coefficient'),
        ('algo', 'temporal_coefficient'), ('runner', 'wandb_log_name')]
values = [[1.0, 1.0, 1.0, 'mst_0614_mix_res_rssm_contrast_gaussian'], ]
dir_names = ['mstc_0614_run']
variant_levels.append(VariantLevel(keys, values, dir_names))


variants, log_dirs = make_variants(*variant_levels)

default_config_key = "drone_mstc"
run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
