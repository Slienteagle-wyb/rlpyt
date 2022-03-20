from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=6,  # 6 for local
    n_gpu=1,
    contexts_per_gpu=1,  # How many experiment to share each GPU
)
print(affinity_code)
script = "rlpyt/ul/experiments/ul_for_rl/scripts/drone_racing/train_ul/drone_stc.py"
runs_per_setting = 1
experiment_title = "stc_pretrain"
default_config_key = "drone_stc"
variant_levles = list()

# make a varent of runs
keys = [('optim', 'lr'), ]
values = [[1e-3], ]
dir_names = ['stc_0319_cloud_run1', ]
variant_levles.append(VariantLevel(keys, values, dir_names))

keys = [('algo', 'spr_loss_coefficient'), ('algo', 'contrast_loss_coefficient'),
        ('runner', 'wandb_log_name')]
# values = [[1.0, 1.0, 'stc_dmnet_0318_spr1'], [2.0, 1.0, 'stc_dmnet_0318_spr2'],
#           [3.0, 1.0, 'stc_dmnet_0318_spr3'], [4.0, 1.0, 'stc_dmnet_0318_spr4']]
# dir_names = ['stc_dmnet_0318_spr1', 'stc_dmnet_0318_spr2', 'stc_dmnet_0318_spr3', 'stc_dmnet_0318_spr4']
values = [[1.0, 0.2, 'stc_dmnet_0319l_spr1_0.2'], ]
dir_names = ['stc_dmnetl_0319l_spr1_0.2', ]
variant_levles.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levles)

print(log_dirs, variants)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
