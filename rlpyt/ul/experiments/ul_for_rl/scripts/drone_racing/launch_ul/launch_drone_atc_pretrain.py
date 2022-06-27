import sys
from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

args = sys.argv[1:]
assert len(args) == 2 or len(args) == 0
if len(args) == 0:
    my_computer, num_computers = 0, 1
else:
    my_computer = int(args[0])
    num_computers = int(args[1])

print(f"MY_COMPUTER: {my_computer},  NUM_COMPUTERS: {num_computers}")
script = "rlpyt/ul/experiments/ul_for_rl/scripts/drone_racing/train_ul/drone_atc.py"
affinity_code = quick_affinity_code(contexts_per_gpu=1)
runs_per_setting = 1
experiment_title = "atc_pretrain"

# make a varent of runs
keys = [('algo', 'delta_T'), ('runner', 'wandb_log_name')]
values = [[3, 'atc_0627_mix_res_delta_3'], ]
dir_names = ['atc_0627_run1']


variant_levles = list()
variant_levles.append(VariantLevel(keys, values, dir_names))
variants, log_dirs = make_variants(*variant_levles)
# variants_2, log_dirs_2 = make_variants(*variant_levels_2)


num_variants = len(variants)
variants_per = num_variants // num_computers

my_start = my_computer * variants_per
if my_computer == num_computers - 1:
    my_end = num_variants
else:
    my_end = (my_computer + 1) * variants_per
my_variants = variants[my_start:my_end]
my_log_dirs = log_dirs[my_start:my_end]
print(my_log_dirs, my_variants)

default_config_key = "drone_atc"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)