import sys
import pprint
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.ul.algos.ul_for_rl.mst import DroneMST
from rlpyt.ul.runners.unsupervised_learning import UnsupervisedLearning
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.ul.experiments.ul_for_rl.configs.drone_racing.drone_mst import configs


def build_and_train(
        slot_affinity_code="0slt_1gpu_1cpu",
        log_dir="test",
        run_ID="0",
        config_key="drone_mst",
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)  # combine the config made in the launch file variant

    pprint.pprint(config)

    algo = DroneMST(
        optim_kwargs=config['optim'],
        encoder_kwargs=config['encoder'],
        replay_kwargs=config['replay'],
        sched_kwargs=config['sched'],
        **config['algo']
    )
    runner = UnsupervisedLearning(
        algo=algo,
        config_dict=config,
        affinity=affinity,
        **config["runner"]
    )
    name = config["name"]
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last+gap"):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
