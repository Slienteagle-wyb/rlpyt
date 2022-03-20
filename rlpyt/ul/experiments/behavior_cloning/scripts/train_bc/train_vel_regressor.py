import sys
import pprint
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.ul.algos.downstreams.bc_vel_regressor import VelRegressBc
from rlpyt.ul.runners.behavior_cloning import BehaviorCloning
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.ul.experiments.behavior_cloning.configs.vel_regressor import configs


def build_and_train(
        slot_affinity_code="0slt_1gpu_1cpu",
        log_dir="test",
        run_ID="0",
        config_key="vel_regressor",
        ):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)  # combine the config made in the launch file variant

    pprint.pprint(config)

    algo = VelRegressBc(
        optim_kwargs=config['optim'],
        encoder_kwargs=config['encoder'],
        train_replay_kwargs=config['train_replay'],
        val_replay_kwargs=config['val_replay'],
        **config['algo']
    )
    runner = BehaviorCloning(
        algo=algo,
        affinity=affinity,
        **config["runner"]
    )
    name = config["name"]
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="all"):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])