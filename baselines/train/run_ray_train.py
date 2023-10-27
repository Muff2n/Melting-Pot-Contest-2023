import argparse
import ray


from typing import *
from ray import tune
from configs import get_experiment_config
from ray.tune import registry
from baselines.train import make_envs

from baselines.train.callbacks import MyCallbacks

def get_cli_args():

  parser = argparse.ArgumentParser(description="Training Script for Multi-Agent RL in Meltingpot")


  parser.add_argument(
      "--num_cpus", type=int, required=True, help="number of CPUs to use")
  parser.add_argument(
      "--num_gpus", type=int, required=True, help="number of GPUs to use")
  parser.add_argument(
      "--n_iterations", type=int, required=True, help="number of training iterations to use")
  parser.add_argument(
        "--framework",
        choices=["tf", "torch"],
        default="torch",
        help="The DL framework specifier (tf2 eager is not supported).",
  )
  parser.add_argument(
      "--exp",
      type=str,
      choices = ['pd_arena','al_harvest','clean_up','territory_rooms'],
      default="pd_arena",
      help="Name of the substrate to run",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=123,
      help="Seed to run",
  )
  parser.add_argument(
      "--results_dir",
      type=str,
      default="./results",
      help="Name of the wandb group",
  )
  parser.add_argument(
        "--logging",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="The level of training and data flow messages to print.",
  )

  parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Whether to use WanDB logging.",
  )

  parser.add_argument(
        "--downsample",
        type=bool,
        default=True,
        help="Whether to downsample substrates in MeltingPot. Defaults to 8.",
  )

  parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test.",
  )

  parser.add_argument(
      "--tmp_dir",
      type=str,
      default=None,
      help="Custom tmp location for temporary ray logs")

  args = parser.parse_args()
  print("Running trails with the following arguments: ", args)
  return args


if __name__ == "__main__":

  args = get_cli_args()

  # Set up Ray. Use local mode for debugging. Ignore reinit error.
  ray.init(
    address="local",
    num_cpus=args.num_cpus,
    num_gpus=args.num_gpus,
    ignore_reinit_error=True,
    # logging_level=args.log,
    _temp_dir=args.tmp_dir
  )

  # Register meltingpot environment
  registry.register_env("meltingpot", make_envs.env_creator)

  # Fetch experiment configurations
  config, run_config, tune_config = get_experiment_config(args)
  # print(config)
  policies = config.multiagent.get("policies", {})
  n = len(policies)

  # TODO: MyCallbacks are putting the reward as a list or scalar, and that is the opposite to what is required
  # MyCallbacks.set_transfer_map({f"policy_{i}": 1 - i/5 for i in range(n)})
  # config = config.callbacks(MyCallbacks)

  # Run Trials
  results = tune.Tuner(
      "PPO",
      param_space=config,
      tune_config=tune_config,
      run_config=run_config,
  ).fit()

  best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
  print(best_result)

  ray.shutdown()
