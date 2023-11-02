from random import choice
import os

from meltingpot import substrate
from ray.air import CheckpointConfig, RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy import policy
from ray.tune import TuneConfig

from baselines.train import make_envs

SUPPORTED_SCENARIOS = [
    'allelopathic_harvest__open_0',
    'allelopathic_harvest__open_1',
    'allelopathic_harvest__open_2',
    'clean_up_2',
    'clean_up_3',
    'clean_up_4',
    'clean_up_5',
    'clean_up_6',
    'clean_up_7',
    'clean_up_8',
    'prisoners_dilemma_in_the_matrix__arena_0',
    'prisoners_dilemma_in_the_matrix__arena_1',
    'prisoners_dilemma_in_the_matrix__arena_2',
    'prisoners_dilemma_in_the_matrix__arena_3',
    'prisoners_dilemma_in_the_matrix__arena_4',
    'prisoners_dilemma_in_the_matrix__arena_5',
    'territory__rooms_0',
    'territory__rooms_1',
    'territory__rooms_2',
    'territory__rooms_3',
]

IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']

NUM_ENVS_PER_WORKER = 1
NUM_EPISODES_PER_WORKER = 1

KEEP_CHECKPOINTS_NUM = 1  # Default None
CHECKPOINT_FREQ = 10  # Default 0

VERBOSE = 1

SGD_MINIBATCH_SIZE = 4000
LR = 2e-4  # 2e-4 for 4096
VF_CLIP_PARAM = 2.0
NUM_SGD_ITER = 10
ENTROPY_COEFF = 0.003

def get_experiment_config(args):

    if args.exp == 'pd_arena':
        substrate_name = "prisoners_dilemma_in_the_matrix__arena"
    elif args.exp == 'al_harvest':
        substrate_name = "allelopathic_harvest__open"
        horizon = 2000
        player_roles = substrate.get_config(substrate_name).default_player_roles
        num_players = len(player_roles)
        unique_roles = list(set(player_roles))


        def policy_mapping_fn(aid, *args, **kwargs):
          if aid in [f"player_{i}" for i in range(8)]:
              return unique_roles[0]
          elif aid in [f"player_{i}" for i in range(8, 16)]:
              return unique_roles[1]
          assert False
    elif args.exp == 'clean_up':
        substrate_name = "clean_up"
        horizon = 1000 + (100 / 0.2)
    elif args.exp == 'territory_rooms':
        substrate_name = "territory__rooms"
    else:
        raise Exception("Please set --exp to be one of ['pd_arena', 'al_harvest', 'clean_up', \
                        'territory_rooms']. Other substrates are not supported.")

    num_workers = args.num_cpus - 1

    train_batch_size = max(
      1, num_workers) * NUM_ENVS_PER_WORKER * NUM_EPISODES_PER_WORKER * horizon * num_players

    if args.downsample:
        scale_factor = 8
    else:
        scale_factor = 1

    env_config = {"substrate": substrate_name,
                  "roles": player_roles,
                  "scaled": scale_factor}

    base_env = make_envs.env_creator(env_config)
    rgb_shape = base_env.observation_space["player_0"]["RGB"].shape
    sprite_x = rgb_shape[0]
    sprite_y = rgb_shape[1]

    # Each player needs to have the same
    policies = {}
    for role in unique_roles:
        policies[role] = policy.PolicySpec(
            # policy_class=None,  # use default policy
            observation_space=base_env.observation_space[f"player_0"],
            action_space=base_env.action_space[f"player_0"])


    CUSTOM_MODEL = {
        "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1], [64, [sprite_x, sprite_y], 1]],
        "conv_activation": "relu",
        "post_fcnet_activation": "relu",
        "post_fcnet_hiddens": [64, 64],
        "post_fcnet_activation": "relu",
        "no_final_linear": True,
        # needs vf_loss_coeff to be tuned if True
        "vf_share_layers": True,
        "use_lstm": True,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": True,
        "lstm_use_prev_reward": False,
        # "max_seq_len": = 20,
    }

    config = PPOConfig().training(
        model=CUSTOM_MODEL,
        lr=LR,
        train_batch_size=train_batch_size,
        lambda_=0.80,
        vf_loss_coeff=0.5,
        entropy_coeff=ENTROPY_COEFF,
        clip_param=0.2,
        vf_clip_param=VF_CLIP_PARAM,
        sgd_minibatch_size=SGD_MINIBATCH_SIZE,
        num_sgd_iter=NUM_SGD_ITER,
        _enable_learner_api=False,
    ).rollouts(
        # batch_mode="complete_episodes",
        num_rollout_workers=num_workers,
        rollout_fragment_length=100,  # or horizon
        num_envs_per_worker=NUM_ENVS_PER_WORKER,
    ).multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    ).fault_tolerance(
        recreate_failed_workers=True,
        num_consecutive_worker_failures_tolerance=3,
    ).environment(
        env="meltingpot",
        env_config=env_config,
    ).debugging(
        log_level=args.logging,
        seed=args.seed,
    ).resources(
        num_gpus=args.num_gpus,
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
        num_cpus_for_local_worker=1,
        num_learner_workers=0,
        # num_cpus_per_learner_worker: Optional[Union[float, int]] = NotProvided,
        # num_gpus_per_learner_worker: Optional[Union[float, int]] = NotProvided,
    ).framework(framework=args.framework,
    ).reporting(metrics_num_episodes_for_smoothing=1,
    ).evaluation(
        evaluation_interval=None,  # don't evaluate unless we call evaluation()
        # evaluation_config={
        #     "explore": EXPLORE_EVAL,
        #     "env_config": env_eval_config,
        # },
        # evaluation_duration=EVAL_DURATION,
    ).experimental(
        _disable_preprocessor_api=True
    ).rl_module(
        _enable_rl_module_api=False
        )


    ckpt_config = CheckpointConfig(num_to_keep=KEEP_CHECKPOINTS_NUM,
                                   checkpoint_frequency=CHECKPOINT_FREQ,
                                   checkpoint_at_end=True)

    # Setup WanDB
    if "WANDB_API_KEY" in os.environ and args.wandb:
        wandb_project = f'{args.exp}_{args.framework}'
        wandb_group = "meltingpot"

        # Set up Weights And Biases logging if API key is set in environment variable.
        wdb_callbacks = [
            WandbLoggerCallback(
                project=wandb_project,
                group=wandb_group,
                api_key=os.environ["WANDB_API_KEY"],
                log_config=True,
            )
        ]
    else:
        wdb_callbacks = []
        print("WARNING! No wandb API key found, running without wandb!")

    run_config = RunConfig(name=args.exp,
                           callbacks=wdb_callbacks,
                           local_dir=f"{args.results_dir}/{args.framework}",
                           stop={"training_iteration": args.n_iterations},
                           checkpoint_config=ckpt_config,
                           verbose=VERBOSE)

    tune_config = TuneConfig(reuse_actors=False)

    return config, run_config, tune_config
