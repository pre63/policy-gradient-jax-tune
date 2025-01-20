import argparse

import optuna
from optuna.samplers import TPESampler
from policy_gradients_jax.ppo import Config, main

from tune.base import evaluate_model, selectParam

# Static search space
SEARCH_SPACE = {
  # Environment parameters
  "clip_actions": ("categorical", [True, False]),
  "normalize_observations": ("categorical", [True, False]),
  "normalize_rewards": ("categorical", [True, False]),
  "clip_observations": ("float", (1.0, 20.0), {"step": 1.0}),
  "clip_rewards": ("float", (1.0, 20.0), {"step": 1.0}),
  "num_eval_episodes": ("categorical", [10]),
  "eval_every": ("categorical", [100]),
  # Algorithm hyperparameters
  "total_timesteps": ("float", (1e5, 1e8), {"log": True}),
  "learning_rate": ("float", (1e-5, 1e-2), {"log": True}),
  "unroll_length": ("categorical", [512, 1024, 2048]),
  "anneal_lr": ("categorical", [True, False]),
  "gamma": ("float", (0.95, 0.999), {"step": 0.001}),
  "gae_lambda": ("float", (0.9, 1.0), {"step": 0.01}),
  "batch_size": ("categorical", [1, 4, 8, 16]),
  "num_minibatches": ("categorical", [1, 2, 4, 8]),
  "update_epochs": ("categorical", [1, 2, 5]),
  "normalize_advantages": ("categorical", [True, False]),
  "entropy_cost": ("float", (0.0, 0.1), {"step": 0.01}),
  "vf_cost": ("float", (0.1, 1.0), {"step": 0.1}),
  "max_grad_norm": ("float", (0.1, 1.0), {"step": 0.1}),
  "reward_scaling": ("float", (0.1, 10.0), {"log": True}),
  # Policy parameters
  "policy_hidden_layer_sizes": ("categorical", [2 * 32, 4 * 32, 3 * 64]),
  "value_hidden_layer_sizes": ("categorical", [3 * 256, 5 * 256, 5 * 128]),
  "activation": ("categorical", ["nn.swish", "nn.relu", "nn.tanh"]),
  "squash_distribution": ("categorical", [True, False]),
  # Atari parameters
  "atari_dense_layer_sizes": None,
}


def objective(envs, capture_video=False, write_logs_to_file=True, seed=10, num_envs=1, parallel_envs=False):
  SEARCH_SPACE["env_id"] = ("categorical", envs)

  def objective(trial):
    selector = selectParam(SEARCH_SPACE, trial)

    Config.seed = seed
    Config.capture_video = capture_video
    Config.write_logs_to_file = write_logs_to_file
    Config.num_envs = num_envs

    # Ensure Config.num_envs is not zero
    if Config.num_envs <= 0:
      Config.num_envs = 1  # Default to 1 if invalid

    # Ensure the assertion condition is met
    total_steps = Config.batch_size * Config.num_minibatches
    if total_steps % Config.num_envs != 0:
      # Adjust Config.num_envs to a divisor of total_steps
      for divisor in range(total_steps, 0, -1):
        if total_steps % divisor == 0:
          Config.num_envs = divisor
          break

    Config.parallel_envs = parallel_envs
    Config.clip_actions = selector("clip_actions")
    Config.normalize_observations = selector("normalize_observations")
    Config.normalize_rewards = selector("normalize_rewards")
    Config.clip_observations = selector("clip_observations")
    Config.clip_rewards = selector("clip_rewards")
    Config.num_eval_episodes = selector("num_eval_episodes")
    Config.eval_every = selector("eval_every")
    Config.total_timesteps = selector("total_timesteps")
    Config.learning_rate = selector("learning_rate")
    Config.unroll_length = selector("unroll_length")
    Config.anneal_lr = selector("anneal_lr")
    Config.gamma = selector("gamma")
    Config.gae_lambda = selector("gae_lambda")
    Config.batch_size = selector("batch_size")
    Config.num_minibatches = selector("num_minibatches")
    Config.update_epochs = selector("update_epochs")
    Config.normalize_advantages = selector("normalize_advantages")
    Config.entropy_cost = selector("entropy_cost")
    Config.vf_cost = selector("vf_cost")
    Config.max_grad_norm = selector("max_grad_norm")
    Config.reward_scaling = selector("reward_scaling")
    Config.policy_hidden_layer_sizes = selector("policy_hidden_layer_sizes")
    Config.value_hidden_layer_sizes = selector("value_hidden_layer_sizes")
    Config.activation = selector("activation")
    Config.squash_distribution = selector("squash_distribution")
    if SEARCH_SPACE["atari_dense_layer_sizes"]:
      Config.atari_dense_layer_sizes = selector("atari_dense_layer_sizes")

    # Replace with your evaluation function
    return evaluate_model()

  return objective
