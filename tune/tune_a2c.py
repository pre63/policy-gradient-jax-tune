import argparse

import optuna
from optuna.samplers import TPESampler
from policy_gradients_jax.ppo import Config, main

from tune.base import evaluate_model, selectParam

# Static search space
SEARCH_SPACE = {
  # Experiment parameters
  "seed": ("categorical", [10]),
  "capture_video": ("categorical", [True, False]),
  "write_logs_to_file": ("categorical", [True, False]),
  # Environment parameters
  "env_id": ("categorical", ["Humanoid-v5"]),
  "num_envs": ("categorical", [1]),
  "parallel_envs": ("categorical", [True, False]),
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


def objective(trial):
  Config.seed = selectParam(SEARCH_SPACE, trial, "seed")

  Config.capture_video = selectParam(SEARCH_SPACE, trial, "capture_video")
  Config.write_logs_to_file = selectParam(SEARCH_SPACE, trial, "write_logs_to_file")
  Config.num_envs = selectParam(SEARCH_SPACE, trial, "num_envs")
  Config.parallel_envs = selectParam(SEARCH_SPACE, trial, "parallel_envs")
  Config.clip_actions = selectParam(SEARCH_SPACE, trial, "clip_actions")
  Config.normalize_observations = selectParam(SEARCH_SPACE, trial, "normalize_observations")
  Config.normalize_rewards = selectParam(SEARCH_SPACE, trial, "normalize_rewards")
  Config.clip_observations = selectParam(SEARCH_SPACE, trial, "clip_observations")
  Config.clip_rewards = selectParam(SEARCH_SPACE, trial, "clip_rewards")
  Config.num_eval_episodes = selectParam(SEARCH_SPACE, trial, "num_eval_episodes")
  Config.eval_every = selectParam(SEARCH_SPACE, trial, "eval_every")
  Config.total_timesteps = selectParam(SEARCH_SPACE, trial, "total_timesteps")
  Config.learning_rate = selectParam(SEARCH_SPACE, trial, "learning_rate")
  Config.unroll_length = selectParam(SEARCH_SPACE, trial, "unroll_length")
  Config.anneal_lr = selectParam(SEARCH_SPACE, trial, "anneal_lr")
  Config.gamma = selectParam(SEARCH_SPACE, trial, "gamma")
  Config.gae_lambda = selectParam(SEARCH_SPACE, trial, "gae_lambda")
  Config.batch_size = selectParam(SEARCH_SPACE, trial, "batch_size")
  Config.num_minibatches = selectParam(SEARCH_SPACE, trial, "num_minibatches")
  Config.update_epochs = selectParam(SEARCH_SPACE, trial, "update_epochs")
  Config.normalize_advantages = selectParam(SEARCH_SPACE, trial, "normalize_advantages")
  Config.entropy_cost = selectParam(SEARCH_SPACE, trial, "entropy_cost")
  Config.vf_cost = selectParam(SEARCH_SPACE, trial, "vf_cost")
  Config.max_grad_norm = selectParam(SEARCH_SPACE, trial, "max_grad_norm")
  Config.reward_scaling = selectParam(SEARCH_SPACE, trial, "reward_scaling")
  # Config.policy_hidden_layer_sizes = selectParam(SEARCH_SPACE, trial, "policy_hidden_layer_sizes")
  # Config.value_hidden_layer_sizes = selectParam(SEARCH_SPACE, trial, "value_hidden_layer_sizes")
  Config.activation = selectParam(SEARCH_SPACE, trial, "activation")
  Config.squash_distribution = selectParam(SEARCH_SPACE, trial, "squash_distribution")
  if SEARCH_SPACE["atari_dense_layer_sizes"]:
    Config.atari_dense_layer_sizes = selectParam(SEARCH_SPACE, trial, "atari_dense_layer_sizes")

  # Replace with your evaluation function
  return evaluate_model()
