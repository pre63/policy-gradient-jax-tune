import argparse

import optuna
from optuna.samplers import TPESampler
from policy_gradients_jax.trpo import Config, main

from tune.base import evaluate_model, selectParam

# Static search space
SEARCH_SPACE = {
  # Experiment parameters
  "seed": ("categorical", [10]),
  "capture_video": ("categorical", [False]),
  "write_logs_to_file": ("categorical", [True]),
  # Environment parameters
  "env_id": ("categorical", ["Humanoid-v5"]),
  "num_envs": ("categorical", [1]),
  "parallel_envs": ("categorical", [True]),
  "clip_actions": ("categorical", [False]),
  "normalize_observations": ("categorical", [True]),
  "normalize_rewards": ("categorical", [False]),
  "clip_observations": ("float", (1.0, 20.0), {"step": 1.0}),
  "clip_rewards": ("float", (1.0, 20.0), {"step": 1.0}),
  "num_eval_episodes": ("categorical", [10]),
  "eval_every": ("categorical", [100]),
  # Algorithm hyperparameters
  "total_timesteps": ("float", (1e6, 1e8), {"log": True}),
  "learning_rate": ("float", (1e-5, 1e-2), {"log": True}),
  "unroll_length": ("categorical", [1024, 2048, 4096]),
  "anneal_lr": ("categorical", [True, False]),
  "gamma": ("float", (0.95, 0.999), {"step": 0.001}),
  "gae_lambda": ("float", (0.9, 1.0), {"step": 0.01}),
  "batch_size": ("categorical", [1, 2, 4, 8]),
  "num_minibatches": ("categorical", [4, 8, 16]),
  "update_epochs": ("categorical", [5, 10, 20]),
  "normalize_advantages": ("categorical", [True, False]),
  "target_kl": ("float", (0.001, 0.05), {"step": 0.001}),
  "cg_damping": ("float", (0.01, 1.0), {"log": True}),
  "cg_max_iterations": ("categorical", [5, 10, 20]),
  "line_search_max_iter": ("categorical", [5, 10, 20]),
  "line_search_shrinking_factor": ("float", (0.5, 0.9), {"step": 0.05}),
  "vf_cost": ("float", (0.1, 10.0), {"log": True}),
  "max_grad_norm": ("float", (0.1, 1.0), {"step": 0.1}),
  "reward_scaling": ("float", (0.1, 10.0), {"log": True}),
  # Policy parameters
  "policy_hidden_layer_sizes": ("categorical_seq", [2 * 32, 2 * 64]),
  "value_hidden_layer_sizes": ("categorical_seq", [2 * 256, 2 * 128, 2 * 256]),
  "activation": ("categorical_fn", ["nn.swish", "nn.relu", "nn.tanh"]),
  "squash_distribution": ("categorical", [True, False]),
  # Atari parameters
  "atari_dense_layer_sizes": None,  # ("categorical", [(512,), (1024,), (512, 512)], [])
}


def objective(trial):
  Config.seed = selectParam(SEARCH_SPACE, trial, "seed")
  Config.env_id = selectParam(SEARCH_SPACE, trial, "env_id")
  Config.batch_size = selectParam(SEARCH_SPACE, trial, "batch_size")
  Config.num_minibatches = selectParam(SEARCH_SPACE, trial, "num_minibatches")
  Config.num_envs = selectParam(SEARCH_SPACE, trial, "num_envs")

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

  Config.capture_video = selectParam(SEARCH_SPACE, trial, "capture_video")
  Config.write_logs_to_file = selectParam(SEARCH_SPACE, trial, "write_logs_to_file")
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
  Config.update_epochs = selectParam(SEARCH_SPACE, trial, "update_epochs")
  Config.normalize_advantages = selectParam(SEARCH_SPACE, trial, "normalize_advantages")
  Config.target_kl = selectParam(SEARCH_SPACE, trial, "target_kl")
  Config.cg_damping = selectParam(SEARCH_SPACE, trial, "cg_damping")
  Config.cg_max_iterations = selectParam(SEARCH_SPACE, trial, "cg_max_iterations")
  Config.line_search_max_iter = selectParam(SEARCH_SPACE, trial, "line_search_max_iter")
  Config.line_search_shrinking_factor = selectParam(SEARCH_SPACE, trial, "line_search_shrinking_factor")
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
  return evaluate_model(main)
