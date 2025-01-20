import argparse

from policy_gradients_jax.trpo import Config, main

from tune.base import evaluate_model, selectParam

# Static search space
SEARCH_SPACE = {
  # Environment parameters
  "env_id": ("categorical", ["Humanoid-v5"]),
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


def objective(envs, capture_video=False, write_logs_to_file=True, seed=10, num_envs=1, parallel_envs=False):
  SEARCH_SPACE["env_id"] = ("categorical", envs)

  def objective(trial):
    selector = selectParam(SEARCH_SPACE, trial)

    Config.seed = seed
    Config.env_id = selector("env_id")
    Config.batch_size = selector("batch_size")
    Config.num_minibatches = selector("num_minibatches")
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

    Config.capture_video = capture_video
    Config.write_logs_to_file = write_logs_to_file
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
    Config.update_epochs = selector("update_epochs")
    Config.normalize_advantages = selector("normalize_advantages")
    Config.target_kl = selector("target_kl")
    Config.cg_damping = selector("cg_damping")
    Config.cg_max_iterations = selector("cg_max_iterations")
    Config.line_search_max_iter = selector("line_search_max_iter")
    Config.line_search_shrinking_factor = selector("line_search_shrinking_factor")
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
    return evaluate_model(main)

  return objective
