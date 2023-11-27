import argparse
import os

import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from tune.base import evaluate_model, selectParam


def config(model_name):
  if model_name == "trpo":
    from tune.tune_trpo import objective
  elif model_name == "ppo":
    from tune.tune_ppo import objective
  elif model_name == "a2c":
    from tune.tune_a2c import objective
  elif model_name == "reinforce":
    from tune.tune_reinforce import objective
  elif model_name == "vmpo":
    from tune.tune_vmpo import objective
  else:
    raise ValueError(f"Unknown model name: {model_name}")

  return objective


if __name__ == "__main__":
  print("Running Optuna hyperparameter optimization.")
  parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization.")
  parser.add_argument("--model", type=str, choices=["trpo", "ppo", "a2c", "reinforce"], default="ppo", help="Model to optimize.")
  parser.add_argument("--trials", type=int, default=1000, help="Number of trials for the optimization.")
  parser.add_argument("--env", type=str, default="Humanoid-v5", help="Environment to optimize.")
  parser.add_argument("--store-optuna", type=bool, default=False, help="Store the optuna study.")
  params = parser.parse_args()

  print(f"Optimizing {params.model} on {params.env} with {params.trials} trials, using {os.environ['JAX_PLATFORMS']}.")

  objective = config(params.model)

  sampler = TPESampler()
  pruner = optuna.pruners.MedianPruner()
  study_name = f"{params.model}_{params.env}"
  storage = None

  if params.store_optuna:
    optuna_dir = f".optuna/{params.model}/{params.env}"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(f"{optuna_dir}/storage"))

  print(f"Preparing to optimize {params.model} with {params.trials} trials")
  study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage, study_name=study_name, load_if_exists=True, pruner=pruner)

  print(f"Optimizing {params.model} with {params.trials} trials")
  study.optimize(objective, n_trials=params.trials)

  # Log the best trial
  print("Best trial:")
  print(f"  Value: {study.best_trial.value}")
  print("  Params: ")
  for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
