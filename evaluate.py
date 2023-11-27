import argparse
import importlib

import yaml


def load_hyperparameters(model_name, env):
  # Function to load hyperparameters from YAML
  try:
    with open(f"hyperparams/{model_name}.yml", "r") as file:
      hyperparams = yaml.safe_load(file)
    if env not in hyperparams:
      raise ValueError(f"Environment '{env}' not found in {model_name}.yml")
    return hyperparams[env]
  except FileNotFoundError:
    raise FileNotFoundError(f"File 'hyperparams/{model_name}.yml' not found.")
  except Exception as e:
    raise RuntimeError(f"Error loading hyperparameters: {e}")


def configure_model(config_class, hyperparams):
  # Function to configure the Config class
  for key, value in hyperparams.items():
    if hasattr(config_class, key):
      setattr(config_class, key, value)


def launch_model(model_name):
  # Function to dynamically import and run the model
  try:
    # Dynamically import the Config and main function for the model
    model_module = importlib.import_module(model_name)
    config_class = getattr(model_module, "Config")
    model_main = getattr(model_module, "main")

    return config_class, model_main
  except ImportError:
    raise ImportError(f"Model '{model_name}' is not available.")
  except AttributeError:
    raise AttributeError(f"Model '{model_name}' does not have a 'Config' or 'main' function.")


# Main function
if __name__ == "__main__":
  """
    python evaluate.py --model ppo --env Humanoid-v5
    """
  parser = argparse.ArgumentParser(description="Evaluate models with specific hyperparameters.")
  parser.add_argument("--model", type=str, required=True, help="Model to evaluate (e.g., ppo)")
  parser.add_argument("--env", type=str, required=True, help="Environment ID (e.g., Humanoid-v5)")

  args = parser.parse_args()
  model_name = args.model
  env = args.env

  try:
    # Load the correct model
    Config, main = launch_model(model_name)

    # Load hyperparameters
    hyperparams = load_hyperparameters(model_name, env)

    # Configure the Config class
    configure_model(Config, hyperparams)

    # Launch the model
    main()

  except Exception as e:
    print(f"Error: {e}")
