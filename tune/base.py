import flax.linen as nn


def evaluate_model(main):
  scores, training_walltime, metrics, eval_metrics = main(None)

  # Extract evaluation metric (mean score)
  eval_score = eval_metrics.get("eval/mean_score", -float("inf"))
  print(f"Eval score: {eval_score}")
  return eval_score


def selectParam(search_space, trial):
  def selector(key):
    params = search_space[key]
    param_type, param_range, kwargs = params[0], params[1], params[2] if len(params) > 2 else {}
    suggest_methods = {
      "categorical": trial.suggest_categorical,
      "float": trial.suggest_float,
      "int": trial.suggest_int,
    }

    if param_type == "categorical_fn":
      name = trial.suggest_categorical(key, param_range)
      activation = {
        "nn.swish": nn.swish,
        "nn.relu": nn.relu,
        "nn.tanh": nn.tanh,
      }[name]

      return activation
    elif param_type == "categorical_seq":
      size = trial.suggest_categorical(key, param_range)
      val, mul = {
        2 * 128: (128, 2),
        2 * 256: (256, 2),
        2 * 32: (32, 2),
        2 * 64: (64, 2),
        3 * 256: (256, 3),
        3 * 32: (32, 3),
        3 * 64: (64, 3),
        4 * 32: (32, 4),
        5 * 128: (128, 5),
        5 * 256: (256, 5),
      }[size]

      return (val,) * mul
    if param_type == "categorical":
      return suggest_methods[param_type](key, param_range)
    else:
      return suggest_methods[param_type](key, param_range[0], param_range[1], **kwargs)

  return selector
