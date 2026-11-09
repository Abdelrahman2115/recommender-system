"""Hyperparameters chosen on the validation set are stored here so later notebooks reuse them."""
import json
import os

PARAMS_PATH = "reports/best_params.json"


def load_params(path=PARAMS_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def update_params(new, path=PARAMS_PATH):
    params = load_params(path)
    params.update(new)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return params
