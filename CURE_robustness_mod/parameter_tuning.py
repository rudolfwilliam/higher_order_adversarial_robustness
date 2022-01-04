import optuna
import joblib
from datetime import datetime
from main import train_CURE
from utils.config import CIFAR_CONFIG
import sys
from pathlib import Path
import logging


def objective_CURE(trial):
    """
    Trains a CURE model with a specified regularization coefficient lambda_1 and then returns
    the resulting adversarial accuracy

    Args:
        trial (optuna.Trial) An optuna Trial object

    Returns:
        A float value denoting the adversarial accuracy achieved by this run.
    """
    # Load the config
    config = dict(CIFAR_CONFIG)
    config["dataset"] = 'CIFAR10'
    config["model_name"] = 'ResNet18'
    config['batch_size_test'] = 200
    config["accuracy"] = 1
    config["epsilon"] = 8 / 255
    config["epochs"] = 20
    config["lambda_0"] = 0
    config["lambda_2"] = 0

    # Let optuna select a the tunable parameters
    config["lambda_1"] = trial.suggest_uniform('lambda_1', 0, 15)
    config["optimizer_arguments"]["lr"] = trial.suggest_loguniform('lr', 1e-8, 1e-2)

    for i in range(5):
        config["h"][i] = trial.suggest_uniform(f'h_{i}', 0.01, 5)

    # Train CURE with the selected hyperparameters
    net_CURE = train_CURE(config, plot_results=False, trial=trial)

    return net_CURE.test_acc_adv[-1]


def objective_find_best_lambdas(trial):
    """
    Trains a CURE model with specified regularization coefficients lambda_0, lambda_1 and lambda_2
    and then returns a linear combination of the resulting adversarial- and natural accuracy

    Args:
        trial (optuna.Trial) An optuna Trial object

    Returns:
        A float value denoting a linear combination of the resulting adversarial- and natural accuracy
    """
    # Load the config
    config = dict(CIFAR_CONFIG)
    config["dataset"] = 'CIFAR10'
    config["model_name"] = 'ResNet18'
    config['batch_size_test'] = 200
    config["accuracy"] = 1
    config["epsilon"] = 8 / 255
    config["epochs"] = 20

    # Let optuna select a the tunable parameters
    scaler = trial.suggest_uniform('scaler', 0, 100)
    lambda_0 = trial.suggest_uniform('lambda_0', 0, 1)
    lambda_1 = trial.suggest_uniform('lambda_1', 0, 1)
    lambda_2 = trial.suggest_uniform('lambda_2', 0, 1)

    config['lambda_0'] = scaler * lambda_0
    config['lambda_1'] = scaler * lambda_1
    config['lambda_2'] = scaler * lambda_2

    # Train CURE with the selected hyperparameters
    net_CURE = train_CURE(config, plot_results=False, trial=trial)

    value = net_CURE.test_acc_adv[-1] + net_CURE.test_acc_clean[-1] / 5

    return value


def tune_hyperparameters(n_trials=1000, save_frequency=10, save_path=".", existing_study_name=None, objective=objective_CURE):
    """
    Performs a hyperparameter optimization with a given objective function.

    Args:
        n_trials (int): The number of different parameter configurations which should be tried out
        save_frequency (int): After how many trials the study object shall be cached to disk
        save_path (str): A path to the place where the study object shall be cached
        existing_study_name (str or None): If not None, resuem optimization from an existing study
            object stored at this path
        objective (func): An objective function accepting an optuna.Trial object and returning a score value
    """

    # Enable logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Create the study if it doesn't exist yet
    if existing_study_name is None:
        study_path = Path(save_path) / f"optuna_study_{datetime.now()}.pkl"

        # Create the study
        study = optuna.create_study(
            study_name=f"optuna_study_{datetime.now()}.pkl",
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(),
            direction="maximize"
        )

        # Store the study
        joblib.dump(study, study_path)

    else:
        study_path = Path(save_path) / existing_study_name

    # Start optimizing the hyperparameters. Save the study every 'save_frequency' steps
    total_trials = 0
    while total_trials < n_trials:
        # Load the previous checkpoint
        current_study = joblib.load(study_path)

        # Optimize the study for 'save_frequency' steps
        current_study.optimize(objective, n_trials=save_frequency, gc_after_trial=True)

        # Store a checkpoint of the study
        joblib.dump(current_study, study_path)

        total_trials += save_frequency


if __name__ == "__main__":
    tune_hyperparameters()
