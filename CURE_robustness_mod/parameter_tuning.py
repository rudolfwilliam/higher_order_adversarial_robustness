import optuna
import joblib
from datetime import datetime
from main import train_CURE
from utils.config import CIFAR_CONFIG
import sys
from pathlib import Path
import logging


def objective(trial):
    # Load the config
    config = CIFAR_CONFIG

    # Let optuna select a the tunable parameters
    config["lambda_"] = trial.suggest_uniform('lambda_', 0, 15)
    #config["optimization_algorithm"] = trial.suggest_categorical("optimization_algorithm", ['SGD', 'Adam'])
    config["optimizer_arguments"]["lr"] = trial.suggest_loguniform('lr', 1e-8, 1e-2)

    for i in range(5):
        config["h"][i] = trial.suggest_uniform(f'h_{i}', 0.01, 5)

    # Train CURE with the selected hyperparameters
    adversarial_accuracy = train_CURE(config, plot_results=False, trial=trial)[-1]

    return adversarial_accuracy


def objective_find_best_lambdas(trial):
    # Load the config
    config = CIFAR_CONFIG

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


def tune_hyperparameters(n_trials=1000, save_frequency=10, save_path=".", existing_study_name=None):
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
        current_study.optimize(objective_find_best_lambdas, n_trials=save_frequency, gc_after_trial=True)

        # Store a checkpoint of the study
        joblib.dump(current_study, study_path)

        total_trials += save_frequency


if __name__ == "__main__":
    tune_hyperparameters()
