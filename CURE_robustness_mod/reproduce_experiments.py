from utils.config import CIFAR_CONFIG
from main import train_CURE
from parameter_tuning import objective_CURE, objective_find_best_lambdas, tune_hyperparameters
import json
from pathlib import Path


def store_config(config, file_name):
    checkpoint_path = Path("./data/checkpoints/")
    store_path = checkpoint_path / (file_name + ".json")

    json_data = json.dumps(config)

    with open(store_path, 'w') as f:
        f.write(json_data)


def get_base_config(epsilon=2 / 255, accuracy=1):
    config = dict(CIFAR_CONFIG)
    config["dataset"] = 'CIFAR10'
    config["model_name"] = 'ResNet18'
    config['batch_size_test'] = 200
    config["accuracy"] = accuracy
    config["epsilon"] = epsilon
    config["epochs"] = 20

    return config


def reproduce_CURE(accuracy):
    config = get_base_config(accuracy=accuracy)

    config["lambda_0"] = 0
    config["lambda_1"] = 4
    config["lambda_2"] = 0

    for epsilon in [1, 2, 4, 8]:
        config["epsilon"] = epsilon / 255

        file_name = f"experiment_reproduce_CURE_eps{epsilon}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def CURE_higher_accuracy():
    config = get_base_config()

    config["lambda_0"] = 0
    config["lambda_1"] = 4
    config["lambda_2"] = 0

    for accuracy in [2, 4, 6]:
        for epsilon in [1, 2, 4, 8]:
            config["epsilon"] = epsilon / 255
            config["accuracy"] = accuracy

            file_name = f"experiment_reproduce_CURE_eps{epsilon}_acc{accuracy}.data"
            config['checkpoint_file'] = file_name

            store_config(config, file_name)
            train_CURE(config, plot_results=False)


def find_best_hyperparams_CURE():
    tune_hyperparameters(
        n_trials=200,
        save_frequency=10,
        save_path="./data/checkpoints/",
        existing_study_name=None,
        objective=objective_CURE)


def CURE_and_1st_order_reg(epsilon, accuracy):
    config = get_base_config(epsilon, accuracy)

    config["lambda_1"] = 4
    config["lambda_2"] = 0

    for lambda_0 in [0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 500]:
        config["lambda_0"] = lambda_0

        file_name = f"experiment_15d_eps{epsilon}_lambda0_{lambda_0}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def CURE_and_3rd_order_reg(epsilon, accuracy):
    config = get_base_config(epsilon, accuracy)

    config["lambda_0"] = 0
    config["lambda_1"] = 4

    for lambda_2 in [0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 500]:
        config["lambda_2"] = lambda_2

        file_name = f"experiment_15d_eps{epsilon}_lambda2_{lambda_2}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def CURE_and_3rd_order_reg_and_h_scaler(epsilon, accuracy):
    config = get_base_config(epsilon, accuracy)

    config["lambda_0"] = 0
    config["lambda_1"] = 4
    config["lambda_2"] = 5

    for h_scaler in [0.001, 0.01, 0.1, 1, 2, 4, 10]:
        config["h"] = [h * h_scaler for h in config['h']]

        file_name = f"experiment_15d_eps{epsilon}_lambda2_5_hscaled_{h_scaler}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def find_best_lambda_combination():
    tune_hyperparameters(
        n_trials=400,
        save_frequency=10,
        save_path="./data/checkpoints/",
        existing_study_name=None,
        objective=objective_find_best_lambdas)


if __name__ == "__main__":
    # REMEMBER that epsilons have to be divided by 255
    # So instead of calling a function with "epsilon = 2"
    # use "epsilon = 2 / 255"

    pass
