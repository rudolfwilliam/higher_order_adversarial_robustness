from utils.config import CIFAR_CONFIG
from main import train_CURE
from parameter_tuning import objective_CURE, objective_find_best_lambdas, tune_hyperparameters
import json
from pathlib import Path


def store_config(config, file_name):
    """
    Stores a config dictionary in a .json file

    Args:
        config (dict): The dictionary object to save
        file_name (str): The file name used to store the config into
    """
    checkpoint_path = Path("./data/checkpoints/")
    store_path = checkpoint_path / (file_name + ".json")

    # Convert the dictionary into a json object
    json_data = json.dumps(config)

    # Store the json object in a file
    with open(store_path, 'w') as f:
        f.write(json_data)


def get_base_config(epsilon=2 / 255, accuracy=1):
    """
    Returns a basic configuration dictionary with a few important but
    repeatedly used arguments already set

    Args:
        epsilon (float): A float value used to configure the strength of PGD
        accuracy (int): One of [1,2,4,6,8] denoting the accuracy level of the finite
            difference computation

    Returns:
        A python dict object storing a base config
    """
    # Make a copy of the default config
    config = dict(CIFAR_CONFIG)

    # Set a few parameters required for all experiments
    config["dataset"] = 'CIFAR10'
    config["model_name"] = 'ResNet18'
    config['batch_size_test'] = 200
    config["accuracy"] = accuracy
    config["epsilon"] = epsilon
    config["epochs"] = 20

    return config


def reproduce_CURE(accuracy):
    """
    Runs our model with the same hyperparameters as the CURE paper used.

    Args:
        accuracy (int): One of [1,2,4,6,8] denoting the accuracy level of the finite
            difference computation
    """
    # Set a few base parameters
    config = get_base_config(accuracy=accuracy)

    config["lambda_0"] = 0
    config["lambda_1"] = 4
    config["lambda_2"] = 0

    # Iterate over different pgd strength levels
    for epsilon in [1, 2, 4, 8]:
        config["epsilon"] = epsilon / 255

        file_name = f"experiment_reproduce_CURE_eps{epsilon}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        # Run the experiment
        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def CURE_higher_accuracy():
    """
    Runs our model with the same hyperparameters as the CURE paper used. However,
    the accuracy of the finite-difference computation is increased.
    """
    # Set a few base parameters
    config = get_base_config()

    config["lambda_0"] = 0
    config["lambda_1"] = 4
    config["lambda_2"] = 0

    # Iterate over the higher order accuracy levels
    for accuracy in [2, 4, 6]:
        # Iterate over different pgd strength levels
        for epsilon in [1, 2, 4, 8]:
            config["epsilon"] = epsilon / 255
            config["accuracy"] = accuracy

            file_name = f"experiment_reproduce_CURE_eps{epsilon}_acc{accuracy}.data"
            config['checkpoint_file'] = file_name

            # Run the experiment
            store_config(config, file_name)
            train_CURE(config, plot_results=False)


def find_best_hyperparams_CURE():
    """
    Performs a hyperparameter search to find the best hyperparameters for standard CURE
    """
    # Start the hyperparameter optimization
    tune_hyperparameters(
        n_trials=200,
        save_frequency=10,
        save_path="./data/checkpoints/",
        existing_study_name=None,
        objective=objective_CURE)


def CURE_and_1st_order_reg(epsilon, accuracy):
    """
    Combines CURE with our first order regularizer. The experiment tries out different
    combinations of regularization coefficients lambda_0

    Args:
        epsilon (float): A float value used to configure the strength of PGD
        accuracy (int): One of [1,2,4,6,8] denoting the accuracy level of the finite
            difference computation
    """
    # Set a few base parameters
    config = get_base_config(epsilon, accuracy)

    config["lambda_1"] = 4
    config["lambda_2"] = 0

    # Iterate over different choices of lambda_0
    for lambda_0 in [0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 500]:
        config["lambda_0"] = lambda_0

        file_name = f"experiment_15d_eps{epsilon}_lambda0_{lambda_0}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        # Run experiment
        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def CURE_and_3rd_order_reg(epsilon, accuracy):
    """
    Combines CURE with our third order regularizer. The experiment tries out different
    combinations of regularization coefficients lambda_2

    Args:
        epsilon (float): A float value used to configure the strength of PGD
        accuracy (int): One of [1,2,4,6,8] denoting the accuracy level of the finite
            difference computation
    """
    # Set a few base parameters
    config = get_base_config(epsilon, accuracy)

    config["lambda_0"] = 0
    config["lambda_1"] = 4

    # Iterate over different choices of lambda_2
    for lambda_2 in [0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 500]:
        config["lambda_2"] = lambda_2

        file_name = f"experiment_15d_eps{epsilon}_lambda2_{lambda_2}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        # Run experiment
        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def CURE_and_3rd_order_reg_and_h_scaler(epsilon, accuracy):
    """
    Scales the 'h' values for different scaling values, while keeping the remaining
    parameters constant.

    Args:
        epsilon (float): A float value used to configure the strength of PGD
        accuracy (int): One of [1,2,4,6,8] denoting the accuracy level of the finite
            difference computation
    """
    # Set a few base parameters
    config = get_base_config(epsilon, accuracy)

    config["lambda_0"] = 0
    config["lambda_1"] = 4
    config["lambda_2"] = 5

    # Iterate over different scaling constants h_scaler
    for h_scaler in [0.001, 0.01, 0.1, 1, 2, 4, 10]:
        config["h"] = [h * h_scaler for h in config['h']]

        file_name = f"experiment_15d_eps{epsilon}_lambda2_5_hscaled_{h_scaler}_acc{accuracy}.data"
        config['checkpoint_file'] = file_name

        # Run experiment
        store_config(config, file_name)
        train_CURE(config, plot_results=False)


def find_best_lambda_combination():
    """
    Performs a hyperparameter search to find the best combination of regularization
    coefficients lambda_0, lambda_1, lambda_2
    """
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

    # Main experiments
    reproduce_CURE(accuracy=1)
    CURE_and_3rd_order_reg(epsilon=2/255, accuracy=1)
    CURE_and_3rd_order_reg_and_h_scaler(epsilon=2/255, accuracy=1)
    CURE_and_1st_order_reg(epsilon=2/255, accuracy=1)
    CURE_higher_accuracy()

    find_best_lambda_combination()
    find_best_hyperparams_CURE()

    # Experiments from appendix
    CURE_and_3rd_order_reg(epsilon=8/255, accuracy=1)
    CURE_and_1st_order_reg(epsilon=8/255, accuracy=1)
    CURE_and_3rd_order_reg_and_h_scaler(epsilon=8/255, accuracy=1)
