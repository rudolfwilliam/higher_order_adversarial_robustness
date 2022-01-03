# Hessian vector product accuracies
# LOW corresponds to original CURE
LOW = 0
MED = 1
HIGH = 2


CIFAR_CONFIG = {
    # Constants
    "device": "cpu",

    # Data parameters
    "batch_size_train": 100,  # E: Bugs want this to be 100 for cifar
    "batch_size_test": 1000,
    "shuffle_train": True,
    "image_min": 0,
    "image_max": 1,


    # Getter functions
    "dataset": 'MNIST', # MNIST or CIFAR10
    "model_name": 'SimpleModel',  # SimpleModel or ResNet18

    # CURE configurations
    "accuracy": MED,
    "lambda_0": 0,
    "lambda_1": 4,
    "lambda_2": 0,
    "h": [0.1, 0.5, 0.9, 1.3, 1.5],
    "optimization_algorithm": 'Adam',
    "optimizer_arguments": {
        'lr': 1e-4
    },
    "epochs": 20,
    "epsilon": 8 / 255,

    "use_checkpoint": False,
    "checkpoint_file": 'checkpoint_01.data'
}

#CIFAR_CONFIG_RESNET20 = CIFAR_CONFIG
#CIFAR_CONFIG_RESNET20["model_name"] = 'ResNet20'
