CIFAR_CONFIG = {
    # Constants
    "device": "cuda",

    # Data parameters
    "batch_size_train": 30,  # E: Bugs want this to be 100 for cifar
    "batch_size_test": 100,
    "shuffle_train": True,
    "image_min": 0,
    "image_max": 1,

    # Getter functions
    "dataset": 'MNIST',
    "model_name": 'SimpleModel',  # SimpleModel or AlexNet or VGG or ResNet18

    # CURE configurations
    "accuracy": 1, # Can be 1, 2, 4, 6, 8
    "lambda_0": 1, # Gradient regularizer
    "lambda_1": 1, # Original CURE
    "lambda_2": 1, # Third order regularizer
    "h": [0.1, 0.5, 0.9, 1.3, 1.5], # Length determines minimum nr of epochs
    "optimization_algorithm": 'Adam',
    "optimizer_arguments": {
        'lr': 1e-4
    },
    "epochs": 10,
    "epsilon": 4 / 255,

    "use_checkpoint": False,
    "checkpoint_file": 'checkpoint_01.data'
}

CIFAR_CONFIG_RESNET20 = dict(CIFAR_CONFIG)          # This now performs a deep copy
CIFAR_CONFIG_RESNET20["model_name"] = 'ResNet20'
