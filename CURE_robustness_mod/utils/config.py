
CIFAR_CONFIG = {
    # Constants
    "device": "cuda",

    "batch_size_train": 100,  # E: For some reasons the bugs want this to be 100 for cifar
    "batch_size_test": 1000,

    "shuffle_train": True,

    # Getter functions
    "dataset": 'CIFAR10',
    "model_name": 'ResNet18',  # SimpleModel or ResNet18

    # CURE configurations
    "lambda_": 1,
    "h": [0.1, 0.4, 0.8, 1.8, 3],
    "optimization_algorithm": 'SGD',
    "optimizer_arguments": {
        'lr': 1e-4
    },
    "epochs": 10,

    "use_checkpoint": False,
    "checkpoint_file": 'checkpoint_01.data'
}
