from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_GRAM_VARIANT = "main_adaptive_mixture"

GRAM = {
    "graph_neighbors": 50,
    "diagnostic_neighbors": 50,
    "margin_graph_rank": 64,
    "gradient_graph_rank": 64,
    "active_kernel_components": ("margin", "gradient", "identity"),
    "initial_kernel_weights": (1.0 / 3.0,) * 3,
    "observation_noise_variance": 0.1,
    "weight_regularization": 0.1,
    "minimum_queried_for_weight_learning": 30,
    "require_both_status_classes_for_weight_learning": True,
    "iv_beta": 1.0,
    "verification_batch_size": 32,
}


DATASETS = {
    "cifar10": {
        "modality": "image",
        "source": "uoft-cs/cifar10",
        "data_files": {
            "train": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet",
            "test": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/refs%2Fconvert%2Fparquet/plain_text/test/0000.parquet",
        },
        "image_column": "img",
        "label_column": "label",
        "num_classes": 10,
    },
    "cifar100": {
        "modality": "image",
        "source": "uoft-cs/cifar100",
        "data_files": {
            "train": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/refs%2Fconvert%2Fparquet/cifar100/train/0000.parquet",
            "test": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/refs%2Fconvert%2Fparquet/cifar100/test/0000.parquet",
        },
        "image_column": "img",
        "label_column": "fine_label",
        "num_classes": 100,
    },
    "atis": {"modality": "text", "source": "pfsv/atis", "num_classes": 17},
    "qnli": {
        "modality": "text",
        "source": "nyu-mll/glue",
        "subset": "qnli",
        "num_classes": 2,
    },
    "adult": {"modality": "tabular", "source": "openml:adult:2", "num_classes": 2},
    "letter": {"modality": "tabular", "source": "openml:letter:1", "num_classes": 26},
}


_IMAGE = {
    "model": "resnet18",
    "epochs": 10,
    "early_epochs": 2,
    "batch_size": 128,
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "weight_decay": 0.0005,
    "momentum": 0.9,
    "milestones": (5, 8),
    "lr_gamma": 0.1,
    "num_workers": 4,
}

_TEXT = {
    "model": "FacebookAI/roberta-base",
    "epochs": 3,
    "early_epochs": 1,
    "batch_size": 32,
    "optimizer": "adamw",
    "learning_rate": 2e-05,
    "weight_decay": 0.01,
    "max_length": 128,
    "num_workers": 2,
}

_TABULAR = {
    "model": "mlp",
    "epochs": 50,
    "early_epochs": 10,
    "batch_size": 256,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "hidden_dims": (256, 128, 64),
    "num_workers": 0,
}

MODELS = {
    "cifar10": dict(_IMAGE),
    "cifar100": dict(_IMAGE),
    "atis": dict(_TEXT),
    "qnli": dict(_TEXT),
    "adult": dict(_TABULAR),
    "letter": dict(_TABULAR),
}
