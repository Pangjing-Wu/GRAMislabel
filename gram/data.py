from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache import load_raw_data


@dataclass
class TaskData:
    name: str
    modality: str
    train_x: Any
    clean_train_y: np.ndarray
    noise_x: Any
    num_classes: int
    metadata: dict = field(default_factory=dict)

    @property
    def sample_count(self):
        return len(self.clean_train_y)


class ImageInputs:

    def __init__(self, source, image_column: str | None = None):
        self.source = source
        self.image_column = image_column

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        row = self.source[int(index)]
        return row[self.image_column] if self.image_column is not None else row[0]


def _pooled_cifar_features(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32) / 255.0
    return values.reshape(-1, 8, 4, 8, 4, 3).mean((2, 4)).reshape(len(values), -1)


def _pooled_huggingface_image_features(dataset, image_column: str) -> np.ndarray:
    features = np.empty((len(dataset), 8 * 8 * 3), dtype=np.float32)
    for start in range(0, len(dataset), 1000):
        stop = min(start + 1000, len(dataset))
        images = np.stack(
            [np.asarray(dataset[index][image_column]) for index in range(start, stop)]
        )
        features[start:stop] = _pooled_cifar_features(images)
    return features


def _hashed_text_features(first: list[str], second: list[str] | None):
    from sklearn.feature_extraction.text import HashingVectorizer

    texts = (
        first if second is None else [f"{a} [SEP] {b}" for a, b in zip(first, second)]
    )
    return HashingVectorizer(n_features=256, alternate_sign=True, norm="l2").transform(
        texts
    )


def _encode_texts(first, second, config, cache):
    from transformers import AutoTokenizer

    options = {"cache_dir": str(cache)}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], local_files_only=True, **options
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(config["model"], **options)
    encoded = tokenizer(
        first,
        text_pair=second,
        max_length=config["max_length"],
        padding="max_length",
        truncation=True,
    )
    return {key: np.asarray(value, dtype=np.int64) for key, value in encoded.items()}


def _load_tabular(name, raw, seed):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if name == "adult":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder

        frame = raw.data
        target = raw.target.astype(str).str.strip().str.rstrip(".")
        labels = (target == ">50K").astype(np.int64).to_numpy()
        train_x, _, train_y, _ = train_test_split(
            frame,
            labels,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )
        categorical = frame.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
        numerical = [column for column in frame.columns if column not in categorical]
        processor = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    numerical,
                ),
                (
                    "categorical",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                    list(categorical),
                ),
            ],
            sparse_threshold=0.0,
        )
        train_x = processor.fit_transform(train_x).astype(np.float32)
    else:
        values = np.asarray(raw.data, dtype=np.float32)
        labels = LabelEncoder().fit_transform(raw.target).astype(np.int64)
        train_x, _, train_y, _ = train_test_split(
            values,
            labels,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )
        train_x = StandardScaler().fit_transform(train_x).astype(np.float32)
    return train_x, train_y


def load_data(
    name: str, root: Path, seed: int, dataset_config: dict, model_config: dict
):
    raw = load_raw_data(name, root, dataset_config)
    modality = dataset_config["modality"]
    metadata = {}
    if modality == "image":
        train = raw["train"]
        column = dataset_config["image_column"]
        inputs = ImageInputs(train, column)
        labels = np.asarray(train[dataset_config["label_column"]], dtype=np.int64)
        noise_features = _pooled_huggingface_image_features(train, column)
    elif modality == "text":
        if name == "atis":
            rows = [row for row in raw["train"] if "+" not in row["intent"]]
            classes = sorted({row["intent"] for row in rows})
            if len(classes) != dataset_config["num_classes"]:
                raise ValueError(
                    f"ATIS expected 17 train intents, found {len(classes)}"
                )
            label_id = {value: index for index, value in enumerate(classes)}
            first, second = [row["text"] for row in rows], None
            labels = np.asarray(
                [label_id[row["intent"]] for row in rows], dtype=np.int64
            )
        else:
            train = raw["train"]
            first, second = train["question"], train["sentence"]
            labels = np.asarray(train["label"], dtype=np.int64)
        inputs = _encode_texts(first, second, model_config, root / "huggingface")
        noise_features = _hashed_text_features(first, second)
    else:
        inputs, labels = _load_tabular(name, raw, seed)
        noise_features = inputs
        metadata["input_dim"] = int(inputs.shape[1])
    return TaskData(
        name,
        modality,
        inputs,
        labels,
        noise_features,
        dataset_config["num_classes"],
        metadata,
    )
