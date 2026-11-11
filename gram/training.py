from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from .models import build_model


@dataclass
class TrainingArtifacts:
    probability: np.ndarray
    embedding: np.ndarray
    early_loss: np.ndarray
    margin_trajectory: np.ndarray
    gradient_trajectory: np.ndarray


class ModelDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if isinstance(self.inputs, dict):
            values = {
                key: torch.as_tensor(array[index], dtype=torch.long)
                for key, array in self.inputs.items()
            }
        else:
            values = self.inputs[index]
            values = (
                self.transform(values)
                if self.transform is not None
                else torch.as_tensor(values, dtype=torch.float32)
            )
        return index, values, int(self.labels[index])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _image_transforms(dataset: str):
    from torchvision import transforms

    if dataset == "cifar10":
        mean, std = ((0.4914, 0.4822, 0.4465), (0.247, 0.2435, 0.2616))
    else:
        mean, std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    evaluation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return (train, evaluation)


def _move(values, device: torch.device):
    if isinstance(values, dict):
        return {
            key: tensor.to(device, non_blocking=True) for key, tensor in values.items()
        }
    return values.to(device, non_blocking=True)


def _features_and_logits(model, values):
    if isinstance(values, dict):
        features = model.forward_features(**values)
    else:
        features = model.forward_features(values)
    return (features, model.classifier(features))


def _training_features_and_logits(model, values):
    if isinstance(values, dict):
        features = model.forward_features(**values)
    else:
        features = model.forward_features(values)
    classifier_features = (
        model.dropout(features) if hasattr(model, "dropout") else features
    )
    return (classifier_features, model.classifier(classifier_features))


def _batch_plan(
    sample_count: int,
    batch_size: int,
    seed: int,
    epoch: int,
):
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(sample_count)
    return [
        order[start : start + batch_size]
        for start in range(0, sample_count, batch_size)
    ]


def _loader(dataset, *, batch_size=None, batches=None, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
    }
    if batches is not None:
        return DataLoader(dataset, batch_sampler=batches, **common)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **common)


def _optimizer(model, config: dict):
    name = config["optimizer"]
    active_learning_rate = config["learning_rate"]
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=active_learning_rate,
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=True,
        )
    optimizer_class = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
    return optimizer_class(
        model.parameters(), lr=active_learning_rate, weight_decay=config["weight_decay"]
    )


def train_task_model(
    data, labels: np.ndarray, config: dict, seed: int, cache_dir: Path
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(data, config, str(cache_dir)).to(device)
    optimizer = _optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config["milestones"], gamma=config["lr_gamma"]
        )
        if config["optimizer"] == "sgd"
        else None
    )
    train_transform = evaluation_transform = None
    if data.modality == "image":
        train_transform, evaluation_transform = _image_transforms(data.name)
    train_set = ModelDataset(data.train_x, labels, train_transform)
    count, epochs = len(labels), config["epochs"]
    early_sum = np.zeros(count, dtype=np.float64)
    early_count = np.zeros(count, dtype=np.int64)
    margin_trajectory = np.empty((count, epochs), dtype=np.float32)
    gradient_trajectory = np.empty((count, epochs), dtype=np.float32)
    for epoch in range(1, epochs + 1):
        epoch_seed = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
        seed_everything(epoch_seed)
        loader = _loader(
            train_set,
            batches=_batch_plan(count, config["batch_size"], seed, epoch),
            workers=config["num_workers"],
            seed=epoch_seed,
        )
        model.train()
        for ids, values, batch_labels in loader:
            positions = ids.numpy()
            values = _move(values, device)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            features, logits = _training_features_and_logits(model, values)
            losses = functional.cross_entropy(logits, batch_labels, reduction="none")
            if epoch <= config["early_epochs"]:
                early_sum[positions] += losses.detach().double().cpu().numpy()
                early_count[positions] += 1
            detached_logits = logits.detach().float().cpu().numpy().copy()
            row, assigned_labels = np.arange(len(positions)), batch_labels.cpu().numpy()
            assigned = detached_logits[row, assigned_labels].copy()
            detached_logits[row, assigned_labels] = -np.inf
            margin_trajectory[positions, epoch - 1] = assigned - detached_logits.max(
                axis=1
            )
            residual = torch.softmax(logits.detach(), dim=1).float().cpu().numpy()
            residual[row, assigned_labels] -= 1.0
            detached_features = features.detach().float().cpu().numpy()
            gradient_trajectory[positions, epoch - 1] = np.linalg.norm(
                residual, axis=1
            ) * np.sqrt(
                1.0 + np.einsum("ij,ij->i", detached_features, detached_features)
            )
            losses.mean().backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        print(f"Training epoch {epoch}/{epochs}", flush=True)
    probability = np.empty((count, data.num_classes), dtype=np.float32)
    embedding = np.empty((count, model.embedding_dim), dtype=np.float32)
    evaluation_set = ModelDataset(data.train_x, labels, evaluation_transform)
    loader = _loader(
        evaluation_set,
        batch_size=config["batch_size"],
        workers=config["num_workers"],
        seed=0,
    )
    model.eval()
    with torch.no_grad():
        for ids, values, _ in loader:
            features, logits = _features_and_logits(model, _move(values, device))
            probability[ids.numpy()] = torch.softmax(logits, dim=1).cpu().numpy()
            embedding[ids.numpy()] = features.float().cpu().numpy()
    return TrainingArtifacts(
        probability,
        embedding,
        early_sum / np.maximum(early_count, 1),
        margin_trajectory,
        gradient_trajectory,
    )
