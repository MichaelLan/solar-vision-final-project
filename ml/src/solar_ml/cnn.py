"""Small CNN for 24x40 thermal images (training only; inference is exported to JSON)."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from solar_ml.constants import HEIGHT, WIDTH

PER_IMAGE_EPS = 1e-6

RECIPE = {
    "architecture": "conv3x3(1->w)-relu-maxpool2 / conv3x3(w->2w)-relu-maxpool2 / "
    "conv3x3(2w->4w)-relu / global_avg_pool / dropout(0.2, training only) / dense(4w->4) / "
    "softmax; padding 1, stride 1, no batch normalization",
    "input_normalization": "per_image: (x - mean(x)) / (std(x) + 1e-6), population std",
    "augmentation": "independent random horizontal and vertical flips, p=0.5 each",
    "optimizer": "AdamW lr=3e-3 weight_decay=1e-3",
    "schedule": "OneCycleLR max_lr=3e-3 over all steps",
    "loss": "cross_entropy label_smoothing=0.05",
    "batch_size": 64,
    "epochs": 40,
    "early_stopping": False,
    "checkpoint": "final epoch",
    "torch_threads": 4,
}


class SmallCNN(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.width = width
        self.conv1 = nn.Conv2d(1, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, 2 * width, 3, padding=1)
        self.conv3 = nn.Conv2d(2 * width, 4 * width, 3, padding=1)
        self.dense = nn.Linear(4 * width, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.dropout(x.mean((2, 3)), 0.2, self.training)
        return self.dense(x)


def per_image_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / (std + PER_IMAGE_EPS)


def to_tensor(features: np.ndarray) -> torch.Tensor:
    normalized = per_image_normalize(features).reshape(-1, 1, HEIGHT, WIDTH)
    return torch.tensor(normalized, dtype=torch.float32)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_cnn(
    features: np.ndarray,
    labels: np.ndarray,
    width: int,
    seed: int,
    epochs: int = RECIPE["epochs"],
    batch_size: int = RECIPE["batch_size"],
) -> tuple[SmallCNN, float]:
    torch.set_num_threads(RECIPE["torch_threads"])
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    inputs = to_tensor(features)
    targets = torch.tensor(np.asarray(labels), dtype=torch.long)
    model = SmallCNN(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    steps_per_epoch = (len(inputs) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, total_steps=epochs * steps_per_epoch
    )
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(inputs), generator=generator)
        for offset in range(0, len(inputs), batch_size):
            batch = order[offset : offset + batch_size]
            x = inputs[batch].clone()
            flip = torch.rand(len(batch), generator=generator) < 0.5
            x[flip] = x[flip].flip(3)
            flip = torch.rand(len(batch), generator=generator) < 0.5
            x[flip] = x[flip].flip(2)
            loss = F.cross_entropy(model(x), targets[batch], label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.eval()
    return model, time.perf_counter() - start


def predict_proba(model: SmallCNN, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return F.softmax(model(to_tensor(features)), dim=1).double().numpy()
