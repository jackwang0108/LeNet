"""
Pathconfig configures paths of the project
"""

from typing import *
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


class PathConfig:
    # Project
    base: Path = Path(__file__).resolve().parent

    # Dataset
    dataset: Path = base.joinpath("dataset")
    train_images: Path = dataset / "train_images"
    train_labels: Path = dataset / "train_labels"
    test_images: Path = dataset / "test_images"
    test_labels: Path = dataset / "test_labels"

    # checkpoint
    checkpoint: Path = base.joinpath("checkpoint")

    # runs
    runs: Path = base.joinpath("runs")


def visualize(image: np.ndarray, label: Union[None, int] = None):
    image = image.cpu() if isinstance(image, torch.Tensor) else image
    label = label.cpu() if isinstance(label, torch.Tensor) else label
    plt.imshow(image)
    if label is not None:
        plt.title(f"{label}")
    plt.show()


if __name__ == "__main__":
    for name, value in PathConfig.__dict__.items():
        if name[:2] != "__":
            print(f"{name}: {value}, {value.exists()}")
