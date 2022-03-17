"""
Preprocess is used to convert original data into paired image and label
"""

import struct
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

from helper import PathConfig, visualize


def load_images(path: Path) -> np.ndarray:
    with path.open(mode="rb") as f:
        buffer = f.read()
    # deserialize image
    magic, nums, rows, cols = struct.unpack_from(">IIII", buffer, 0)
    bits = nums * rows * cols
    images = struct.unpack_from(f">{bits}B", buffer, struct.calcsize(">IIII"))
    images = np.array(images).reshape(nums, rows, cols)
    return images


def load_labels(path: Path):
    with path.open(mode="rb") as f:
        buffer = f.read()
    magic, num = struct.unpack_from(">II", buffer, 0)
    labels = struct.unpack_from(f">{num}B", buffer, struct.calcsize(">II"))
    labels = np.array(labels)
    return labels


def preprocess(images: np.ndarray):
    images = images / 255
    return images


if __name__ == "__main__":
    images = load_images(PathConfig.train_images)
    labels = load_labels(PathConfig.train_labels)
    visualize(images[1], labels[1])
    # print(preprocess(images)[0], labelsl)
