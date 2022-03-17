"""
dataset
"""

import numpy as np
import torch
import torch.utils.data as data

from helper import visualize, PathConfig
from preprocess import load_labels, load_images, preprocess


def gen_split(train_nums: int):
    import random
    idx = list(range(train_nums))
    random.shuffle(idx)
    split = int(train_nums * 0.8)
    train_idx, val_idx = np.array(idx[: split]), np.array(idx[split:])
    np.savez(PathConfig.base.joinpath("trainval.npz"), train=train_idx, val=val_idx)
    return train_idx, val_idx

def load_split():
    split = np.load(PathConfig.base.joinpath("trainval.npz"))
    return split["train"], split["val"]



class MnistDataset(data.Dataset):
    def __init__(self, split: str):
        assert split in ["train", "validation", "test"], f"Invalid split"
        # load
        if split == "test":
            images, labels = load_images(PathConfig.test_images), load_labels(PathConfig.test_labels)
        else:
            images, labels = load_images(PathConfig.train_images), load_labels(PathConfig.train_labels)
            # get train val split idx
            if PathConfig.base.joinpath("trainval.npz").exists():
                train, val = load_split()
            else:
                train, val = gen_split(train_nums=len(images))
            # return image and idx
            if split == "train":
                images, labels = images[train], labels[train]
            else:
                images, labels = images[val], labels[val]

        self.images, self.labels = preprocess(images), labels
        return None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).unsqueeze(0), torch.Tensor([self.labels[idx]])


if __name__ == "__main__":
    x: torch.Tensor
    y: torch.Tensor
    set = MnistDataset(split="train")
    loader = data.DataLoader(set, shuffle=True, batch_size=64, num_workers=1)
    for x, y in loader:
        x, y = x.to("cuda:0"), y.to("cuda:0")
        print(x.shape)
        print(y.shape)
    visualize(x[0], y[0])

