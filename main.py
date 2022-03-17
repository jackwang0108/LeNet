import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import network
from network import LeNet
from dataset import MnistDataset
from helper import visualize, PathConfig


class Trainer:
    available_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __init__(self, net: nn.Module):
        self.network = net.to(self.available_device)
        # paths
        self.checkpoint_path = PathConfig.checkpoint / self.network.__class__.__name__ / (self.network.__class__.__name__ + "-best.pt")
        self.run_path = PathConfig.runs / self.network.__class__.__name__

        # summary
        self.writer = SummaryWriter(log_dir=self.run_path.__str__())

        # loader
        self.train_loader = data.DataLoader(MnistDataset(split="train"), batch_size=64, shuffle=True, num_workers=1)
        self.val_loader = data.DataLoader(MnistDataset(split="validation"), batch_size=64, shuffle=False, num_workers=1)
        self.test_loader = data.DataLoader(MnistDataset(split="test"), batch_size=64, shuffle=False, num_workers=1)

        # loss function
        self.lossfunc = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = optim.SGD(params=self.network.parameters(), lr=1e-4)

    def __del__(self):
        self.writer.close()

    def train(self, n_epoch: int, early_stop: int):
        x: torch.Tensor
        y: torch.Tensor
        max_acc = 0
        early_stop_cnt = 0
        for epoch in (tt := tqdm.trange(n_epoch)):
            # train
            for step, (x, y) in enumerate(self.train_loader):
                x, y = x.to(device=self.available_device, dtype=self.network.dtype), y.to(device=self.available_device,
                                                                                          dtype=self.network.dtype)
                self.network.zero_grad()
                y_pred: torch.Tensor = self.network(x)
                train_loss: torch.Tensor = self.lossfunc(y_pred, y.squeeze().long())
                train_loss.backward()
                self.optimizer.step()
                self.writer.add_scalar(tag="loss/train", scalar_value=train_loss.item(),
                                       global_step=len(self.train_loader) * epoch + step)

            # val
            acc = 0
            all_example = 0
            with torch.no_grad():
                for step, (x, y) in enumerate(self.val_loader):
                    x, y = x.to(device=self.available_device, dtype=self.network.dtype), y.to(
                        device=self.available_device,
                        dtype=self.network.dtype)
                    y_pred: torch.Tensor = self.network(x)
                    val_loss: torch.Tensor = self.lossfunc(y_pred, y.squeeze().long())
                    self.writer.add_scalar(tag="loss/val", scalar_value=train_loss.item(),
                                           global_step=len(self.train_loader) * epoch + step)
                    # get acc
                    acc += (y_pred.argmax(dim=1) == y.squeeze()).sum()
                    all_example += x.shape[0]
            self.writer.add_scalar(tag="acc", scalar_value=(cur_acc := acc / all_example), global_step=epoch)

            if cur_acc > max_acc:
                max_acc = cur_acc
                early_stop_cnt = 0
                if not self.checkpoint_path.parent.exists():
                    self.checkpoint_path.parent.mkdir(parents=True)
                torch.save(self.network.state_dict(), self.checkpoint_path)
            else:
                early_stop_cnt += 1
            tt.write(
                f"Epoch [{epoch:>5d}|{n_epoch:>5d}], train_loss {train_loss:>7.4f}, val_loss {val_loss:>7.4f}, "
                f"early_stop_cnt: [{early_stop_cnt:>5d}|{early_stop:>5d}]")
            if early_stop_cnt >= early_stop:
                break


if __name__ == "__main__":
    net = network.LeNet()
    trainer = Trainer(net)
    trainer.train(n_epoch=1000, early_stop=300)