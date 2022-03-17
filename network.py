import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), padding=0, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.linear1 = nn.Linear(in_features=400, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=10)

        self.activation = nn.Tanh()

        for module in self.modules():
            print(module)
            if isinstance(module, nn.Module):
                if (weight := getattr(module, "weight", None)) is not None:
                    self.dtype = weight.dtype
                    break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activation(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        return self.output(x)


if __name__ == "__main__":
    LeNet()