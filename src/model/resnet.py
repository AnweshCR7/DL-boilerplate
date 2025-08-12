"""
ResNet model.
"""

import torch


class ResNetBlock(torch.nn.Module):
    """
    ResNet block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the kernel.
        padding (int): Padding of the kernel
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Forward pass.
        """

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class ResNet(torch.nn.Module):
    """
    ResNet model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the kernel.
        padding (int): Padding of the kernel.
        num_classes (int): Number of classes.
        num_blocks (int): Number of blocks
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_classes: int = 10,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.resnet_blocks = torch.nn.Sequential(
            *[
                ResNetBlock(out_channels, out_channels, kernel_size, stride, padding)
                for _ in range(num_blocks)
            ]
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(out_channels, num_classes)

    def forward(self, x):
        """
        Forward pass.
        """

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.resnet_blocks(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
