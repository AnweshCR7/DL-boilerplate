"""
Simple U-Net model for image segmentation.
Lightweight implementation for testing PyTorch Lightning pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleUNet(nn.Module):
    """
    Simple U-Net model for image segmentation.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 3 for Oxford Pet)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        
        # Decoder (upsampling path)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 256 = 128 (skip) + 128 (up)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)   # 128 = 64 (skip) + 64 (up)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)    # 64 = 32 (skip) + 32 (up)
        
        # Final output layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder with skip connections
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        return self.final_conv(dec1)


def create_simple_unet(in_channels: int = 3, num_classes: int = 3) -> SimpleUNet:
    """Factory function to create a simple U-Net model."""
    return SimpleUNet(in_channels=in_channels, num_classes=num_classes)
