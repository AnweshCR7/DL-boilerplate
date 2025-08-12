"""
Fashion-MNIST dataset for the generic DataModule.
"""

from typing import Any, Dict, Optional

import albumentations as A
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class FashionMNISTDataset(Dataset):
    """
    Fashion-MNIST dataset that can be used with the generic DataModule.
    
    Args:
        mode: Dataset mode ('train', 'val', 'test')
        data_dir: Directory to store/load Fashion MNIST data
        download: Whether to download the dataset if not found
        image_size: Target image size (height, width)
        mean: Normalization mean values
        std: Normalization std values
        augmentations: Augmentation configuration
    """

    def __init__(
        self,
        mode: str,
        data_dir: str = "data/",
        download: bool = True,
        image_size: tuple[int, int] = (28, 28),
        mean: list[float] = [0.2860],
        std: list[float] = [0.3530],
        augmentations: dict[str, any] | None = None,
        **kwargs,
    ):
        super().__init__()
        
        self.mode = mode
        self.data_dir = data_dir
        self.download = download
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augmentations = augmentations or {}
        
        # Setup dataset and transforms
        self._setup_dataset()
        self._setup_transforms()
    
    def _setup_dataset(self):
        """Setup the base dataset."""
        if self.mode in ["train", "val"]:
            # For train/val, we'll use the training split and split it ourselves
            self.base_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir,
                train=True,
                transform=None,
                download=self.download,
            )
            
            # Split trainval into train/val (80/20 split)
            total_size = len(self.base_dataset)
            train_size = int(0.8 * total_size)
            
            if self.mode == "train":
                self.indices = list(range(train_size))
            else:  # val
                self.indices = list(range(train_size, total_size))
                
        elif self.mode == "test":
            self.base_dataset = torchvision.datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                transform=None,
                download=self.download,
            )
            self.indices = list(range(len(self.base_dataset)))
    
    def _setup_transforms(self):
        """Setup transforms based on mode."""
        if self.mode == "train":
            # Training transforms with augmentations
            self.transforms = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=self.augmentations.get("horizontal_flip", 0.0)),
                A.Rotate(limit=self.augmentations.get("rotation", 0), p=0.5),
                A.ColorJitter(
                    brightness=self.augmentations.get("brightness", 0.0),
                    contrast=self.augmentations.get("contrast", 0.0),
                    p=0.5,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
        else:
            # Validation/Test transforms (no augmentation)
            self.transforms = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        # Get the actual index
        actual_idx = self.indices[idx]
        
        # Get image and label from base dataset
        image, label = self.base_dataset[actual_idx]
        
        # Convert PIL to numpy
        image = np.array(image, dtype=np.float32)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor manually if no transforms
            image = torch.from_numpy(image).float() / 255.0
            # Add channel dimension if needed
            if len(image.shape) == 2:
                image = image.unsqueeze(0)
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
    
    def collate_fn(self, batch):
        """Custom collate function for the dataset."""
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return {
            "image": images,
            "label": labels,
        }
