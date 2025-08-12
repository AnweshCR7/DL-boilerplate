"""Dataset classes for the project."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet


class OxfordPetDataset(Dataset):
    """Oxford Pet dataset that can be used with the generic DataModule."""
    
    def __init__(
        self,
        mode: str,
        data_dir: str = "data/",
        download: bool = True,
        image_size: tuple[int, int] = (256, 256),
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
        augmentations: dict[str, any] | None = None,
        **kwargs,
    ):
        """Initialize the Oxford Pet Dataset.
        
        Args:
            mode: Dataset mode ('train', 'val', 'test')
            data_dir: Directory to store/load Oxford Pet data
            download: Whether to download the dataset if not found
            image_size: Target image size (height, width)
            mean: Normalization mean values
            std: Normalization std values
            augmentations: Augmentation configuration
        """
        super().__init__()
        
        self.mode = mode
        self.data_dir = data_dir
        self.download = download
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augmentations = augmentations or {}
        
        # Download dataset if needed
        if self.download:
            self._download_dataset()
        
        # Setup dataset and transforms
        self._setup_dataset()
        self._setup_transforms()
    
    def _download_dataset(self):
        """Download Oxford Pet dataset if needed."""
        if self.mode in ["train", "val"]:
            OxfordIIITPet(self.data_dir, split="trainval", target_types="segmentation", download=self.download)
        if self.mode == "test":
            OxfordIIITPet(self.data_dir, split="test", target_types="segmentation", download=self.download)
    
    def _setup_dataset(self):
        """Setup the base dataset and indices."""
        if self.mode in ["train", "val"]:
            # Load the full trainval dataset
            self.base_dataset = OxfordIIITPet(
                self.data_dir, 
                split="trainval", 
                target_types="segmentation", 
                download=False
            )
            
            # Split trainval into train/val (80/20 split)
            total_size = len(self.base_dataset)
            train_size = int(0.8 * total_size)
            
            if self.mode == "train":
                self.indices = list(range(train_size))
            else:  # val
                self.indices = list(range(train_size, total_size))
                
        elif self.mode == "test":
            self.base_dataset = OxfordIIITPet(
                self.data_dir, 
                split="test", 
                target_types="segmentation", 
                download=False
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
                    saturation=self.augmentations.get("saturation", 0.0),
                    hue=self.augmentations.get("hue", 0.0),
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
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        # Get the actual index
        actual_idx = self.indices[idx]
        
        # Get image and mask from base dataset
        image, mask = self.base_dataset[actual_idx]
        
        # Convert PIL to numpy
        image = np.array(image)
        mask = np.array(mask)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            if isinstance(mask, torch.Tensor):
                # Remove any singleton dimensions and ensure 2D
                mask = mask.squeeze()
                # If somehow still has more than 2 dimensions, take the first channel
                if len(mask.shape) > 2:
                    mask = mask[0] if mask.shape[0] == 1 else mask[0]
                mask = mask.long()
            else:
                # Convert numpy to tensor
                mask = torch.from_numpy(mask).long()
                # Ensure 2D
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
        else:
            # Convert to tensors manually if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        # Final safety check: ensure mask is exactly 2D
        if len(mask.shape) != 2:
            raise ValueError(f"Mask must be 2D (H, W), got shape {mask.shape}. Original mask shape after transforms: {mask.shape}")
        
        return {
            "image": image,
            "mask": mask,
        }
    
    def collate_fn(self, batch):
        """Custom collate function for the dataset."""
        images = torch.stack([item["image"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])
        return {
            "image": images,
            "mask": masks,
        }
