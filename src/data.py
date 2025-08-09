"""PyTorch Lightning DataModule for the project."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet


class OxfordPetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Oxford-IIIT Pet segmentation dataset."""
    
    def __init__(
        self,
        data_dir: str = "data/",
        download: bool = True,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: tuple[int, int] = (256, 256),
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
        augmentations: dict[str, any] | None = None,
        **kwargs,
    ):
        """Initialize the Oxford Pet DataModule.
        
        Args:
            data_dir: Directory to store/load Oxford Pet data
            download: Whether to download the dataset if not found
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            pin_memory: Whether to pin memory
            persistent_workers: Whether to use persistent workers
            image_size: Target image size (height, width)
            mean: Normalization mean values
            std: Normalization std values
            augmentations: Augmentation configuration
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.data_dir = data_dir
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Disable pin_memory for MPS compatibility
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()
        self.persistent_workers = persistent_workers
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augmentations = augmentations or {}
        
        # Initialize datasets as None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self) -> None:
        """Download Oxford Pet dataset if needed."""
        OxfordIIITPet(self.data_dir, split="trainval", target_types="segmentation", download=self.download)
        OxfordIIITPet(self.data_dir, split="test", target_types="segmentation", download=self.download)
    
    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for different stages."""
        if stage == "fit" or stage is None:
            # Training transforms with augmentations
            train_transforms = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=self.augmentations.get("train", {}).get("horizontal_flip", 0.0)),
                A.Rotate(limit=self.augmentations.get("train", {}).get("rotation", 0), p=0.5),
                A.ColorJitter(
                    brightness=self.augmentations.get("train", {}).get("brightness", 0.0),
                    contrast=self.augmentations.get("train", {}).get("contrast", 0.0),
                    saturation=self.augmentations.get("train", {}).get("saturation", 0.0),
                    hue=self.augmentations.get("train", {}).get("hue", 0.0),
                    p=0.5,
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
            
            # Validation transforms (no augmentation)
            val_transforms = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
            
            # Load the full trainval dataset
            full_dataset = OxfordIIITPet(
                self.data_dir, 
                split="trainval", 
                target_types="segmentation", 
                download=False
            )
            
            # Split trainval into train/val (80/20 split)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            # Create wrapped datasets with transforms
            self.train_dataset = OxfordPetSegmentationDataset(
                full_dataset, 
                indices=list(range(train_size)),
                transforms=train_transforms
            )
            
            self.val_dataset = OxfordPetSegmentationDataset(
                full_dataset,
                indices=list(range(train_size, train_size + val_size)),
                transforms=val_transforms
            )
        
        if stage == "test" or stage is None:
            test_transforms = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
            
            test_dataset = OxfordIIITPet(
                self.data_dir, 
                split="test", 
                target_types="segmentation", 
                download=False
            )
            
            self.test_dataset = OxfordPetSegmentationDataset(
                test_dataset,
                indices=list(range(len(test_dataset))),
                transforms=test_transforms
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class OxfordPetSegmentationDataset(Dataset):
    """Wrapper dataset for Oxford Pet with albumentations transforms."""
    
    def __init__(self, base_dataset, indices: list[int], transforms: A.Compose | None = None):
        """Initialize the wrapper dataset.
        
        Args:
            base_dataset: Base Oxford Pet dataset
            indices: Indices to use from the base dataset
            transforms: Albumentations transforms to apply
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.transforms = transforms
    
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
