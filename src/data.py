"""PyTorch Lightning DataModule for the project."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import OxfordIIITPet


class SurgicalToolsDataset(Dataset):
    """Dataset class for surgical tools segmentation."""
    
    def __init__(
        self,
        data_path: str,
        image_size: Tuple[int, int] = (512, 512),
        transforms: Optional[A.Compose] = None,
        is_training: bool = True,
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the dataset directory
            image_size: Target image size (height, width)
            transforms: Albumentations transforms
            is_training: Whether this is training data (affects augmentations)
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.transforms = transforms
        self.is_training = is_training
        
        # Find all image files
        self.image_paths = list(self.data_path.glob("*.png")) + list(self.data_path.glob("*.jpg"))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_path}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        
        # For now, create a dummy mask (replace with actual mask loading)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Basic preprocessing
            image = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])(image=image, mask=mask)
            image = image["image"]
            mask = torch.from_numpy(mask).long()
        
        # Extract label from filename (placeholder implementation)
        # Replace this with your actual label extraction logic
        label = self._extract_label_from_filename(image_path.name)
        
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.float32),
            "image_path": str(image_path),
        }
    
    def _extract_label_from_filename(self, filename: str) -> float:
        """Extract label from filename.
        
        This is a placeholder implementation. Replace with your actual logic.
        """
        # Example: extract a value from filename like "image_0.5.png"
        try:
            parts = filename.split("_")
            if len(parts) >= 2:
                return float(parts[-1].split(".")[0])
        except (ValueError, IndexError):
            pass
        return 0.0  # Default value


class SurgicalToolsDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for surgical tools dataset."""
    
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: Tuple[int, int] = (512, 512),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        augmentations: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the DataModule.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            test_data_path: Path to test data (optional)
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
        
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augmentations = augmentations or {}
        
        # Initialize datasets as None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for different stages."""
        if stage == "fit" or stage is None:
            # Training transforms
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
            
            # Validation transforms
            val_transforms = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
            
            self.train_dataset = SurgicalToolsDataset(
                data_path=self.train_data_path,
                image_size=self.image_size,
                transforms=train_transforms,
                is_training=True,
            )
            
            self.val_dataset = SurgicalToolsDataset(
                data_path=self.val_data_path,
                image_size=self.image_size,
                transforms=val_transforms,
                is_training=False,
            )
        
        if stage == "test" or stage is None:
            if self.test_data_path:
                test_transforms = A.Compose([
                    A.Resize(self.image_size[0], self.image_size[1]),
                    A.Normalize(mean=self.mean, std=self.std),
                    ToTensorV2(),
                ])
                
                self.test_dataset = SurgicalToolsDataset(
                    data_path=self.test_data_path,
                    image_size=self.image_size,
                    transforms=test_transforms,
                    is_training=False,
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
        if self.test_dataset is None:
            raise ValueError("Test dataset not initialized. Make sure test_data_path is provided.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


class MNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MNIST dataset."""
    
    def __init__(
        self,
        data_dir: str = "data/",
        download: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: Tuple[int, int] = (28, 28),
        mean: List[float] = [0.1307],
        std: List[float] = [0.3081],
        augmentations: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the MNIST DataModule.
        
        Args:
            data_dir: Directory to store/load MNIST data
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
        self.pin_memory = pin_memory
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
        """Download MNIST dataset if needed."""
        datasets.MNIST(self.data_dir, train=True, download=self.download)
        datasets.MNIST(self.data_dir, train=False, download=self.download)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for different stages."""
        if stage == "fit" or stage is None:
            # Training transforms
            train_transforms_list = [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
            
            # Add augmentations if specified
            if self.augmentations.get("train", {}):
                aug_config = self.augmentations["train"]
                if "rotation" in aug_config:
                    train_transforms_list.insert(-2, transforms.RandomRotation(aug_config["rotation"]))
                if "translate" in aug_config:
                    train_transforms_list.insert(-2, transforms.RandomAffine(
                        degrees=0, translate=aug_config["translate"]
                    ))
                if "scale" in aug_config:
                    train_transforms_list.insert(-2, transforms.RandomAffine(
                        degrees=0, scale=aug_config["scale"]
                    ))
            
            train_transforms = transforms.Compose(train_transforms_list)
            
            # Validation transforms (no augmentation)
            val_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            
            # Load full training dataset
            full_train_dataset = datasets.MNIST(
                self.data_dir, train=True, transform=train_transforms
            )
            
            # Split training data into train/val (80/20 split)
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            
            self.train_dataset, _ = torch.utils.data.random_split(
                full_train_dataset, [train_size, val_size]
            )
            
            # Apply validation transforms to validation split
            val_dataset_full = datasets.MNIST(
                self.data_dir, train=True, transform=val_transforms
            )
            _, self.val_dataset = torch.utils.data.random_split(
                val_dataset_full, [train_size, val_size]
            )
        
        if stage == "test" or stage is None:
            test_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            
            self.test_dataset = datasets.MNIST(
                self.data_dir, train=False, transform=test_transforms
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
        image_size: Tuple[int, int] = (256, 256),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        augmentations: Optional[Dict[str, Any]] = None,
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
        self.pin_memory = pin_memory
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
    
    def setup(self, stage: Optional[str] = None) -> None:
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
    
    def __init__(self, base_dataset, indices: List[int], transforms: Optional[A.Compose] = None):
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
            # Ensure mask is long tensor
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()
        else:
            # Convert to tensors manually if no transforms
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            "image": image,
            "mask": mask,
        }
