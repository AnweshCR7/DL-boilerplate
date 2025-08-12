#!/usr/bin/env python3
"""
Test script to verify the refactored DataModule works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.datasets.datamodule import DataModule
from src.datasets.data import OxfordPetDataset


def test_dataset_creation():
    """Test that the OxfordPetDataset can be created with different modes."""
    print("Testing dataset creation...")
    
    # Test training dataset
    train_dataset = OxfordPetDataset(
        mode="train",
        data_dir="data/",
        download=False,  # Don't download for testing
        image_size=(256, 256)
    )
    print(f"✓ Training dataset created with {len(train_dataset)} samples")
    
    # Test validation dataset
    val_dataset = OxfordPetDataset(
        mode="val",
        data_dir="data/",
        download=False,
        image_size=(256, 256)
    )
    print(f"✓ Validation dataset created with {len(val_dataset)} samples")
    
    # Test test dataset
    test_dataset = OxfordPetDataset(
        mode="test",
        data_dir="data/",
        download=False,
        image_size=(256, 256)
    )
    print(f"✓ Test dataset created with {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


def test_datamodule_creation():
    """Test that the generic DataModule can be created."""
    print("\nTesting DataModule creation...")
    
    # Test training DataModule
    train_datamodule = DataModule(
        mode="train",
        dataset="src.datasets.data.OxfordPetDataset",
        dataset_params={
            "data_dir": "data/",
            "download": False,
            "image_size": (256, 256)
        },
        dataloader_params={
            "batch_size": 2,
            "num_workers": 0  # Use 0 for testing
        }
    )
    print("✓ Training DataModule created successfully")
    
    # Test test DataModule
    test_datamodule = DataModule(
        mode="test",
        dataset="src.datasets.data.OxfordPetDataset",
        dataset_params={
            "data_dir": "data/",
            "download": False,
            "image_size": (256, 256)
        },
        dataloader_params={
            "batch_size": 2,
            "num_workers": 0
        }
    )
    print("✓ Test DataModule created successfully")
    
    return train_datamodule, test_datamodule


def test_dataloaders():
    """Test that dataloaders work correctly."""
    print("\nTesting dataloaders...")
    
    train_datamodule, test_datamodule = test_datamodule_creation()
    
    # Test training dataloader
    try:
        train_loader = train_datamodule.train_dataloader()
        train_batch = next(iter(train_loader))
        print(f"✓ Training dataloader works - batch shape: {train_batch['image'].shape}")
    except Exception as e:
        print(f"✗ Training dataloader failed: {e}")
        return False
    
    # Test validation dataloader
    try:
        val_loader = train_datamodule.val_dataloader()
        val_batch = next(iter(val_loader))
        print(f"✓ Validation dataloader works - batch shape: {val_batch['image'].shape}")
    except Exception as e:
        print(f"✗ Validation dataloader failed: {e}")
        return False
    
    # Test test dataloader
    try:
        test_loader = test_datamodule.test_dataloader()
        test_batch = next(iter(test_loader))
        print(f"✓ Test dataloader works - batch shape: {test_batch['image'].shape}")
    except Exception as e:
        print(f"✗ Test dataloader failed: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    print("=== Testing DataModule Refactor ===\n")
    
    try:
        # Test dataset creation
        test_dataset_creation()
        
        # Test DataModule creation
        test_datamodule_creation()
        
        # Test dataloaders
        success = test_dataloaders()
        
        if success:
            print("\n=== All tests passed! ===")
            print("✓ Dataset classes work correctly")
            print("✓ Generic DataModule works correctly")
            print("✓ Dataloaders work correctly")
            print("\nThe refactor was successful!")
        else:
            print("\n=== Some tests failed! ===")
            return 1
            
    except Exception as e:
        print(f"\n=== Test failed with error: {e} ===")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
