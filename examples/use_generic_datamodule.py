#!/usr/bin/env python3
"""
Example script demonstrating how to use the generic DataModule with Oxford Pet dataset.
"""

import yaml
from src.datasets.datamodule import DataModule
from utils import visualize_segmentation


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Main function demonstrating the generic DataModule usage."""
    
    # Load configuration
    config = load_config("../configs/data/oxford_pet.yaml")
    
    print("=== Dataset with Generic DataModule ===")
    print(f"Using dataset: {config['data']['dataset']}")
    
    # Create training DataModule
    print("\n1. Creating training DataModule...")
    
    # Modify dataloader params to avoid multiprocessing issues
    dataloader_params = config["data"]["dataloader_params"].copy()
    dataloader_params["num_workers"] = 0  # Disable multiprocessing
    dataloader_params["pin_memory"] = False  # Disable pin_memory for MPS compatibility
    dataloader_params["persistent_workers"] = False  # Disable persistent workers when num_workers=0
    
    train_datamodule = DataModule(
        mode="train",
        dataset=config["data"]["dataset"],
        dataset_params=config["data"]["dataset_params"],
        dataloader_params=dataloader_params,
        train_dataloader_params=config["data"]["train_dataloader_params"],
        val_dataloader_params=config["data"]["val_dataloader_params"],
    )
    
    print(f"Training dataset size: {len(train_datamodule.data_train)}")
    print(f"Validation dataset size: {len(train_datamodule.data_val)}")
    
    # Get a sample from training dataset
    sample = train_datamodule.data_train[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    # visualize_segmentation(sample["image"].numpy(), sample["mask"].numpy())
    
    # Create test DataModule
    print("\n2. Creating test DataModule...")
    test_datamodule = DataModule(
        mode="test",
        dataset=config["data"]["dataset"],
        dataset_params=config["data"]["dataset_params"],
        dataloader_params=dataloader_params,
        test_dataloader_params=config["data"].get("test_dataloader_params", {}),
    )
    
    print(f"Test dataset size: {len(test_datamodule.data_test)}")
    
    # Test dataloaders
    print("\n3. Testing dataloaders...")
    
    # Training dataloader
    train_loader = train_datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    print(f"Training batch - Images: {train_batch['image'].shape}, Masks: {train_batch['mask'].shape}")
    
    # Validation dataloader
    val_loader = train_datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    print(f"Validation batch - Images: {val_batch['image'].shape}, Masks: {val_batch['mask'].shape}")
    
    # Test dataloader
    test_loader = test_datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    print(f"Test batch - Images: {test_batch['image'].shape}, Masks: {test_batch['mask'].shape}")
    
    print("\n=== Success! Generic DataModule is working correctly ===")


if __name__ == "__main__":
    main()
