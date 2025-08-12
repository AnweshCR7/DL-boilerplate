#!/usr/bin/env python3
"""
Example script demonstrating how to switch between different datasets.
"""

import yaml
import hydra
from omegaconf import DictConfig
from src.datasets.datamodule import DataModule


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_dataset(dataset_name: str):
    """Test a specific dataset configuration."""
    print(f"\n{'='*50}")
    print(f"Testing {dataset_name.upper()} dataset")
    print(f"{'='*50}")
    
    # Load the specific dataset config
    config = load_config(f"configs/data/{dataset_name}.yaml")
    
    try:
        # Create DataModule
        datamodule = DataModule(
            mode="train",
            dataset=config["data"]["dataset"],
            dataset_params=config["data"]["dataset_params"],
            dataloader_params=config["data"]["dataloader_params"],
            train_dataloader_params=config["data"]["train_dataloader_params"],
            val_dataloader_params=config["data"]["val_dataloader_params"],
        )
        
        print(f"✓ {dataset_name} DataModule created successfully")
        print(f"  - Training samples: {len(datamodule.data_train)}")
        print(f"  - Validation samples: {len(datamodule.data_val)}")
        
        # Test a sample
        sample = datamodule.data_train[0]
        print(f"  - Sample keys: {list(sample.keys())}")
        print(f"  - Sample shapes: {[(k, v.shape) for k, v in sample.items()]}")
        
        # Test dataloader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        print(f"  - Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
        
        return True
        
    except Exception as e:
        print(f"✗ {dataset_name} failed: {e}")
        return False


def main():
    """Main function demonstrating dataset switching."""
    
    print("=== Dataset Switching Demo ===")
    print("This demonstrates how to switch between different datasets using modular configs.")
    
    # Test different datasets
    datasets = ["oxford_pet", "fashion_mnist"]
    results = {}
    
    for dataset in datasets:
        results[dataset] = test_dataset(dataset)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    for dataset, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{dataset:15} {status}")
    
    # Instructions for switching
    print(f"\n{'='*50}")
    print("HOW TO SWITCH DATASETS")
    print(f"{'='*50}")
    print("1. Edit configs/config.yaml")
    print("2. Change the 'data' line in defaults:")
    print("   - data: oxford_pet    # for Oxford Pet")
    print("   - data: fashion_mnist # for Fashion MNIST")
    print("3. Run your training script")
    print("\nExample:")
    print("   python src/train.py data=fashion_mnist")
    print("   python src/train.py data=oxford_pet")


if __name__ == "__main__":
    main()
