# DataModule Refactor Documentation

## Overview

The datamodule has been refactored to use a common, generic Lightning DataModule wrapper instead of specific implementations for each dataset. This approach provides better reusability, consistency, and maintainability.

## Key Changes

### 1. Generic DataModule (`src/datasets/datamodule.py`)

The `DataModule` class is now a generic wrapper that can work with any dataset class. It provides:

- **Dynamic dataset instantiation** using the `utils.get_instance()` function
- **Mode-based setup** (train/test) with automatic train/val split for training mode
- **Flexible parameter configuration** for both datasets and dataloaders
- **Automatic collate function detection** from dataset classes
- **Better error handling** and logging
- **Reproducible worker initialization** for multi-processing

### 2. Dataset Classes (`src/datasets/data.py`)

Instead of specific DataModule implementations, we now have dataset classes that:

- **Accept a `mode` parameter** ('train', 'val', 'test')
- **Handle their own setup** (downloads, transforms, splits)
- **Provide custom collate functions** when needed
- **Are reusable** with the generic DataModule

### 3. Configuration-Driven Approach

Dataset configurations are stored in separate YAML files (`configs/data/`) that define:

- Dataset class and parameters
- Dataloader settings
- Mode-specific overrides

This modular approach allows easy switching between different datasets.

## Usage Examples

### Basic Usage

```python
from src.datasets.datamodule import DataModule

# Training mode (creates train and val datasets)
datamodule = DataModule(
    mode="train",
    dataset="src.datasets.data.OxfordPetDataset",
    dataset_params={
        "data_dir": "data/",
        "image_size": (256, 256),
        "download": True
    },
    dataloader_params={
        "batch_size": 8,
        "num_workers": 4
    }
)

# Test mode (creates only test dataset)
test_datamodule = DataModule(
    mode="test",
    dataset="src.datasets.data.OxfordPetDataset",
    dataset_params={"data_dir": "data/"},
    dataloader_params={"batch_size": 8}
)
```

### Configuration-Based Usage

```python
import yaml
from src.datasets.datamodule import DataModule

# Load specific dataset configuration
with open("configs/data/oxford_pet.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Create DataModule from config
datamodule = DataModule(
    mode="train",
    dataset=config["data"]["dataset"],
    dataset_params=config["data"]["dataset_params"],
    dataloader_params=config["data"]["dataloader_params"],
    train_dataloader_params=config["data"]["train_dataloader_params"],
    val_dataloader_params=config["data"]["val_dataloader_params"],
)
```

### Switching Between Datasets

```bash
# Use Oxford Pet dataset
python src/train.py data=oxford_pet

# Use Fashion MNIST dataset  
python src/train.py data=fashion_mnist
```

## Creating New Datasets

To add a new dataset, follow these steps:

### 1. Create Dataset Class

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, mode: str, **kwargs):
        self.mode = mode
        # Setup dataset based on mode
        self._setup_dataset()
        self._setup_transforms()
    
    def _setup_dataset(self):
        # Initialize dataset based on mode
        if self.mode in ["train", "val"]:
            # Setup train/val split
            pass
        elif self.mode == "test":
            # Setup test dataset
            pass
    
    def _setup_transforms(self):
        # Setup transforms based on mode
        if self.mode == "train":
            # Training transforms with augmentations
            pass
        else:
            # Validation/test transforms
            pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return data sample
        return {"image": image, "label": label}
    
    def collate_fn(self, batch):
        # Custom collate function if needed
        return {"images": images, "labels": labels}
```

### 2. Create Dataset Configuration File

```yaml
# configs/data/my_dataset.yaml
data:
  _target_: src.datasets.datamodule.DataModule
  mode: "train"
  dataset: "src.datasets.data.MyDataset"
  dataset_params:
    data_dir: "data/my_dataset/"
    image_size: [224, 224]
    # Other dataset-specific parameters
  dataloader_params:
    batch_size: 16
    num_workers: 4
    pin_memory: true
  train_dataloader_params:
    shuffle: true
    drop_last: true
  val_dataloader_params:
    shuffle: false
    drop_last: false
  test_dataloader_params:
    shuffle: false
    drop_last: false
```

### 3. Use with Generic DataModule

```python
# Direct usage
datamodule = DataModule(
    mode="train",
    dataset="src.datasets.data.MyDataset",
    dataset_params={"data_dir": "data/my_dataset/"},
    dataloader_params={"batch_size": 16}
)

# Or via Hydra configuration
python src/train.py data=my_dataset
```

## Benefits

1. **Reusability**: Same DataModule works for any dataset
2. **Consistency**: All datasets follow the same interface
3. **Flexibility**: Easy to add new datasets without modifying the DataModule
4. **Configuration-driven**: Dataset parameters managed through config files
5. **Testability**: Easier to test individual components
6. **Maintainability**: Less code duplication and clearer separation of concerns

## Migration Guide

If you have existing specific DataModule implementations:

1. Extract dataset logic into a separate dataset class
2. Add `mode` parameter to the dataset constructor
3. Move dataloader-specific logic to the generic DataModule
4. Create configuration files for your datasets
5. Update training scripts to use the generic approach

## Example Script

Run the example script to test the new approach:

```bash
python examples/use_generic_datamodule.py
```

This will demonstrate the complete workflow with the Oxford Pet dataset.
