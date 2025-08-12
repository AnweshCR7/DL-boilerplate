#!/usr/bin/env python3
"""
Simple test script for the training pipeline without Hydra.
"""

import torch
import pytorch_lightning as pl
from src.datasets.datamodule import DataModule
from src.model.simple_unet_lightning import SimpleUNetLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


def test_training_pipeline():
    """Test the complete training pipeline with Simple U-Net."""
    print("=== Testing Training Pipeline with Simple U-Net ===")
    
    # 1. Create DataModule
    print("1. Creating DataModule...")
    datamodule = DataModule(
        mode="train",
        dataset="src.datasets.data.OxfordPetDataset",
        dataset_params={
            "data_dir": "data/",
            "download": True,
            "image_size": [256, 256],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "augmentations": {
                "horizontal_flip": 0.5,
                "rotation": 10,
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.05,
            }
        },
        dataloader_params={
            "batch_size": 4,  # Smaller batch for testing
            "num_workers": 0,  # No multiprocessing
            "pin_memory": False,
            "persistent_workers": False,
            "shuffle": True,
            "drop_last": True,
        },
        train_dataloader_params={"shuffle": True, "drop_last": True},
        val_dataloader_params={"shuffle": False, "drop_last": False},
    )
    
    print(f"Training dataset size: {len(datamodule.data_train)}")
    print(f"Validation dataset size: {len(datamodule.data_val)}")
    
    # 2. Create Model
    print("\n2. Creating Simple U-Net model...")
    model = SimpleUNetLightningModule(
        in_channels=3,
        num_classes=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        metrics=["accuracy", "iou"]  # Simplified metrics
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 3. Create Trainer
    print("\n3. Creating Trainer...")
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="mps",
        devices=1,
        precision=32,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        enable_checkpointing=False,  # Disable for testing
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[TQDMProgressBar()],
        logger=TensorBoardLogger("logs/", name="test_simple_unet"),
    )
    
    # 4. Test a single training step
    print("\n4. Testing single training step...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    # Disable logging for this test
    model.log = lambda *args, **kwargs: None
    
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
    
    # 5. Test a single validation step
    print("\n5. Testing single validation step...")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    
    val_loss = model.validation_step(val_batch, 0)
    print(f"Validation loss: {val_loss.item():.4f}")
    
    # 6. Run training for 1 epoch
    print("\n6. Running training for 1 epoch...")
    try:
        trainer.fit(model=model, datamodule=datamodule)
        print("✓ Training completed successfully!")
        
        # 7. Test the model
        print("\n7. Testing the model...")
        test_results = trainer.test(model=model, datamodule=datamodule)
        print(f"Test results: {test_results}")
        
        print("\n=== SUCCESS! Training pipeline is working correctly ===")
        return True
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_training_pipeline()
    if success:
        print("\n🎉 The PyTorch Lightning pipeline is working!")
        print("You can now switch back to SegFormer or other complex models.")
    else:
        print("\n❌ There are still issues to resolve.")
