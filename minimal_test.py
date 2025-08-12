#!/usr/bin/env python3
"""
Minimal test script with no augmentations to get the training pipeline working.
"""

import torch
import pytorch_lightning as pl
from src.datasets.datamodule import DataModule
from src.model.simple_unet_lightning import SimpleUNetLightningModule
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


def minimal_test():
    """Minimal test with no augmentations."""
    print("=== Minimal Test with No Augmentations ===")
    
    # 1. Create DataModule with NO augmentations
    print("1. Creating DataModule (no augmentations)...")
    datamodule = DataModule(
        mode="train",
        dataset="src.datasets.data.OxfordPetDataset",
        dataset_params={
            "data_dir": "data/",
            "download": True,
            "image_size": [256, 256],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "augmentations": {}  # NO augmentations
        },
        dataloader_params={
            "batch_size": 8,  # Larger batch for efficiency
            "num_workers": 4,  # Enable multiprocessing
            "pin_memory": True,  # Enable pin memory for faster GPU transfer
            "persistent_workers": True,  # Keep workers alive between epochs
            "shuffle": True,  # Enable shuffle for better training
            "drop_last": True,
        },
        train_dataloader_params={"shuffle": True, "drop_last": True},
        val_dataloader_params={"shuffle": False, "drop_last": False},
    )
    
    print(f"Training dataset size: {len(datamodule.data_train)}")
    print(f"Validation dataset size: {len(datamodule.data_val)}")
    
    # 2. Create Model with minimal metrics
    print("\n2. Creating Simple U-Net model...")
    model = SimpleUNetLightningModule(
        in_channels=3,
        num_classes=3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        metrics=["accuracy"]  # Only one metric
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 3. Create Trainer with minimal settings
    print("\n3. Creating Trainer...")
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="mps",
        devices=1,
        precision=32,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_checkpointing=False,  # No checkpointing
        enable_progress_bar=True,
        enable_model_summary=False,  # No model summary
        callbacks=[TQDMProgressBar()],
        logger=False,  # No logging
    )
    
    # 4. Test a single batch first
    print("\n4. Testing single batch...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    
    images = batch["image"]
    masks = batch["mask"]
    
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Masks unique values: {torch.unique(masks).tolist()}")
    print(f"  Masks min/max: {masks.min().item()}/{masks.max().item()}")
    
    # Test forward pass
    with torch.no_grad():
        logits = model(images)
        print(f"  Logits shape: {logits.shape}")
    
    # Test loss
    loss = model.loss_fn(logits, masks)
    print(f"  Loss: {loss.item():.4f}")
    
    # 5. Run training for 1 epoch
    print("\n5. Running training for 1 epoch...")
    try:
        trainer.fit(model=model, datamodule=datamodule)
        print("✓ Training completed successfully!")
        
        # 6. Test the model using validation data
        print("\n6. Testing the model...")
        
        # Create a test dataloader using validation data
        test_loader = datamodule.val_dataloader()
        
        # Run test
        test_results = trainer.test(model=model, dataloaders=test_loader)
        print(f"Test results: {test_results}")
        
        print("\n=== SUCCESS! Minimal training pipeline is working ===")
        return True
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = minimal_test()
    if success:
        print("\n🎉 The minimal PyTorch Lightning pipeline is working!")
        print("You can now add back features one by one.")
    else:
        print("\n❌ The minimal test failed.")
