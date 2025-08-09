"""Main training script using PyTorch Lightning and Hydra."""
from __future__ import annotations

import os
import random
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.data import OxfordPetDataModule
from src.model import SegFormerLightningModule


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def instantiate_callbacks(cfg: DictConfig) -> list:
    """Instantiate callbacks from config."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.paths.output_dir) / "checkpoints",
        filename="{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # TQDM progress bar
    progress_bar = TQDMProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks


def instantiate_logger(cfg: DictConfig) -> list:
    """Instantiate loggers from config."""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg.paths.log_dir,
        name=cfg.experiment_name,
        version=None,
        default_hp_metric=False,
    )
    loggers.append(tb_logger)
    
    # WandB logger (optional)
    if cfg.get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=cfg.get("wandb_project", "dl-boilerplate"),
            name=cfg.experiment_name,
            tags=cfg.get("tags", []),
        )
        loggers.append(wandb_logger)
    
    return loggers


def train(cfg: DictConfig) -> dict[str, any]:
    """Training pipeline."""
    set_seed(cfg.seed)
    
    create_output_dir(cfg.paths.output_dir)
    
    logger.info(f"Starting training with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    
    # Initialize model
    logger.info("Initializing model...")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    
    # Initialize callbacks
    callbacks = instantiate_callbacks(cfg)
    
    # Initialize loggers
    loggers = instantiate_logger(cfg)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
    )
    
    # Log hyperparameters
    if loggers:
        for pl_logger in loggers:
            pl_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    # Train the model
    logger.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    
    # Test the model
    if cfg.get("test_after_training", True):
        logger.info("Starting testing...")
        test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    else:
        test_results = None
    
    # Return results
    return {
        "best_model_path": trainer.checkpoint_callback.best_model_path,
        "best_score": trainer.checkpoint_callback.best_model_score,
        "test_results": test_results,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function."""
    try:
        # Run training
        results = train(cfg)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model path: {results['best_model_path']}")
        logger.info(f"Best validation score: {results['best_score']}")
        
        if results["test_results"]:
            logger.info(f"Test results: {results['test_results']}")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
