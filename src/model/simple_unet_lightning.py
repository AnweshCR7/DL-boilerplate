"""PyTorch Lightning module for Simple U-Net segmentation model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class SimpleUNetLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Simple U-Net segmentation model."""
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        learning_rate: float = 1e-3,  # Slightly higher LR for simple model
        weight_decay: float = 1e-4,
        loss_fn: nn.Module | None = None,
        metrics: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the Simple U-Net Lightning module.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            loss_fn: Loss function (defaults to CrossEntropyLoss)
            metrics: List of metrics to track
            features: List of feature dimensions for U-Net levels
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model
        from .simple_unet import SimpleUNet
        self.model = SimpleUNet(
            in_channels=in_channels,
            num_classes=num_classes
        )
        
        # Loss function
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        else:
            # If loss_fn is a DictConfig, instantiate it
            if hasattr(loss_fn, '_target_'):
                import hydra
                self.loss_fn = hydra.utils.instantiate(loss_fn)
            else:
                self.loss_fn = loss_fn
        
        # Metrics
        metrics = metrics or ["accuracy", "iou"]
        self.train_metrics = self._setup_metrics(metrics, "train")
        self.val_metrics = self._setup_metrics(metrics, "val")
        self.test_metrics = self._setup_metrics(metrics, "test")
    
    def _setup_metrics(self, metric_names: list[str], stage: str) -> nn.ModuleDict:
        """Setup metrics for a given stage."""
        metrics_dict = nn.ModuleDict()
        
        for metric_name in metric_names:
            if metric_name == "accuracy":
                metrics_dict[f"{stage}_accuracy"] = torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_classes, ignore_index=255
                )
            elif metric_name == "iou":
                metrics_dict[f"{stage}_iou"] = torchmetrics.JaccardIndex(
                    task="multiclass", num_classes=self.num_classes, ignore_index=255
                )
            elif metric_name == "dice":
                # Use F1 score as a proxy for Dice coefficient
                metrics_dict[f"{stage}_dice"] = torchmetrics.F1Score(
                    task="multiclass", num_classes=self.num_classes, ignore_index=255
                )
            elif metric_name == "precision":
                metrics_dict[f"{stage}_precision"] = torchmetrics.Precision(
                    task="multiclass", num_classes=self.num_classes, ignore_index=255
                )
            elif metric_name == "recall":
                metrics_dict[f"{stage}_recall"] = torchmetrics.Recall(
                    task="multiclass", num_classes=self.num_classes, ignore_index=255
                )
        
        return metrics_dict
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch["image"]
        masks = batch["mask"]
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.loss_fn(logits, masks)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        
        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log metrics
        for metric_name, metric_fn in self.train_metrics.items():
            metric_value = metric_fn(preds, masks)
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch["image"]
        masks = batch["mask"]
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.loss_fn(logits, masks)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        
        # Log loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics
        for metric_name, metric_fn in self.val_metrics.items():
            metric_value = metric_fn(preds, masks)
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        images = batch["image"]
        masks = batch["mask"]
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.loss_fn(logits, masks)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        
        # Log loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        # Log metrics
        for metric_name, metric_fn in self.test_metrics.items():
            metric_value = metric_fn(preds, masks)
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        images = batch["image"]
        logits = self(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return {
            "logits": logits,
            "probs": probs,
            "predictions": preds,
        }
