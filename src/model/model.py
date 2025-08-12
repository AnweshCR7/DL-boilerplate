"""PyTorch Lightning model definitions."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torchmetrics


class SegFormerLightningModule(pl.LightningModule):
    """PyTorch Lightning module for SegFormer segmentation model."""
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_classes: int = 3,  # Fixed: should be 3 for Oxford Pet dataset
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        loss_fn: nn.Module | None = None,
        metrics: list[str] | None = None,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        inference: dict[str, any] | None = None,
        **kwargs,
    ):
        """Initialize the SegFormer Lightning module.
        
        Args:
            model_name: HuggingFace model name or path
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            loss_fn: Loss function (defaults to CrossEntropyLoss)
            metrics: List of metrics to track
            pretrained: Whether to use pretrained weights
            freeze_encoder: Whether to freeze encoder weights
            dropout: Dropout rate
            inference: Inference configuration
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        self.inference_config = inference or {}
        
        # Initialize model
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(
                num_labels=num_classes,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
            self.model = SegformerForSemanticSegmentation(config)
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.model.segformer.encoder.parameters():
                param.requires_grad = False
        
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
                metrics_dict[f"{stage}_dice"] = torchmetrics.F1Score(
                    task="multiclass", num_classes=self.num_classes, ignore_index=255, average="macro"
                )
        
        return metrics_dict
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        outputs = self.model(x)
        return outputs.logits
    
    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
        """Shared step for train/val/test."""
        images = batch["image"]
        targets = batch.get("mask", None)
        
        # Forward pass
        logits = self.forward(images)
        
        # Resize logits to match target size if needed
        if targets is not None and logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        
        loss = None
        if targets is not None:
            # Targets should be (N, H, W) and logits should be (N, C, H, W)
            
            # If targets have extra dimensions, squeeze them
            while len(targets.shape) > 3:
                targets = targets.squeeze(-1)
            
            # If targets are 4D with channel dim of 1, remove it
            if len(targets.shape) == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            
            # Ensure targets are long type
            targets = targets.long()
            
            # Verify shapes before loss computation
            if len(logits.shape) != 4:
                raise ValueError(f"Logits should be 4D (N, C, H, W), got {logits.shape}")
            if len(targets.shape) != 3:
                raise ValueError(f"Targets should be 3D (N, H, W), got {targets.shape}")
            
            loss = self.loss_fn(logits, targets)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
            "targets": targets,
        }
    
    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, "train")
        loss = outputs["loss"]
        
        # Log loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log metrics
        if outputs["targets"] is not None:
            for metric_name, metric in self.train_metrics.items():
                metric_value = metric(outputs["preds"], outputs["targets"])
                self.log(metric_name, metric_value, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        outputs = self._shared_step(batch, "val")
        loss = outputs["loss"]
        
        # Log loss
        if loss is not None:
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics
        if outputs["targets"] is not None:
            for metric_name, metric in self.val_metrics.items():
                metric_value = metric(outputs["preds"], outputs["targets"])
                self.log(metric_name, metric_value, on_step=False, on_epoch=True)
    
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        outputs = self._shared_step(batch, "test")
        loss = outputs["loss"]
        
        # Log loss
        if loss is not None:
            self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        # Log metrics
        if outputs["targets"] is not None:
            for metric_name, metric in self.test_metrics.items():
                metric_value = metric(outputs["preds"], outputs["targets"])
                self.log(metric_name, metric_value, on_step=False, on_epoch=True)
    
    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Prediction step."""
        outputs = self._shared_step(batch, "predict")
        
        # Apply softmax if configured
        if self.inference_config.get("apply_softmax", True):
            probs = F.softmax(outputs["logits"], dim=1)
        else:
            probs = outputs["logits"]
        
        return {
            "predictions": outputs["preds"],
            "probabilities": probs,
            "image_path": batch.get("image_path", None),
        }
    
    def configure_optimizers(self) -> dict[str, any]:
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
