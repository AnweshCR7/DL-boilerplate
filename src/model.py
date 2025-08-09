"""PyTorch Lightning model definitions."""
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
import torchmetrics


class SegFormerLightningModule(pl.LightningModule):
    """PyTorch Lightning module for SegFormer segmentation model."""
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        loss_fn: Optional[nn.Module] = None,
        metrics: List[str] = None,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        inference: Optional[Dict[str, Any]] = None,
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
            from transformers import SegformerConfig
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
    
    def _setup_metrics(self, metric_names: List[str], stage: str) -> nn.ModuleDict:
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
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
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
            loss = self.loss_fn(logits, targets)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
            "targets": targets,
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
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
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
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
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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
    
    def configure_optimizers(self) -> Dict[str, Any]:
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


class SimpleCNN(pl.LightningModule):
    """Simple CNN model for basic tasks (legacy support)."""
    
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        """Initialize simple CNN."""
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Simple CNN architecture for MNIST (grayscale)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        
        logits = self.forward(images)
        loss = self.loss_fn(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, targets)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        images, targets = batch
        
        logits = self.forward(images)
        loss = self.loss_fn(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, targets)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
