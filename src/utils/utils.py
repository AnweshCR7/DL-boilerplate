"""
This module contains utility functions.
"""

import importlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

import numpy as np
import torch
from ruamel.yaml import YAML

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


def get_logger(name: str) -> logging.Logger:
    """Get logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


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


def save_config(config: Dict[str, Any], output_dir: str, filename: str = "config.yaml") -> None:
    """Save configuration to file."""
    from omegaconf import OmegaConf
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_path = output_path / filename
    OmegaConf.save(config, config_path)


def load_image(image_path: str) -> np.ndarray:
    """Load image from file."""
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to file."""
    if image.dtype != np.uint8:
        # Normalize to 0-255 range
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    Image.fromarray(image).save(output_path)


def visualize_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> None:
    """Visualize segmentation results."""
    fig, axes = plt.subplots(1, 3 if prediction is not None else 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Ground truth mask
    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=alpha, cmap="tab10")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    # Prediction (if provided)
    if prediction is not None:
        axes[2].imshow(image)
        axes[2].imshow(prediction, alpha=alpha, cmap="tab10")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_color_map(num_classes: int) -> np.ndarray:
    """Create a color map for visualization."""
    cmap = plt.cm.get_cmap("tab10")
    colors = []
    for i in range(num_classes):
        color = cmap(i / num_classes)
        colors.append([int(c * 255) for c in color[:3]])
    return np.array(colors)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.6,
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlay segmentation mask on image."""
    if colors is None:
        colors = create_color_map(mask.max() + 1)
    
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(len(colors)):
        colored_mask[mask == class_id] = colors[class_id]
    
    # Blend with original image
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return blended


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Compute segmentation metrics."""
    from sklearn.metrics import jaccard_score, accuracy_score
    
    # Flatten tensors
    pred_flat = predictions.flatten().cpu().numpy()
    target_flat = targets.flatten().cpu().numpy()
    
    # Remove ignore index if present
    valid_mask = target_flat != -1
    pred_flat = pred_flat[valid_mask]
    target_flat = target_flat[valid_mask]
    
    # Compute metrics
    accuracy = accuracy_score(target_flat, pred_flat)
    iou = jaccard_score(target_flat, pred_flat, average="macro", labels=range(num_classes), zero_division=0)
    
    return {
        "accuracy": accuracy,
        "iou": iou,
    }


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label="Train")
    axes[0, 0].plot(val_losses, label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves (if available)
    if "accuracy" in train_metrics and "accuracy" in val_metrics:
        axes[0, 1].plot(train_metrics["accuracy"], label="Train")
        axes[0, 1].plot(val_metrics["accuracy"], label="Validation")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # IoU curves (if available)
    if "iou" in train_metrics and "iou" in val_metrics:
        axes[1, 0].plot(train_metrics["iou"], label="Train")
        axes[1, 0].plot(val_metrics["iou"], label="Validation")
        axes[1, 0].set_title("IoU")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("IoU")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate (if available)
    if "lr" in train_metrics:
        axes[1, 1].plot(train_metrics["lr"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(targets, predictions)
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_inference_grid(
    images: List[np.ndarray],
    predictions: List[np.ndarray],
    targets: Optional[List[np.ndarray]] = None,
    max_images: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """Create a grid of inference results."""
    num_images = min(len(images), max_images)
    cols = 3 if targets is not None else 2
    rows = num_images
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis("off")
        
        # Prediction
        axes[i, 1].imshow(images[i])
        axes[i, 1].imshow(predictions[i], alpha=0.6, cmap="tab10")
        axes[i, 1].set_title(f"Prediction {i+1}")
        axes[i, 1].axis("off")
        
        # Ground truth (if available)
        if targets is not None:
            axes[i, 2].imshow(images[i])
            axes[i, 2].imshow(targets[i], alpha=0.6, cmap="tab10")
            axes[i, 2].set_title(f"Ground Truth {i+1}")
            axes[i, 2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_instance(class_path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Dynamically instantiate a class from a string path.
    
    Args:
        class_path: Full path to the class (e.g., 'src.datasets.data.OxfordPetDataset')
        params: Parameters to pass to the class constructor
        
    Returns:
        Instance of the specified class
    """
    if params is None:
        params = {}
    
    try:
        # Split the class path into module and class name
        module_path, class_name = class_path.rsplit('.', 1)
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class
        cls = getattr(module, class_name)
        
        # Instantiate the class with parameters
        return cls(**params)
        
    except (ImportError, AttributeError, ValueError) as e:
        raise RuntimeError(f"Failed to instantiate {class_path}: {str(e)}")
