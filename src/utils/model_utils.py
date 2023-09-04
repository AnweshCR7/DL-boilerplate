"""Model save/load utilities."""
import os
from pathlib import Path
from typing import Any, Tuple

import torch
from loguru import logger


def save_model_checkpoint(
    model: Any,
    loss: float,
    optimizer: Any = None,
    scheduler: Any = None,
    checkpoint_path: str = ".",
    model_filename: str = "running_model.pt",
) -> None:
    """Function to save model checkpoint.
    Args:
        model: model to be saved
        loss: loss computed for the model
        optimizer: optimizer to be saved
        scheduler: scheduler to be saved
        checkpoint_path: path for saving checkpoint
        model_filename: name of the checkpoint file to be saved
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    try:
        torch.save(
            {
                # 'epoch': epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(checkpoint_path, model_filename),
        )
        logger.info(f"{model_filename} saved successfully")
    except:
        logger.error(f"There was an error while saving {model_filename}")


def load_model(
    model: Any,
    checkpoint_path: Path,
    optimizer: Any = None,
    device: torch.device = "cpu",
) -> Tuple[Any, Any, float]:
    """Function to load model from saved checkpoint.
    Args:
        model: model object to which weights will be loaded
        checkpoint_path: path for saving checkpoint
        optimizer: optimizer object
        device: device to load the model onto (cpu, gpu)

    Returns:
        Model, Optimizer and the loss
    """
    loss = 0.0
    checkpoint_path = str(checkpoint_path)
    # check if checkpoint exists
    if os.path.exists(str(checkpoint_path)):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # epoch = checkpoint['epoch']
        loss = checkpoint["loss"]

    return model, optimizer, loss
