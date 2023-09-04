"""Function for running training job."""
import argparse
from pathlib import Path
from typing import List, Any
from loguru import logger
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import config
from src.engine import train_fn, eval_fn
from src.infer import run_inference
from src.models.wide_resnet import WideResNet
from src.utils.dataset import DataLoaderFocusDistance
from models.resnet import ResNet


def run_training(args: argparse.Namespace) -> None:
    """Function to start a training run.
    Args:
        args: command line arguments provided for the training run
    """
    logger.info("Run Training")
    model = ResNet(pretrained=False)
    # model: Any = WideResNet(pretrained=False)

    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        logger.info("GPU available... using GPU 🖥️")
        torch.cuda.manual_seed_all(41)
    else:
        args.device = torch.device("cpu")
        logger.info("GPU not available... using CPU 💻")

    model.to(args.device)

    optimizer: Any = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    criterion: Any = nn.L1Loss()

    train_transforms: Any = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    logger.info("Creating Dataloaders 💽")
    train_image_paths: List[Path] = [
        path for path in (config.DATASET_PATH / Path("train")).glob("*.png")
    ][:2]

    train_dataset: DataLoaderFocusDistance = DataLoaderFocusDistance(
        image_paths=train_image_paths,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        transforms=train_transforms,
    )
    train_loader: DataLoader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    val_image_paths: List[Path] = [
        path for path in (config.DATASET_PATH / Path("val")).glob("*.png")
    ][:2]

    val_dataset: DataLoaderFocusDistance = DataLoaderFocusDistance(
        image_paths=val_image_paths,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        transforms=train_transforms,
    )
    val_loader: DataLoader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    logger.info("Starting training job... 🏋️")
    avg_train_loss = train_fn(
        model=model,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        args=args,
    )

    avg_val_loss = eval_fn(
        model=model, data_loader=val_loader, criterion=criterion, args=args
    )
    logger.info(
        f"Completed {args.epochs} epochs => Training Loss: {avg_train_loss}, Val Loss: {avg_val_loss}"
    )

    if not args.skip_test:
        run_inference(args=args)

    logger.info("Job complete! ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focus Distance Prediction - PyTorch")
    parser.add_argument(
        "--log-step", default=10, type=int, help="Print logs every log_step"
    )
    parser.add_argument(
        "--save-step", default=10, type=int, help="Save checkpoint every save_step"
    )
    parser.add_argument(
        "--eval-step",
        default=10,
        type=int,
        help="Evaluate dataset every eval_step, disabled when eval_step < 0",
    )
    parser.add_argument(
        "--load-pretrained",
        action="store_true",
        help="if true then the model loads pretrained weights either from s3 (--use-s3-weights) or locally",
    )
    parser.add_argument("--use_tensorboard", default=True, action="store_true")
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for processes. Seed must be fixed for distributed training",
    )
    parser.add_argument(
        "--batch-size",
        default=config.BATCH_SIZE,
        type=int,
        help="Size of a batch of data",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--epochs",
        default=config.EPOCHS,
        type=str,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        default=config.LR,
        type=float,
        help="Learning Rate",
    )
    # Always on
    parser.add_argument(
        "--verbose",
        default="True",
        action="store_true",
        help="Provides more logging",
    )
    args: argparse.Namespace = parser.parse_args()
    run_training(args=args)
