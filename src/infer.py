"""Function for running inference job."""
import argparse
from pathlib import Path
from typing import List
from loguru import logger
import torch
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import transforms
import torch.nn as nn
import config
from src.engine import inference_fn
from src.models.wide_resnet import WideResNet
from src.utils.dataset import DataLoaderFocusDistance
from models.resnet import ResNet
from src.utils.model_utils import load_model
from src.utils.plot import scatter_plot_inference, histogram_plot


def run_inference(args):
    logger.info("Run Test")

    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        logger.info("GPU available... using GPU 🖥️")
        torch.cuda.manual_seed_all(42)
    else:
        args.device = torch.device("cpu")
        logger.info("GPU not available... using CPU 💻")

    model = ResNet(pretrained=False)
    model, _, _ = load_model(
        model=model, checkpoint_path=config.SAVED_CHECKPOINT_PATH, device=args.device
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    logger.info("Creating Dataloaders 💽")
    test_image_paths: List[Path] = [
        path for path in (config.DATASET_PATH / Path("test")).glob("*.png")
    ]

    test_dataset = DataLoaderFocusDistance(
        image_paths=test_image_paths,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        transforms=test_transforms,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    logger.info("Starting training job... 🏋️")
    results = inference_fn(
        model=model,
        data_loader=test_loader,
        args=args,
    )

    scatter_plot_inference(data=results, save_path=config.PLOT_PATH)
    histogram_plot(data=results, save_path=config.PLOT_PATH)
    logger.info(
        f"Inference for <relative-focus-distance> => RMSE: {sqrt(mean_squared_error(results['gt'], results['predictions']))}, MAE: {mean_absolute_error(results['gt'], results['predictions'])}"
    )

    logger.info("Job complete! ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focus Distance Prediction - PyTorch")
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
    # Always on
    parser.add_argument(
        "--verbose",
        default="True",
        action="store_true",
        help="Provides more logging",
    )
    args = parser.parse_args()
    run_inference(args=args)
