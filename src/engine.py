"""Engine functions for training/evaluation and inference."""
import argparse
import os
import time
import datetime
from io import BytesIO
from typing import Any, List, Optional, Dict, Tuple
import PIL
from loguru import logger
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# import tensorboardX
from PIL import Image
from tensorboardX import SummaryWriter
from torch import tensor
from tqdm import tqdm
import torch
import config
from src.utils.metric_logger import MetricLogger
from src.utils.plot import scatter_plot_inference, histogram_plot
from utils.model_utils import save_model_checkpoint
from torchvision.transforms import ToTensor


def train_fn(
    model: Any,
    train_data_loader: Any,
    val_data_loader: Any,
    optimizer: Any,
    scheduler: Any,
    criterion: Any,
    args: argparse.Namespace,
) -> float:
    """Training Engine
    Args:
        model: model supplied for training
        train_data_loader: dataloader containing train images and targets
        val_data_loader: dataloader containing validation images and targets
        optimizer: optimizer supplied for training
        scheduler: learning rate scheduler
        criterion: for calculating the loss
        args: args for training

    Returns:
        Computed Loss
    """
    meters: MetricLogger = MetricLogger()
    start_training_time: time = time.time()
    if args.use_tensorboard:
        logger.info("Using Tensorboard for logging!")

        if not os.path.exists(config.TENSORBOARD_DIR):
            os.makedirs(config.TENSORBOARD_DIR)

        # TODO: check if this is over-written
        summary_writer: Optional[SummaryWriter] = SummaryWriter(
            log_dir=os.path.join(config.TENSORBOARD_DIR, "tf_logs")
        )
    else:
        summary_writer: Optional[SummaryWriter] = None

    best_val_loss: float = float("inf")
    for epoch in range(args.epochs):
        running_loss: float = 0.0
        model.train()
        tk = tqdm(train_data_loader, total=len(train_data_loader))
        for data in tk:
            images: tensor = data["images"].to(args.device)
            targets: tensor = data["focus_distance"].to(args.device)
            optimizer.zero_grad()
            out: tensor = model(images)
            # Check loss
            loss: tensor = criterion(out, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # get loss calculation per batch
        running_loss = running_loss / len(train_data_loader)

        if scheduler:
            scheduler.step(running_loss)
        batch_time: time = time.time() - start_training_time
        meters.update(total_loss=running_loss, time=batch_time)
        if args.verbose:
            eta_seconds = meters.time.global_avg * (args.epochs - epoch)
            eta_string: str = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join(
                    [
                        f"epo: {epoch}",
                        f"lr: {optimizer.param_groups[0]['lr']}",
                        f"ImShape: {images.size(2)}, {images.size(3)}",
                        f"{str(meters)}",
                        f"eta: {eta_string}",
                        f"mem: {round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)}M",
                    ]
                )
            )

        if epoch % args.save_step == 0:
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                loss=running_loss,
                checkpoint_path=config.CHECKPOINT_PATH,
            )

        # validation
        if epoch % args.eval_step == 0:
            logger.info(f"Executing validation step at epoch: {epoch}")
            eval_loss, eval_results = eval_fn(
                model=model, data_loader=val_data_loader, criterion=criterion, args=args
            )
            logger.info(f"Validation loss: {eval_loss}")
            if summary_writer:
                global_step = epoch
                summary_writer.add_scalar(
                    "losses/validation_loss", best_val_loss, global_step=global_step
                )
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                save_model_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    loss=best_val_loss,
                    checkpoint_path=config.CHECKPOINT_PATH,
                    model_filename="model_best.pt",
                )
                # PLOTS FOR VALIDATION
                logger.info("Writing validation plots to tensorboard")
                plot_buf: BytesIO = scatter_plot_inference(
                    data=eval_results, save_path=config.PLOT_PATH, for_tensorboard=True
                )
                scatter_plot_tensor: tensor = ToTensor()(PIL.Image.open(plot_buf))
                summary_writer.add_image(
                    tag="Prediction Scatter Plot",
                    img_tensor=scatter_plot_tensor,
                    global_step=epoch,
                )

                plot_buf: BytesIO = histogram_plot(
                    data=eval_results, save_path=config.PLOT_PATH, for_tensorboard=True
                )
                hist_plot_tensor: tensor = ToTensor()(PIL.Image.open(plot_buf))
                summary_writer.add_image(
                    tag="Error Histogram Plot",
                    img_tensor=hist_plot_tensor,
                    global_step=epoch,
                )

            model.train()

        if summary_writer:
            global_step: int = epoch
            summary_writer.add_scalar(
                "losses/total_loss", running_loss, global_step=global_step
            )
            summary_writer.add_scalar(
                "lr", optimizer.param_groups[0]["lr"], global_step=global_step
            )

    save_model_checkpoint(
        model=model,
        optimizer=optimizer,
        loss=running_loss,
        checkpoint_path=config.CHECKPOINT_PATH,
        model_filename="model_final.pt",
    )

    total_training_time: int = int(time.time() - start_training_time)
    total_time_str: str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        f"Total training time: {total_time_str} ({round(total_training_time / args.epochs, 2)} s / it)"
    )

    return running_loss


def eval_fn(
    model: Any, data_loader: Any, criterion: Any, args: argparse.Namespace
) -> Tuple[float, Dict[str, List[float]]]:
    """Evaluation Engine.
    Args:
        model: model supplied for evaluation
        data_loader: dataloader containing validation images and targets
        criterion: criterion for calculating the loss
        args: args for evaluation

    Returns:
        float loss and list of predictions and ground truths
    """
    model.eval()
    running_val_loss: float = 0.0
    focus_predictions: List[float] = []
    focus_gt: List[float] = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            images: tensor = data["images"].to(args.device)
            targets: tensor = data["focus_distance"].to(args.device)
            out: tensor = model(images)
            loss: tensor = criterion(out, targets.unsqueeze(1))
            running_val_loss += loss.item()
            focus_gt.extend(data["focus_distance"].flatten().tolist())
            focus_predictions.extend(out.flatten().tolist())
        return running_val_loss / len(data_loader), {
            "predictions": focus_predictions,
            "gt": focus_gt,
        }


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)


def inference_fn(model: Any, data_loader: Any, args) -> Dict[str, List[float]]:
    """Inference Engine.
    Args:
        model: model supplied for inference
        data_loader: dataloader containing test images and targets
        args: args for inference

    Returns:
        list of predictions and ground truths
    """
    model.eval()

    from pytorch_grad_cam import DeepFeatureFactorization
    from pytorch_grad_cam.utils.image import show_factorization_on_image
    from torchvision.models import resnet34
    resnet = resnet34()
    resnet.eval()
    feature_model = ResnetFeatureExtractor(model=resnet)

    focus_predictions: List[float] = []
    focus_gt: List[float] = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for key, value in data.items():
                data[key] = value.to(args.device)

            rgb_img_float = data["images"][0].numpy()
            out: tensor = model(data["images"])
            # dff = DeepFeatureFactorization(model=model, target_layer=resnet.layer4[-1])
            # concepts, batch_explanations, concept_scores = dff(data["images"])
            # visualization = show_factorization_on_image(rgb_img_float,
            #                                             batch_explanations[0],
            #                                             image_weight=0.3)
            targets = [SimilarityToConceptTarget(out[0,:])]
            # Where is the car in the image
            with GradCAM(model=model,
                         target_layers=[resnet.layer4[-1]],
                         use_cuda=False) as cam:
                car_grayscale_cam = cam(input_tensor=data["images"],
                                        targets=targets)[0, :]
            car_cam_image = show_cam_on_image(rgb_img_float, car_grayscale_cam, use_rgb=True)
            Image.fromarray(car_cam_image)

            focus_gt.extend(data["focus_distance"].flatten().tolist())
            focus_predictions.extend(out.flatten().tolist())

        return {"predictions": focus_predictions, "gt": focus_gt}
