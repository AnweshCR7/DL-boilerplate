"""Configuration for Project."""
from pathlib import Path
import time

# for local
CHECKPOINT_PATH: str = f"/Users/anwesh.marwade@pon.com/anweshcr7_git/focus-distance-prediction/checkpoint_{time.strftime('%Y%m%d-%H%M%S')}"
TENSORBOARD_DIR: str = f"/Users/anwesh.marwade@pon.com/anweshcr7_git/focus-distance-prediction/results/{time.strftime('%Y%m%d-%H%M%S')}"
DATASET_PATH: Path = Path(
    "/Users/anwesh.marwade@pon.com/anweshcr7_git/focus-distance-prediction/data/split_data"
)
SAVED_CHECKPOINT_PATH: Path = Path(
    "/Users/anwesh.marwade@pon.com/anweshcr7_git/focus-distance-prediction/results/20230726-234616_bs_12_1024_l1loss/model_final.pt"
)
PLOT_PATH: Path = SAVED_CHECKPOINT_PATH.parent


NUM_WORKERS: int = 2
DEVICE: str = "gpu"
BATCH_SIZE: int = 8
EPOCHS: int = 150
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
GAMMA: float = 0.1
IMAGE_WIDTH: int = 512
# TODO: maintain aspect ratio
IMAGE_HEIGHT: int = 512
