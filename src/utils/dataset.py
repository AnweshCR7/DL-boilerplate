from pathlib import Path
from typing import List, Tuple, Any, Dict
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataLoaderFocusDistance(Dataset):
    """Dataloader class for focus distance prediction."""

    def __init__(
        self,
        image_paths: List[Path],
        targets: List[float] = None,
        resize=None,
        transforms=None,
    ) -> None:
        """Init function
        Args:
            image_paths: list of image paths to be loaded in the dataloader
            targets: ground truth targets
            resize: (H, W) dimensions for resizing input images
            transforms: transforms to be applied to the input images
        """
        self.image_paths: List[Path] = image_paths
        self.transforms = transforms
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single item from the dataloader
        Args:
            index: index of the item to be return

        Returns:
            Dict of input images and targets
        """
        # Image has 4 channels -> converting to RGB
        image: Image = Image.open(str(self.image_paths[index]))
        targets: float = self.image_name_to_features(file_name=self.image_paths[index])
        if self.resize is not None:
            # write as HxW
            image: np.ndarray = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        # convert to numpy array
        image: np.ndarray = np.array(image)

        if self.transforms is not None:
            image: torch.tensor = self.transforms(image)
        else:
            # Convert to form: CxHxW where C = 1 for greyscale image (IMPLICIT IF USING TRANSFORM)
            image: torch.tensor = torch.tensor(
                np.expand_dims(image, 0).astype(np.float32)
            )

        return {
            "images": image,
            "focus_distance": torch.tensor(targets, dtype=torch.float),
        }

    @staticmethod
    def image_name_to_features(file_name: Path) -> float:
        """Function to obtain the image metadata from the image name.
        Args:
            file_name: Name of the file containing image metadata

        Returns:
            relative focus distance
        """
        absolute_focus_height, relative_focus_distance = file_name.stem.split("_")[-2:]
        image_name = "_".join(file_name.stem.split("_")[:-2])
        # return image_name, absolute_focus_height, relative_focus_distance
        return float(relative_focus_distance)
