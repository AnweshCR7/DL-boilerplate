"""Functions for splitting dataset into train/val/test."""
import json
import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import shutil
from loguru import logger
from pandas import DataFrame
from tqdm import tqdm


def train_test_split(
    df: DataFrame, val_frac: float = 0.20, random_state: int = 50
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    # Set two image_names aside for test
    test_df = df[-2:]

    # get everything but the test sample
    train_val_df = df.drop(index=test_df.index)

    val_df = train_val_df.sample(frac=val_frac, axis=0, random_state=random_state)
    train_df = train_val_df.drop(index=val_df.index)

    return train_df, val_df, test_df


def image_name_to_features(file_name: Path) -> Tuple[str, float, float]:
    absolute_focus_height, relative_focus_distance = file_name.stem.split("_")[-2:]
    image_name = "_".join(file_name.stem.split("_")[:-2])
    return image_name, absolute_focus_height, relative_focus_distance


def move_files_given_unique_names(
    image_names: List[str], source_dir: Path, dest_dir: Path
) -> None:
    if not os.path.exists(str(dest_dir)):
        os.makedirs(str(dest_dir))
    for image_name in image_names:
        images_to_copy = source_dir.rglob(pattern=f"{image_name}*")
        for file in tqdm(images_to_copy):
            shutil.copy(str(file), str(dest_dir / file.name))


def split_using_metadata(metadata: DataFrame, source_dir, dest_dir) -> None:
    counts_df = pd.DataFrame(metadata.image_name.value_counts())
    train_images, val_images, test_images = train_test_split(counts_df)
    logger.info(
        "Dataset breakup -> # Train images: {sum(train_images['count'])} | # Validation images: {sum("
        "val_images['count'])} | # Test images:{sum(test_images['count'])}"
    )
    train_images, val_images, test_images = (
        train_images.index.tolist(),
        val_images.index.tolist(),
        test_images.index.tolist(),
    )
    dataset_split_metadata = {
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
    }
    with open("../data/split_data/dataset_split_metadata.json", "w") as fp:
        json.dump(dataset_split_metadata, fp, indent=4)

    logger.info("Moving train images")
    move_files_given_unique_names(
        image_names=train_images,
        source_dir=source_dir,
        dest_dir=dest_dir / Path("train"),
    )
    logger.info("Moving validation images")
    move_files_given_unique_names(
        image_names=val_images, source_dir=source_dir, dest_dir=dest_dir / Path("val")
    )
    logger.info("Moving test images")
    move_files_given_unique_names(
        image_names=test_images, source_dir=source_dir, dest_dir=dest_dir / Path("test")
    )


if __name__ == "__main__":
    data = pd.read_csv("../data/focus_distance_prediction_metadata.csv")
    dataset_path: Path = Path(
        "/Users/anwesh.marwade@pon.com/anweshcr7_git/focus-distance-prediction/data/focus-distance-prediction"
        "-challenge-dataset"
    )
    split_using_metadata(
        metadata=data,
        source_dir=dataset_path,
        dest_dir=Path(
            "/Users/anwesh.marwade@pon.com/anweshcr7_git/focus-distance-prediction/data/split_data/"
        ),
    )

    # image_paths: List[Path] = [path for path in dataset_path.glob("*.png")]
    # data = []
    #
    # for path in image_paths:
    #     data.append(image_name_to_features(file_name=path))
    # columns = ["image_name", "absolute_focus_height", "relative_focus_distance"]
    # data = pd.DataFrame(data, columns=columns)
    # print('hello0')
