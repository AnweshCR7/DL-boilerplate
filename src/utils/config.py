"""
This module contains the typing for the configuration file.
"""

from pathlib import Path
from typing import Any, Dict, Literal, TypedDict


class LoggerParams(TypedDict):
    """
    This class represents the configuration of the logger.
    """

    project: str
    name: str
    version: str | None
    save_dir: str


class TrainerParams(TypedDict):
    """
    This class represents the configuration of the trainer.
    """

    deterministic: bool
    devices: list[int]
    accelerator: str
    num_sanity_val_steps: int
    max_epochs: int
    precision: Literal[32] | Literal[64]
    limit_train_batches: int | float
    limit_val_batches: int | float


class DataLoadersParams(TypedDict, total=False):
    """
    This class represents the configuration of the dataloaders.
    """

    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    sampler: str | None
    shuffle: bool


class DataModuleParams(TypedDict):
    """
    This class represents the configuration of the datamodule.
    """

    dataset: str
    dataset_params: dict
    dataloader_params: DataLoadersParams
    train_dataloader_params: DataLoadersParams
    val_dataloader_params: DataLoadersParams


class LightningModuleParams(TypedDict):
    """
    This class represents the configuration of the lightning module.
    """

    model: str
    model_params: dict
    loss: str
    loss_params: dict
    optimizer: str
    optimizer_params: dict
    scheduler: str
    scheduler_params: dict
    logging_function: str


class TrainConfig(TypedDict):
    """
    This class represents the configuration of the trainer.
    """

    mode: Literal["train"]
    experiment_name: str
    experiment_path: Path
    resume_path: str | None
    logger_params: LoggerParams
    trainer_params: TrainerParams
    trainer_callbacks: list[Dict[str, Any]]
    datamodule_params: DataModuleParams
    lightning_module: str
    lightning_module_params: LightningModuleParams
    wall: bool
    no_logging: bool
    seed: int
    environ_vars: dict[str, Any] | None
    loglevel: str


class TestConfig(TypedDict):
    """
    This class represents the configuration of the tester.
    """

    mode: Literal["test"]
    checkpoint_path: Path | None
    logger_params: LoggerParams
    trainer_params: TrainerParams
    trainer_callbacks: list[Dict[str, Any]]
    datamodule_params: DataModuleParams
    lightning_module: str
    lightning_module_params: LightningModuleParams
    wall: bool
    no_logging: bool
    seed: int
    environ_vars: dict[str, Any] | None
    loglevel: str


class CriterionConfig(TypedDict):
    """
    This class represents the configuration of a criterion.
    """

    criterion: str
    criterion_params: dict
