"""
DataModule class for PyTorch Lightning
"""

from typing import Optional, Dict, Any
import logging

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..utils import utils
from ..utils.config import DataLoadersParams


logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes, too-many-arguments
class DataModule(LightningDataModule):
    """
    Generic DataModule class for PyTorch Lightning

    Args:
        mode (str): Mode of the DataModule (train or test)
        dataset (str): Dataset class
        dataset_params (dict): Parameters for the dataset
        train_dataset_params (dict): Parameters for the train dataset
        val_dataset_params (dict): Parameters for the validation dataset
        test_dataset_params (dict): Parameters for the test dataset
        dataloader_params (DataLoadersParams): Parameters for the dataloader
        train_dataloader_params (DataLoadersParams): Parameters for the train dataloader
        val_dataloader_params (DataLoadersParams): Parameters for the validation dataloader
        test_dataloader_params (DataLoadersParams): Parameters for the test dataloader
    """

    def __init__(
        self,
        mode: str,
        dataset: str,
        dataset_params: Optional[dict] = None,
        train_dataset_params: Optional[dict] = None,
        val_dataset_params: Optional[dict] = None,
        test_dataset_params: Optional[dict] = None,
        dataloader_params: Optional[DataLoadersParams] = None,
        train_dataloader_params: Optional[DataLoadersParams] = None,
        val_dataloader_params: Optional[DataLoadersParams] = None,
        test_dataloader_params: Optional[DataLoadersParams] = None,
    ):
        super().__init__()
        
        # Validate mode
        if mode not in ["train", "test"]:
            raise ValueError(f"Mode must be 'train' or 'test', got {mode}")
        
        # Initialize parameters with defaults
        dataset_params = dataset_params or {}
        train_dataset_params = train_dataset_params or {}
        val_dataset_params = val_dataset_params or {}
        test_dataset_params = test_dataset_params or {}
        dataloader_params = dataloader_params or {}
        train_dataloader_params = train_dataloader_params or {}
        val_dataloader_params = val_dataloader_params or {}
        test_dataloader_params = test_dataloader_params or {}

        self.mode = mode
        self.dataset_class = dataset
        self.dataset_params = dataset_params
        self.train_dataset_params = train_dataset_params
        self.val_dataset_params = val_dataset_params
        self.test_dataset_params = test_dataset_params
        self.dataloader_params = dataloader_params
        self.train_dataloader_params = train_dataloader_params
        self.val_dataloader_params = val_dataloader_params
        self.test_dataloader_params = test_dataloader_params

        # Initialize datasets
        self.data_train = None
        self.data_val = None
        self.data_test = None

        # Setup datasets based on mode
        self._setup_datasets()

    def _setup_datasets(self):
        """Setup datasets based on the mode."""
        try:
            if self.mode == "train":
                logger.info(f"Setting up train dataset: {self.dataset_class}")
                
                # Convert DictConfig to dict if needed
                dataset_params_dict = self.dataset_params
                train_params_dict = self.train_dataset_params
                if hasattr(dataset_params_dict, '_content'):
                    dataset_params_dict = dict(dataset_params_dict)
                if hasattr(train_params_dict, '_content'):
                    train_params_dict = dict(train_params_dict)
                
                self.data_train = utils.get_instance(
                    self.dataset_class, 
                    {"mode": "train"} | dataset_params_dict | train_params_dict
                )
                
                logger.info(f"Setting up validation dataset: {self.dataset_class}")
                
                val_params_dict = self.val_dataset_params
                if hasattr(val_params_dict, '_content'):
                    val_params_dict = dict(val_params_dict)
                
                self.data_val = utils.get_instance(
                    self.dataset_class, 
                    {"mode": "val"} | dataset_params_dict | val_params_dict
                )
                
            elif self.mode == "test":
                logger.info(f"Setting up test dataset: {self.dataset_class}")
                
                dataset_params_dict = self.dataset_params
                test_params_dict = self.test_dataset_params
                if hasattr(dataset_params_dict, '_content'):
                    dataset_params_dict = dict(dataset_params_dict)
                if hasattr(test_params_dict, '_content'):
                    test_params_dict = dict(test_params_dict)
                
                self.data_test = utils.get_instance(
                    self.dataset_class, 
                    {"mode": "test"} | dataset_params_dict | test_params_dict
                )
                
        except Exception as e:
            logger.error(f"Failed to instantiate dataset {self.dataset_class}: {str(e)}")
            raise RuntimeError(f"Dataset instantiation failed: {str(e)}")

    def _get_collate_fn(self, dataset):
        """Get collate function from dataset if available."""
        if hasattr(dataset, "collate_fn") and dataset.collate_fn is not None:
            return dataset.collate_fn
        return None

    def _get_dataloader_kwargs(self, dataset, dataloader_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get dataloader keyword arguments with proper defaults and dataset-specific settings."""
        # Convert DictConfig to dict if needed
        base_params = self.dataloader_params
        if hasattr(base_params, '_content'):
            base_params = dict(base_params)
        else:
            base_params = base_params.copy()
            
        specific_params = dataloader_params
        if hasattr(specific_params, '_content'):
            specific_params = dict(specific_params)
        
        kwargs = base_params.copy()
        kwargs.update(specific_params)
        
        # Add collate function if available
        collate_fn = self._get_collate_fn(dataset)
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
            
        # Note: worker_init_fn removed to avoid multiprocessing pickling issues
        
        return kwargs

    def train_dataloader(self):
        """Return training dataloader."""
        if self.data_train is None:
            raise RuntimeError("Training dataset not initialized. Make sure mode is 'train'.")
        
        kwargs = self._get_dataloader_kwargs(self.data_train, self.train_dataloader_params)
        return DataLoader(self.data_train, **kwargs)

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.data_val is None:
            raise RuntimeError("Validation dataset not initialized. Make sure mode is 'train'.")
        
        kwargs = self._get_dataloader_kwargs(self.data_val, self.val_dataloader_params)
        return DataLoader(self.data_val, **kwargs)

    def test_dataloader(self):
        """Return test dataloader."""
        if self.data_test is None:
            raise RuntimeError("Test dataset not initialized. Make sure mode is 'test'.")
        
        kwargs = self._get_dataloader_kwargs(self.data_test, self.test_dataloader_params)
        return DataLoader(self.data_test, **kwargs)
    
    def on_exception(self, exception: Exception) -> None:
        """Handle exceptions during training."""
        logger.error(f"DataModule encountered an exception: {exception}")
        # Clean up any resources if needed
