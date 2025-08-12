"""
This file contains the implementation of the MultiCriterion class.
"""

from typing import Any, List

import torch

from ..utils import utils
from ..utils.typing import CriterionConfig


class MultiCriterion(torch.nn.Module):
    """
    This class represents a multi-criterion.

    Args:
        criterions (List[CriterionConfig | str | torch.nn.Module]): The criterions.
        criterion_weights (list | None): The weights of the criterions. Defaults to None.
        reduce (str): The reduction method. Defaults to "mean".

    Raises:
        ValueError: If the reduce is invalid.

    Examples:
        >>> criterions = [
        ...     {"criterion": "torch.nn.CrossEntropyLoss", "criterion_params": {}},
        ...     "torch.nn.MSELoss",
        ...     torch.nn.L1Loss(),
        ... ]
        >>> criterion_weights = [1, 2]
        >>> reduce = "mean"
        >>> multi_criterion = MultiCriterion(criterions, criterion_weights, reduce)
    """

    def __init__(
        self,
        criterions: List[CriterionConfig | str | torch.nn.Module],
        criterion_weights: list | None = None,
        reduce: str = "mean",
    ):
        super().__init__()
        self.criterions = torch.nn.ModuleList()
        for criterion in criterions:
            if isinstance(criterion, dict):
                # Patch: Weight list -> torch.Tensor
                if "weight" in criterion["criterion_params"]:
                    criterion["criterion_params"]["weight"] = torch.Tensor(
                        criterion["criterion_params"]["weight"]
                    )

                self.criterions.append(
                    utils.get_instance(
                        criterion["criterion"], criterion["criterion_params"]
                    )
                )
            elif isinstance(criterion, str):
                self.criterions.append(utils.get_instance(criterion, {}))
            elif isinstance(criterion, torch.nn.Module):
                self.criterions.append(criterion)
        self.criterion_weights = (
            criterion_weights
            if criterion_weights is not None
            else [1] * len(self.criterions)
        )
        self.reduce = reduce

    def forward(self, x: Any, y: Any):
        """
        This function calculates the loss.

        Args:
            x (Any): The input.
            y (Any): The target.

        Returns:
            torch.Tensor: The loss.
        """

        losses = []
        for i, criterion in enumerate(self.criterions):
            losses.append(criterion(x, y) * self.criterion_weights[i])
        if self.reduce == "mean":
            return sum(losses) / len(losses)
        if self.reduce == "sum":
            return sum(losses)
        if self.reduce is None:
            return losses
        raise ValueError(f"Invalid reduce: {self.reduce}")