import abc
from typing import List

import torch

from nuplan_plugin.trajectory.trajectory_sampling import TrajectorySampling
from nuplan_plugin.modeling.types import FeaturesType, TargetsType



class TorchModuleWrapper(torch.nn.Module):
    """Torch module wrapper that encapsulates builders for constructing model features and targets."""

    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        feature_builders: List[any],
        target_builders: List[any],
    ):
        """
        Construct a model with feature and target builders.
        :param future_trajectory_sampling: Parameters for a predicted trajectory.
        :param feature_builders: The list of builders which will compute features for this model.
        :param target_builders: The list of builders which will compute targets for this model.
        """
        super().__init__()

        self.future_trajectory_sampling = future_trajectory_sampling
        self.feature_builders = feature_builders
        self.target_builders = target_builders

    def get_list_of_required_feature(self) -> List[any]:
        """Get list of required input features to the model."""
        return self.feature_builders

    def get_list_of_computed_target(self) -> List[any]:
        """Get list of features that the model computes."""
        return self.target_builders

    @abc.abstractmethod
    def forward(self, features: FeaturesType) -> TargetsType:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
        :return: The results of the inference as a TargetsType.
        """
        pass
