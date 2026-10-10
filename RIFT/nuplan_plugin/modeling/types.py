from typing import Dict, List

import torch


class MissingFeature(Exception):
    """
    Exception used when a features is not present
    """

    pass


FeaturesType = Dict[str, any]
TargetsType = Dict[str, any]
ScenarioListType = List[any]
TensorFeaturesType = Dict[str, torch.Tensor]


def move_features_type_to_device(batch: FeaturesType, device: torch.device) -> FeaturesType:
    """
    Move all features to a device
    :param batch: batch of features
    :param device: new device
    :return: batch moved to new device
    """
    output = {}
    for key, value in batch.items():
        output[key] = value.to_device(device)
    return output
