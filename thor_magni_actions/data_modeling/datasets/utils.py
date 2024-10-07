from typing import Union, List
import torch
import numpy as np


TRANSLATED_ROTATED_TRAJECTORIES_FEATURES_NAMES = [
    "translated_origin",
    "translated_rotated",
]


class PixWorldConverter:
    """Pixel to world converter"""

    def __init__(self, info: dict) -> None:
        self.resolution = info["resolution_pm"]  # 1pix -> m
        self.offset = np.array(info["offset"])

    def convert2pixels(
        self, world_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        if world_locations.ndim == 2:
            return (world_locations / self.resolution) - self.offset

        new_world_locations = [
            self.convert2pixels(world_location) for world_location in world_locations
        ]
        return (
            torch.stack(new_world_locations)
            if isinstance(world_locations, torch.Tensor)
            else np.stack(new_world_locations)
        )

    def convert2world(
        self, pix_locations: Union[np.array, torch.Tensor]
    ) -> Union[np.array, torch.Tensor]:
        return (pix_locations + self.offset) * self.resolution


def get_features_names(features_names_in: List[str]):
    """Get features names from highlevel description"""
    features_names_out = {}
    for feature in features_names_in:
        if feature == "translated_origin":
            features_names_out[feature] = ["x", "y"]
        if feature == "translated_rotated":
            features_names_out[feature] = ["x", "y"]
        if feature == "positions":
            features_names_out[feature] = ["x", "y"]
        if feature == "velocities":
            features_names_out[feature] = ["x_speed", "y_speed"]
        if feature == "accelerations":
            features_names_out[feature] = ["x_acc", "y_acc"]
    return features_names_out
