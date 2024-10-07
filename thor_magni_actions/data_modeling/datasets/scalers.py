from abc import ABC, abstractmethod
import torch


class TrajectoriesNormalizer:
    @abstractmethod
    def translate_to_origin(trajectory):
        origin = trajectory[0]
        translated_trajectory = trajectory - origin
        return translated_trajectory, origin

    @abstractmethod
    def align_to_x_axis(trajectory):
        start_point = trajectory[0]
        end_point = trajectory[1]
        angle = -torch.atan2(
            end_point[1] - start_point[1], end_point[0] - start_point[0]
        )
        rotation_matrix = torch.stack(
            [
                torch.stack([angle.cos(), -angle.sin()]),
                torch.stack([angle.sin(), angle.cos()]),
            ],
            dim=1,
        )
        aligned_trajectory = trajectory @ rotation_matrix
        return aligned_trajectory, angle

    @abstractmethod
    def normalize_velocity(trajectory):
        lengths = torch.sqrt(torch.sum(torch.diff(trajectory, dim=0) ** 2, dim=1))
        total_length = lengths.sum()
        if total_length == 0:
            return trajectory, total_length
        normalized_trajectory = trajectory / total_length
        return normalized_trajectory, total_length

    @abstractmethod
    def normalize(dataset: torch.Tensor, ori: bool, rot: bool, sca: bool):
        dataset = dataset if dataset.ndim == 3 else dataset.unsqueeze(dim=0)
        norm_trajectories = []
        for trajectory in dataset:
            if ori:
                trajectory, _ = TrajectoriesNormalizer.translate_to_origin(trajectory)
            if rot:
                trajectory, _ = TrajectoriesNormalizer.align_to_x_axis(trajectory)
            if sca:
                trajectory, _ = TrajectoriesNormalizer.normalize_velocity(trajectory)
            norm_trajectories.append(trajectory)
        return torch.stack(norm_trajectories)

    @abstractmethod
    def denormalize(
        raw_dataset: torch.Tensor,
        norm_dataset: torch.Tensor,
        ori: bool,
        rot: bool,
        sca: bool,
    ):
        if raw_dataset.ndim != norm_dataset.ndim:
            raise ValueError("raw_dataset and norm_dataset must have the same shapes")

        if raw_dataset.ndim != 3:
            raw_dataset = raw_dataset.unsqueeze(dim=0)
            norm_dataset = norm_dataset.unsqueeze(dim=0)

        denorm_trajectories = []
        for raw_trajectory, norm_trajectory in zip(raw_dataset, norm_dataset):
            if sca:
                _, traj_length = TrajectoriesNormalizer.normalize_velocity(
                    raw_trajectory
                )
                trajectory = norm_trajectory * traj_length
            if rot:
                _, rotation_matrix = TrajectoriesNormalizer.align_to_x_axis(
                    raw_trajectory
                )
                trajectory = norm_trajectory @ rotation_matrix.transpose(-1, -2)
            if ori:
                _, origin = TrajectoriesNormalizer.translate_to_origin(raw_trajectory)
                trajectory = norm_trajectory + origin
            denorm_trajectories.append(trajectory)
        return torch.stack(denorm_trajectories)


class TraditionalScalers(ABC):
    def __init__(self) -> None:
        super().__init__()

    def convert_to_2d(self, x: torch.Tensor):
        if x.ndim == 3:
            return x.reshape(x.shape[0], -1)
        return x

    def convert_to_original_shape(self, x_ori: torch.Tensor, x_tgt: torch.Tensor):
        original_shape = x_ori.shape
        return x_tgt.reshape(*original_shape)

    @abstractmethod
    def calculate_params(self, x: torch.Tensor):
        pass

    @abstractmethod
    def scale(self, x: torch.Tensor):
        pass

    @abstractmethod
    def descale(self, x: torch.Tensor):
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass


class StandardScaler(TraditionalScalers):
    """Scale features with shape (num_peds, length_of_time, 2) to zero mean std 1"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def calculate_params(self, x: torch.Tensor):
        x = self.convert_to_2d(x.clone())
        self.input_mean = x.mean(0, keepdim=True)
        self.input_std = x.std(0, unbiased=False, keepdim=True) + 1e-18

    def scale(self, x: torch.Tensor):
        _in = x.clone()
        _out = self.convert_to_2d(_in)

        _out -= self.input_mean
        _out /= self.input_std
        out = self.convert_to_original_shape(_in, _out)
        return out

    def descale(self, x: torch.Tensor):
        _in = x.clone()
        _out = self.convert_to_2d(_in)

        _out *= self.input_std
        _out += self.input_mean
        out = self.convert_to_original_shape(_in, _out)
        return out

    def set_params(self, **kwargs):
        self.input_mean, self.input_std = kwargs["input_mean"], kwargs["input_std"]


class MinMaxNormalizer(TraditionalScalers):
    """Normalize features with shape (num_peds, length_of_time, 2) to range_min, range_max

    Args:
       feature_range = tuple (min, max), default=(-1, 1)

    """

    def __init__(self, feature_range=(-1, 1)) -> None:
        self.feature_min, self.feature_max = feature_range[0], feature_range[1]

    def calculate_params(self, x):
        x = self.convert_to_2d(x.clone())
        self.input_min = x.min(0, keepdim=True)[0]
        self.input_max = x.max(0, keepdim=True)[0]

    def scale(self, x: torch.Tensor):
        _in = x.clone()
        _out = self.convert_to_2d(_in)
        x_std = (_out - self.input_min) / (self.input_max - self.input_min)
        x_normalized = x_std * (self.feature_max - self.feature_min) + self.feature_min
        out = self.convert_to_original_shape(_in, x_normalized)
        return out

    def descale(self, x: torch.Tensor):
        _in = x.clone()
        _out = self.convert_to_2d(_in)

        x_std_denormalized = (_out - self.feature_min) / (
            self.feature_max - self.feature_min
        )
        x_denormalized = (
            x_std_denormalized * (self.input_max - self.input_min) + self.input_min
        )
        out = self.convert_to_original_shape(_in, x_denormalized)

        return out

    def set_params(self, **kwargs):
        self.input_min, self.input_max = kwargs["input_min"], kwargs["input_max"]
        self.feature_min, self.feature_max = (
            kwargs["feature_min"],
            kwargs["feature_max"],
        )


def get_scalers_stats(train_features_scaler: dict, **kwargs):
    scalers_stats = {}
    for feature_name, feature_scaler in train_features_scaler.items():
        if isinstance(feature_scaler, MinMaxNormalizer):
            scalers_stats[feature_name] = dict(
                input_min=feature_scaler.input_min,
                input_max=feature_scaler.input_max,
                feature_min=feature_scaler.feature_min,
                feature_max=feature_scaler.feature_max,
            )
        elif isinstance(feature_scaler, StandardScaler):
            scalers_stats[feature_name] = dict(
                input_mean=feature_scaler.input_mean, input_std=feature_scaler.input_std
            )
        else:
            continue
    return scalers_stats


def build_features_scaler_from_stats(features_scalers_stats: dict, output_type: str):
    """build scaler objects from statistics"""
    stats_names = features_scalers_stats[output_type].keys()
    if "input_mean" in stats_names:
        scaler = StandardScaler()
    elif "input_min" in stats_names:
        scaler = MinMaxNormalizer()
    else:
        raise NotImplementedError(f"Only MinMax and StandardScaler available got {stats_names}")
    scaler.set_params(**features_scalers_stats[output_type])
    return scaler