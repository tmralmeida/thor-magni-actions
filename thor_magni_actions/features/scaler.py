from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TrajectoriesScaler:
    def __init__(self, tracklets_group_cols: List[str] | str) -> None:
        self.tracklets_group_cols = tracklets_group_cols
        self.available_features = [
            "trajectories",
            "velocities",
            "accelerations",
            "headings",
        ]

    def scale_dataset(self, data: pd.DataFrame):
        """standard scaling features of a trajectories dataset"""
        out_df = data.copy()
        traj_scaler = StandardScaler()
        vel_scaler = StandardScaler()
        acc_scaler = StandardScaler()
        orientation_scaler = StandardScaler()

        trajectories, velocities, accelerations, orientations = [], [], [], []
        tracklets_info = data.groupby(self.tracklets_group_cols)
        for _, tracklet in tracklets_info:
            trajectories.append(tracklet[["x", "y"]].values.flatten())
            velocities.append(tracklet[["x_speed", "y_speed"]].values.flatten())
            accelerations.append(tracklet[["x_acc", "y_acc"]].values.flatten())
            orientations.append(tracklet["heading"].values.flatten())

        trajectories = np.stack(trajectories).astype(float)
        velocities = np.stack(velocities).astype(float)
        accelerations = np.stack(accelerations).astype(float)
        orientations = np.stack(orientations).astype(float)

        trajectories_scaled = traj_scaler.fit_transform(trajectories)
        velocities_scaled = vel_scaler.fit_transform(velocities)
        accelerations_scaled = acc_scaler.fit_transform(accelerations)
        orientations_scaled = orientation_scaler.fit_transform(orientations)

        self.scalers = dict(
            trajectories=traj_scaler,
            velocities=vel_scaler,
            accelerations=acc_scaler,
            headings=orientation_scaler,
        )
        n_points = data.shape[0]
        out_df.loc[:, ["x_scl", "y_scl"]] = trajectories_scaled.reshape(n_points, 2)
        out_df.loc[:, ["x_speed_scl", "y_speed_scl"]] = velocities_scaled.reshape(
            n_points, 2
        )
        out_df.loc[:, ["x_acc_scl", "y_acc_scl"]] = accelerations_scaled.reshape(
            n_points, 2
        )
        out_df.loc[:, ["heading_scl"]] = orientations_scaled.reshape(n_points, 1)
        return out_df

    def descale_features(self, features: dict) -> dict:
        features_scaled = {}
        for feature_name, feature_values in features.items():
            if feature_name not in self.available_features:
                raise ValueError(f"{feature_name} not available")
            features_scaled[feature_name] = (
                self.scalers[feature_name]
                .inverse_transform(feature_values.flatten()[None, :])
                .reshape(-1, feature_values.shape[-1])
            )
        return features_scaled

    def scale_features(self, features: dict) -> dict:
        features_scaled = {}
        for feature_name, feature_values in features.items():
            if feature_name not in self.available_features:
                raise ValueError(f"{feature_name} not available")
            features_scaled[feature_name] = (
                self.scalers[feature_name]
                .transform(feature_values.flatten()[None, :])
                .reshape(-1, feature_values.shape[-1])
            )
        return features_scaled

    def transform_velocities_to_positions(
        self, speeds: np.array, initial_2d_points: np.array, period: float
    ):
        """scaled speed -> unscaled speed -> displacements -> locations"""
        if initial_2d_points.ndim == 1:
            initial_2d_points = np.expand_dims(initial_2d_points, axis=0)
        unscaled_speeds = self.descale_features(dict(velocities=speeds))["velocities"]
        displacements = unscaled_speeds * period
        displacements[:, 0, :] += initial_2d_points
        return np.cumsum(displacements, axis=1)
