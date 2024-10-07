import os
from collections import defaultdict
from typing import List, Optional
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from thor_magni_actions.io import load_json_file
from .utils import (
    PixWorldConverter,
    get_features_names,
    TRANSLATED_ROTATED_TRAJECTORIES_FEATURES_NAMES,
)
from .scalers import TrajectoriesNormalizer, MinMaxNormalizer, StandardScaler


class DatasetObjects:
    def __init__(self, set_type: str, data_cfg: dict, **kwargs) -> None:
        self.set_type = set_type
        self.cfg = data_cfg
        self.features_names_in = (
            [self.cfg["features_in"]]
            if isinstance(self.cfg["features_in"], str)
            else self.cfg["features_in"]
        )
        self.features_scaler = {}
        prior_dataset_frame = (
            kwargs["dataset_dataframe"] if "dataset_dataframe" in kwargs else None
        )
        self.init_dataset_paths(dataset_dataframe=prior_dataset_frame)

    def init_dataset_paths(self, dataset_dataframe=None):
        self.observation_len = self.cfg["observation_len"]
        self.trajectory_len = self.observation_len + self.cfg["prediction_len"]
        tested_ds = self.cfg.get("test_dataset")
        data_path = os.path.join(self.cfg["data_dir"], tested_ds or "")
        if dataset_dataframe is not None:
            self.dataset = dataset_dataframe
        else:
            self.dataset = DatasetFromPath.get_data(
                trajectories_path=data_path, set_type=self.set_type
            )

    def _set_feature_scaler(self, scaler_type, feature_name: str):
        scaler_mapping = {"minmax": MinMaxNormalizer, "standardscaler": StandardScaler}
        if feature_name not in self.features_scaler:
            scaler_class = scaler_mapping.get(scaler_type)
            if not scaler_class:
                raise ValueError(f"{scaler_type} not implemented!")
            self.features_scaler[feature_name] = scaler_class()

    def load_input_features(self):
        features_out = get_features_names(self.features_names_in)
        out_values = {}
        for f_in, f_out in features_out.items():
            feat_values = self.dataset[f_out].values.reshape(-1, self.trajectory_len, 2)
            if f_in == "velocities":
                feat_values[:, 0, :] = 0
            out_values[f_in] = feat_values
        return out_values

    def create_transformed_rotated_trajectories(self, features):
        features_in = defaultdict(dict)
        for feature_transf in TRANSLATED_ROTATED_TRAJECTORIES_FEATURES_NAMES:
            trajectories = features.get(feature_transf)
            if trajectories is None:
                continue
            trajectories = torch.tensor(trajectories, dtype=torch.float)
            obs_trajs = trajectories[:, :self.observation_len, :]
            pred_trajs = trajectories[:, self.observation_len:, :]
            normed_trajs = TrajectoriesNormalizer.normalize(
                trajectories,
                ori=True,
                rot=feature_transf == "translated_rotated",
                sca=False,
            )
            scl_obs, scl_pred = (
                normed_trajs[:, :self.observation_len, :],
                normed_trajs[:, self.observation_len:, :],
            )
            features_in[feature_transf].update(
                {
                    "scl_obs": scl_obs,
                    "scl_pred": scl_pred,
                    "gt_obs": obs_trajs,
                    "gt_pred": pred_trajs,
                }
            )
        return features_in

    def create_input_features(self, features=None, **kwargs):
        if features is None:
            features = self.load_input_features()
            gt_trajectories = self.dataset[["x", "y"]].values.reshape(
                -1, self.trajectory_len, 2
            )
        else:
            gt_trajectories = kwargs["gt_trajectories"]

        features_in = self.create_transformed_rotated_trajectories(features)

        scaler_type = self.cfg.get("features_trasnsformer_type", "standardscaler")
        for feature_name, feature_values in features.items():
            if feature_name in TRANSLATED_ROTATED_TRAJECTORIES_FEATURES_NAMES:
                continue
            feature_values = torch.from_numpy(feature_values).type(torch.float)
            if self.set_type == "train":
                self._set_feature_scaler(scaler_type, feature_name)
                self.features_scaler[feature_name].calculate_params(feature_values)
            scaled_features = self.features_scaler[feature_name].scale(feature_values)
            features_in[feature_name] = {
                "scl_obs": scaled_features[:, :self.observation_len, :],
                "scl_pred": scaled_features[:, self.observation_len:, :],
                "gt_obs": feature_values[:, :self.observation_len, :],
                "gt_pred": feature_values[:, self.observation_len:, :],
            }
        features_in["gt_obs"] = torch.from_numpy(
            gt_trajectories[:, :self.observation_len, :]
        ).type(torch.float)
        features_in["gt_pred"] = torch.from_numpy(
            gt_trajectories[:, self.observation_len:, :]
        ).type(torch.float)
        features_in["delta_time"] = torch.from_numpy(
            np.median(
                np.diff(
                    self.dataset.index.values.reshape(-1, self.trajectory_len), axis=1
                ),
                axis=1,
            )
        ).type(torch.float)

        return features_in

    def load_dataset(self):
        features_in = self.create_input_features()
        self.add_metadata(features_in)

        dataset_handler = {
            "thor_magni": MAGNIDataset,
            "synthetic": SyntheticDataset,
        }.get(
            self.cfg["dataset"], lambda x: None
        )(input_data=features_in, features_names=self.features_names_in)
        return dataset_handler

    def add_metadata(self, features_in):
        if "dataset_name" in self.dataset.columns:
            dataset_name = self.dataset["dataset_name"].values.reshape(
                -1, self.trajectory_len
            )[:, 0]
            features_in["dataset_name"] = dataset_name
        if "agent_type" in self.dataset.columns:
            prior_ag_type = self.dataset["agent_type"].values.reshape(
                -1, self.trajectory_len
            )
            features_in["agent_type"] = prior_ag_type
        if "action" in self.dataset.columns:
            prior_action = self.dataset["action"].values.reshape(
                -1, self.trajectory_len
            )
            features_in["action"] = prior_action


class DatasetFromPath:
    """Load trajectories form input path object"""

    @staticmethod
    def get_data(trajectories_path: str, set_type: str) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(trajectories_path, set_type + ".csv"), index_col=0
        )


class DeepLearningDataset(Dataset):
    """Default dataset loader object"""

    def __init__(self, input_data: dict, features_names: List[str]) -> None:
        super().__init__()
        self.features_names = features_names
        self.input_data = input_data

    def convert_to_torch(self, arr: np.array) -> torch.Tensor:
        return torch.from_numpy(arr).type(torch.float)

    def get_common_inputs(self, index):
        out_inputs = {"features": {}}
        for feature_name in self.features_names:
            feat_vals = {
                feat_type: self.input_data[feature_name][feat_type][index]
                for feat_type in self.input_data[feature_name].keys()
            }
            out_inputs["features"].update({feature_name: feat_vals})
        out_inputs.update(
            dict(
                gt_obs=self.input_data["gt_obs"][index],
                gt_pred=self.input_data["gt_pred"][index],
                delta_time=self.input_data["delta_time"][index],
            )
        )
        if "agent_type" in self.input_data:
            out_inputs.update(agent_type=self.input_data["agent_type"][index])
        if "action" in self.input_data:
            out_inputs.update(action=self.input_data["action"][index])
        return out_inputs

    def __getitem__(self, index):
        return self.get_common_inputs(index)

    def __len__(self):
        return len(self.input_data["gt_obs"])


class MAGNIDataset(DeepLearningDataset):
    def __init__(
        self,
        input_data: dict,
        features_names: List[str],
        visuals_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(input_data, features_names)
        self.mapping_agent_types = {
            label: i for i, label in enumerate(np.unique(self.input_data["agent_type"]))
        }
        self.mapping_actions = {
            label: i for i, label in enumerate(np.unique(self.input_data["action"]))
        }
        self.mapping_datasets = {
            ds: i for i, ds in enumerate(np.unique(self.input_data["dataset_name"]))
        }
        self.imgs, self.window_size, self.pix2world_conv = {}, None, None
        if visuals_path:
            visuals_info = load_json_file(os.path.join(visuals_path, "info.json"))
            self.obs_len = kwargs["obs_len"]
            self.pix2world_conv = PixWorldConverter(visuals_info)
            self.window_size = int(
                kwargs["window_size"] / self.pix2world_conv.resolution
            )
            datasets = np.unique(self.input_data["dataset_name"])
            for scenario in datasets:
                vis_path = os.path.join(visuals_path, scenario + ".png")
                img = np.array(Image.open(vis_path))
                self.imgs[scenario] = np.flipud(img[:, :, :3])

    def get_mapping_cat_vars(self, cat_var: str, mapping: dict):
        return self.convert_to_torch(np.array([mapping[cat_var]]))

    def __getitem__(self, index):
        new_inputs = self.get_common_inputs(index)
        roles = self.input_data["agent_type"][index]
        actions = self.input_data["action"][index]
        datasets = self.input_data["dataset_name"][index]
        if isinstance(datasets, np.ndarray):
            mapped_datasets = [
                self.get_mapping_cat_vars(dataset, self.mapping_datasets)
                for dataset in datasets
            ]
        else:
            mapped_datasets = self.get_mapping_cat_vars(datasets, self.mapping_datasets)
        if isinstance(roles, str):
            mapped_roles = self.get_mapping_cat_vars(roles, self.mapping_agent_types)
        elif isinstance(roles, np.ndarray):
            mapped_roles = torch.tensor([
                self.get_mapping_cat_vars(role, self.mapping_agent_types)
                for role in roles
            ])
        else:
            mapped_roles = roles
        if isinstance(actions, str):
            mapped_actions = self.get_mapping_cat_vars(
                roles, self.mapping_actions
            )
        elif isinstance(actions, np.ndarray):
            mapped_actions = torch.tensor([
                self.get_mapping_cat_vars(act, self.mapping_actions)
                for act in actions
            ])
        else:
            mapped_actions = actions
        new_inputs.update(
            {
                "agent_type": mapped_roles,
                "action": mapped_actions,
                "dataset": mapped_datasets,
            }
        )
        if len(self.imgs) > 0:
            dataset = self.input_data[index]["dataset_name"].iloc[0]
            visual = torch.from_numpy(
                self.imgs[dataset].transpose(2, 0, 1).copy()
            ).float()
            visual /= visual.max()
            trajs_pix = self.pix2world_conv.convert2pixels(new_inputs["trajectories"])
            last_pt = trajs_pix[self.obs_len - 1]
            col_min, col_max = (
                max(0, int(last_pt[0]) - self.window_size),
                min(int(last_pt[0]) + self.window_size, visual.shape[1] - 1),
            )
            row_min, row_max = (
                max(0, int(last_pt[1]) - self.window_size),
                min(int(last_pt[1]) + self.window_size, visual.shape[2] - 1),
            )
            target_visual = visual[:, row_min:row_max, col_min:col_max]
            new_inputs.update(dict(img=target_visual))
        return new_inputs


class SyntheticDataset(DeepLearningDataset):
    def __init__(self, input_data: dict, features_names: List[str]) -> None:
        super().__init__(input_data, features_names)
        self.mapping_agent_types = {
            label: i for i, label in enumerate(np.unique(self.input_data["agent_type"]))
        }
        self.mapping_actions = {
            label: i for i, label in enumerate(np.unique(self.input_data["action"]))
        }

    def get_mapping_cat_vars(self, cat_var: str, mapping: dict):
        return self.convert_to_torch(np.array([mapping[cat_var]]))

    def __getitem__(self, index):
        new_inputs = self.get_common_inputs(index)
        agent_types = self.input_data["agent_type"][index]
        actions = self.input_data["action"][index]

        mapped_agent_types = (
            self.get_mapping_cat_vars(agent_types, self.mapping_agent_types)
            if isinstance(agent_types, str)
            else agent_types
        )
        mapped_actions = (
            self.get_mapping_cat_vars(actions, self.mapping_actions)
            if isinstance(actions, str)
            else actions
        )
        new_inputs.update(
            {"agent_type": mapped_agent_types, "action": mapped_actions}
        )
        return new_inputs
