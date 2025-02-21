import os
import math
from typing import Dict
from tqdm import tqdm
import pandas as pd

from .abs import Processor


SCENARIOS_MAP_NAME = {
    "Scenario_1": ["SC1A", "SC1B"],
    "Scenario_2": ["SC2"],
    "Scenario_3": ["SC3A", "SC3B"],
    "Scenario_4": ["SC4A", "SC4B"],
    "Scenario_5": ["SC5"],
}

AGENT_CLASS_REPLACE = {
    "Carrier-Large Object Leader": "Carrier-Large Object",
    "Carrier-Large Object Follower": "Carrier-Large Object",
    "Visitors-Group 2": "Visitors-Group",
    "Visitors-Group 3": "Visitors-Group",
}


class MagniProcessor(Processor):
    def __init__(
        self, actions_path: str, min_speed: int, max_speed: int, **kwargs
    ) -> None:
        super().__init__(actions_path, min_speed, max_speed, **kwargs)
        self.traj_len = kwargs["traj_len"]
        self.skip_window = kwargs["skip_window"]
        self.min_pedestrians = kwargs["min_pedestrians"]
        self.tracking_cols = None

    def get_scenario_namming(self, scenario_file_name: str) -> str:
        for k, v in SCENARIOS_MAP_NAME.items():
            if scenario_file_name in v:
                return k

    def extract_trajectories(
        self, helmets_df: pd.DataFrame, tracklet_id_init: int
    ) -> pd.DataFrame:
        agents = helmets_df["ag_id"].unique()
        tracklet_id = tracklet_id_init
        tracklets = []
        for agent_id in agents:
            groups_of_continuous_tracking = self.get_groups_continuous_tracking(
                helmets_df[helmets_df["ag_id"] == agent_id]
            )
            for _, group in groups_of_continuous_tracking:
                if group[self.tracking_cols].isna().any(axis=0).all():
                    continue
                num_tracklets = int(
                    math.ceil((len(group) - self.traj_len + 1) / self.traj_len)
                )
                if num_tracklets == 0:
                    continue

                for i in range(0, num_tracklets * self.traj_len, self.traj_len):
                    tracklet = group.iloc[i: i + self.traj_len]
                    trajectory = tracklet.copy()
                    trajectory.loc[:, "tracklet_id"] = tracklet_id
                    tracklet_id += 1
                    tracklets.append(trajectory)
        return pd.concat(tracklets)

    def get_groups_continuous_tracking(self, dynamic_agent_data: pd.DataFrame):
        """get groups of continuous tracking/no-tracking"""
        mask = dynamic_agent_data[self.tracking_cols].isna().any(axis=1)
        groups = (mask != mask.shift()).cumsum()
        groups_of_continuous_tracking = dynamic_agent_data.groupby(groups)
        return groups_of_continuous_tracking

    def merge_actions_trajectories(
        self,
        humans_trajectories_df: pd.DataFrame,
        file_actions: pd.DataFrame,
    ) -> pd.DataFrame:
        act_trajs_dfs = []
        actions_helmets = file_actions["ag_id"].unique()
        for helmet_id in actions_helmets:
            helmet_trajs_df = humans_trajectories_df[
                humans_trajectories_df["ag_id"] == helmet_id
            ]
            helmet_act_df = file_actions[file_actions.ag_id == helmet_id]
            merged_df = pd.merge_asof(
                helmet_trajs_df.sort_values("frame_id"),
                helmet_act_df[["file_name", "qtm_frame_act", "action"]].sort_values(
                    "qtm_frame_act"
                ),
                left_on="frame_id",
                right_on="qtm_frame_act",
                direction="nearest",
                # tolerance=None,
            )
            act_trajs_dfs.append(merged_df)
        actions_trajs_merged = pd.concat(act_trajs_dfs).set_index("Time").sort_index()
        return actions_trajs_merged

    def process_inputs(self, src_path: str) -> Dict:
        scenarios = sorted(os.listdir(src_path))
        files_save = {sc_id: [] for sc_id in scenarios}
        actions_df_fn = self.actions_df.groupby("file_name")

        for scenario_id in tqdm(scenarios, desc="scenarios"):
            ps = os.path.join(src_path, scenario_id)
            for _file in tqdm(sorted(os.listdir(ps)), desc="runs"):
                fp = os.path.join(ps, _file)
                if _file not in actions_df_fn.groups.keys():
                    continue
                file_actions = actions_df_fn.get_group(_file)
                if len(file_actions["ag_id"].unique()) == 0:
                    continue

                trajectories_df = pd.read_csv(fp)
                if self.tracking_cols is None:
                    self.tracking_cols = trajectories_df.columns[
                        trajectories_df.columns.str.startswith(("x", "y", "z", "rot"))
                    ]
                    print("Tracking columns:", self.tracking_cols)
                actions_trajs_merged = trajectories_df[
                    trajectories_df.ag_id.str.startswith("Helmet")
                ]
                if "action" not in actions_trajs_merged.columns:
                    actions_trajs_merged = self.merge_actions_trajectories(
                        humans_trajectories_df=actions_trajs_merged,
                        file_actions=file_actions,
                    )
                helmets_df = actions_trajs_merged.copy()
                helmets_df.loc[:, "dataset_name"] = scenario_id
                helmets_df = helmets_df.rename({"data_label": "agent_type"}, axis=1)
                helmets_df["agent_type"] = helmets_df["agent_type"].replace(
                    AGENT_CLASS_REPLACE
                )

                if len(files_save[scenario_id]) == 0:
                    tracklet_id_init = 0
                else:
                    tracklet_id_init = max(
                        [
                            files_data["tracklet_id"].max()
                            for files_data in files_save[scenario_id]
                        ]
                    )
                trajectories = self.extract_trajectories(
                    helmets_df, tracklet_id_init + 1
                )
                filtered_speed = trajectories.groupby("tracklet_id").filter(
                    lambda x: ~(
                        (x["2D_speed"] < self.min_speed)
                        | (x["2D_speed"] > self.max_speed)
                    ).any()
                )
                files_save[scenario_id].append(filtered_speed)

        for scenario_id, scenario_data in files_save.items():
            files_save[scenario_id] = pd.concat(scenario_data)
        return files_save
