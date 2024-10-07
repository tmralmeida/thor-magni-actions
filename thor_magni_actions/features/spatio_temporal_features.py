import logging
from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

LOGGER = logging.getLogger(__name__)


class SpatioTemporalFeatures:
    @staticmethod
    def get_delta_columns(input_df: pd.DataFrame):
        out_df = input_df.copy()
        out_df = out_df.diff().add_suffix("_delta")
        out_df.loc[:, "Time_delta"] = out_df.index.to_series().diff()
        return out_df

    @staticmethod
    def get_acceleration(
        trajectories: pd.DataFrame,
        speed_col_name: str = "speed",
        out_col_name: str = "acc",
    ) -> List[pd.DataFrame]:
        """it receives a pandas dataframes of trajectories and it computes the acceleration
        Time must be passed as the index of the dataframe

        trajectories[index]:
            | frame_id | ag_id | x | y | z |

        Returns accelerations[index]:
            | frame_id | ag_id | x | y | z | x_acc | y_acc | z_acc | 2D_acc | 3D_acc
        """
        if f"2D_{speed_col_name}" not in trajectories.columns:
            trajectories = SpatioTemporalFeatures.get_speed(trajectories)
        input_cols_name = ["x", "y", "2D"]
        target_cols_name = [
            f"x_{out_col_name}",
            f"y_{out_col_name}",
            f"2D_{out_col_name}",
        ]
        if "z" in trajectories.columns:
            input_cols_name.extend(["z", "3D"])
            target_cols_name.extend([f"z_{out_col_name}", f"3D_{out_col_name}"])
        agents = trajectories["ag_id"].unique()
        acceleration_dfs = []
        for agent in agents:
            agent_trajectory = trajectories[trajectories["ag_id"] == agent]
            delta_df = SpatioTemporalFeatures.get_delta_columns(
                agent_trajectory[
                    [f"{coord}_{speed_col_name}" for coord in input_cols_name]
                ]
            )
            delta_df.loc[
                :,
                target_cols_name,
            ] = (
                delta_df[[f"{coord}_{speed_col_name}_delta" for coord in input_cols_name]]
                .div(delta_df["Time_delta"].values, axis=0)
                .values
            )
            agent_trajectory = agent_trajectory.join(delta_df[target_cols_name])
            acceleration_dfs.append(agent_trajectory)
        acceleration_dfs = pd.concat(acceleration_dfs)
        LOGGER.info("%s created", target_cols_name)
        return acceleration_dfs

    @staticmethod
    def get_speed(
        trajectories: pd.DataFrame,
        out_col_name: str = "speed",
    ) -> List[pd.DataFrame]:
        """it receives a pandas dataframes of trajectories and it computes the speed.
        Time must be passed as the index of the dataframe

        trajectories:
            | frame_id | ag_id | x | y | z |

        Returns speeds:
            | frame_id | ag_id | x | y | z | x_speed | y_speed | z_speed | 2D_speed | 3D_speed
        """
        input_cols_name = ["x", "y"]
        target_cols_name = [
            "x_delta",
            "y_delta",
            f"x_{out_col_name}",
            f"y_{out_col_name}",
            "2D_norm_delta",
            f"2D_{out_col_name}",
        ]
        if "z" in trajectories.columns:
            input_cols_name.append("z")
            target_cols_name.extend(
                ["z_delta", f"z_{out_col_name}", "3D_norm_delta", f"3D_{out_col_name}"]
            )

        agents = trajectories["ag_id"].unique()
        speeds_agents = []
        for agent in agents:
            agent_trajectory = trajectories[trajectories["ag_id"] == agent]
            delta_df = SpatioTemporalFeatures.get_delta_columns(
                agent_trajectory[input_cols_name]
            )
            delta_df["2D_norm_delta"] = np.sqrt(
                np.square(delta_df[["x_delta", "y_delta"]]).sum(axis=1)
            )
            delta_df.loc[:, f"2D_{out_col_name}"] = (
                delta_df["2D_norm_delta"]
                .div(delta_df["Time_delta"].values, axis=0)
                .values
            )
            delta_df.loc[
                :, [f"{coord}_{out_col_name}" for coord in input_cols_name]
            ] = (
                delta_df[[f"{coord}_delta" for coord in input_cols_name]]
                .div(delta_df["Time_delta"].values, axis=0)
                .values
            )
            if "z" not in trajectories.columns:
                agent_trajectory = agent_trajectory.join(delta_df[target_cols_name])
                speeds_agents.append(agent_trajectory)
                continue
            delta_df["3D_norm_delta"] = np.sqrt(
                np.square(delta_df[["x_delta", "y_delta", "z_delta"]]).sum(axis=1)
            )
            delta_df.loc[:, f"3D_{out_col_name}"] = (
                delta_df["3D_norm_delta"]
                .div(delta_df["Time_delta"].values, axis=0)
                .values
            )
            agent_trajectory = agent_trajectory.join(delta_df[target_cols_name])
            speeds_agents.append(agent_trajectory)
        speeds_agents = pd.concat(speeds_agents)
        LOGGER.info("%s created", target_cols_name)
        return speeds_agents

    @staticmethod
    def get_path_efficiency_index(
        trajectories: pd.DataFrame,
        out_col_name: str = "path_efficiency",
    ) -> List[pd.DataFrame]:
        """it receives a pandas dataframes of trajectories and it computes the path
        efficiency. This feature is given by the distance between the origin and destination
        divided by the cumulative displacements between the origin and the destination.
        Time must be passed as the index of the dataframe

        trajectories:
            | frame_id | ag_id | x | y | z |

        Returns path_effficiency:
            | frame_id | ag_id | x | y | z | path_effficiency |
        """
        si_dfs = []
        agents = trajectories["ag_id"].unique()
        for agent in agents:
            agent_trajectory = trajectories[trajectories["ag_id"] == agent].copy()
            if not set(["x_delta", "y_delta"]).issubset(agent_trajectory.columns):
                agent_trajectory = pd.concat(
                    [
                        agent_trajectory,
                        SpatioTemporalFeatures.get_delta_columns(
                            agent_trajectory[["x", "y"]]
                        ),
                    ],
                    axis=1,
                )
            if "norm_delta_2D" not in agent_trajectory.columns:
                agent_trajectory["norm_delta_2D"] = np.sqrt(
                    np.square(agent_trajectory[["x_delta", "y_delta"]]).sum(axis=1)
                )
            agent_trajectory["cumsum_delta"] = agent_trajectory[
                "norm_delta_2D"
            ].cumsum()
            first_location = (
                agent_trajectory["x"].iloc[0],
                agent_trajectory["y"].iloc[0],
            )
            agent_trajectory["dist_origin_loc_i"] = agent_trajectory.apply(
                lambda row: euclidean(first_location, (row["x"], row["y"])), axis=1
            )
            agent_trajectory[out_col_name] = (
                agent_trajectory["dist_origin_loc_i"] / agent_trajectory["cumsum_delta"]
            )
            agent_trajectory[out_col_name] = agent_trajectory[out_col_name].fillna(1.0)
            si_dfs.append(agent_trajectory)
        si_dfs = pd.concat(si_dfs)
        LOGGER.info("%s created", out_col_name)
        return si_dfs

    @staticmethod
    def get_heading_orientation(
        trajectories: pd.DataFrame, out_col_name: str = "heading"
    ):
        """it receives a pandas dataframes of trajectories and it computes the heading
        orientation.
        Time must be passed as the index of the dataframe

        trajectories:
            | frame_id | ag_id | x | y | z |

        Returns path_effficiency:
            | frame_id | ag_id | x | y | z | heading |
        """
        heading_dfs = []
        agents = trajectories["ag_id"].unique()
        for agent in agents:
            agent_trajectory = trajectories[trajectories["ag_id"] == agent].copy()
            if not set(["x_delta", "y_delta"]).issubset(agent_trajectory.columns):
                agent_trajectory = SpatioTemporalFeatures.get_delta_columns(
                    agent_trajectory[["x", "y"]]
                )
            agent_trajectory[out_col_name] = np.arctan2(
                agent_trajectory["y_delta"], agent_trajectory["x_delta"]
            )
            heading_dfs.append(agent_trajectory)

        heading_dfs = pd.concat(heading_dfs)
        LOGGER.info("%s created", out_col_name)
        return heading_dfs


def get_spatiotemporal_features(trajectories: pd.DataFrame) -> pd.DataFrame:
    trajectories[["x", "y", "z"]] /= 1000
    trajectories = trajectories.set_index("Time", drop=True)
    speed = SpatioTemporalFeatures.get_speed(trajectories)
    acc = SpatioTemporalFeatures.get_acceleration(speed)
    heading = SpatioTemporalFeatures.get_heading_orientation(acc)
    return heading
