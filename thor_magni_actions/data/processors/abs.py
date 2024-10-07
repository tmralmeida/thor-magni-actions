import os
import logging
from typing import Dict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from thor_magni_actions.io import create_dir

LOGGER = logging.getLogger(__name__)


class Processor(ABC):
    """Preprocessors abstract class"""

    def __init__(self, actions_path: str, min_speed: int, max_speed: int, **kwargs) -> None:
        super().__init__()
        self.actions_df = pd.read_csv(actions_path, index_col=0)
        self.min_speed = min_speed
        self.max_speed = max_speed

    @abstractmethod
    def process_inputs(self, src_path: str) -> Dict:
        pass

    def compute_polar_angle(self, trajectories: pd.DataFrame) -> pd.DataFrame:
        out_df = []
        for agent in trajectories.ag_id.unique():
            trajectory = trajectories[trajectories["ag_id"] == agent]
            polars_df = trajectory.copy()
            period = polars_df.index.to_series().diff().iloc[-1]
            polars_df["theta_delta"] = np.arctan2(
                polars_df["y_delta"], polars_df["x_delta"]
            )
            polars_df["ang_speed"] = polars_df["theta_delta"] / period
            out_df.append(polars_df)
        return pd.concat(out_df)

    def save_data(self, dst_path: str, data: Dict):
        """save statistics and organized data"""
        for subdir_name, subdir in data.items():
            if isinstance(subdir, pd.DataFrame):
                create_dir(dst_path)
                subdir.to_csv(os.path.join(dst_path, subdir_name + ".csv"))
                continue
            for fn, data_to_save in subdir.items():
                save_path = os.path.join(dst_path, subdir_name)
                create_dir(save_path)
                data_to_save.to_csv(os.path.join(save_path, fn))

    def run(self, src_path, dst_path):
        # computing displacements, polar coordinates
        data = self.process_inputs(src_path=src_path)
        LOGGER.info("Speed, displacements and polar coordinates extracted!")
        self.save_data(dst_path, data)
        LOGGER.info("Data saved!")
