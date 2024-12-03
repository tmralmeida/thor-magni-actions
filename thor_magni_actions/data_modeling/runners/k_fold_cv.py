# -*- coding: utf-8 -*-
import os
import random
import logging
from pathlib import Path
import click
from dotenv import find_dotenv, load_dotenv
import ray
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from thor_magni_actions.data_modeling.cfgs.seeds import K_FOLD_SEED
from thor_magni_actions.data_modeling.utils import load_config, merge_dicts
from thor_magni_actions.io import load_json_file, dump_json_file
from thor_magni_actions.data_modeling.datasets import DatasetFromPath
from .utils import get_trainer_objects_kfold_cv, run_trainer


@ray.remote
def ray_run_trainer(trainer_options):
    return run_trainer(trainer_options)


@click.command()
@click.argument("k_folds", type=click.INT)
@click.argument("cfg_file", type=click.Path(exists=True))
def main(k_folds, cfg_file):
    """Cross validation on some dataset"""
    logger = logging.getLogger(__name__)
    cfg = load_config(cfg_file)
    data_cfg = cfg["data"]
    dataset_name = data_cfg["dataset"]
    model_name = cfg["model"]
    accelerator = "cpu"
    test_dataset = data_cfg["test_dataset"]
    str_logs = [dataset_name]
    if test_dataset:
        str_logs += [test_dataset]
    str_logs += [model_name]

    path_ds_target = data_cfg["data_dir"]

    new_subdir = None
    logger.info("Training %s", "-".join(str_logs))

    # prepare kfold-cv
    all_trajectories = DatasetFromPath.get_data(
        path_ds_target, "train" if dataset_name == "synthetic" else test_dataset
    )
    trajectories_tracklets = all_trajectories.groupby(
        "ag_id" if dataset_name == "synthetic" else ["dataset_name", "tracklet_id"]
    )
    input_trajectories = [tracklet for _, tracklet in trajectories_tracklets]
    random.seed(K_FOLD_SEED)
    random.shuffle(input_trajectories)

    dummy_x = np.random.randn(len(input_trajectories), 1)
    kf = KFold(n_splits=k_folds)
    new_subdir = f"{new_subdir}/fold" if new_subdir else "fold"

    ray.init()
    trainers = [
        get_trainer_objects_kfold_cv(
            input_trajectories,
            train_index,
            validation_index,
            cfg,
            accelerator,
            f"{new_subdir}_{i}",
        )
        for i, (train_index, validation_index) in enumerate(kf.split(dummy_x))
    ]
    logging_paths = ray.get(
        [ray_run_trainer.remote(tr_options) for tr_options in trainers]
    )

    save_path = os.path.join(os.path.dirname(logging_paths[0]), "n_runs_metrics.json")
    overall_metrics = load_json_file(
        os.path.join(logging_paths[0], "test_metrics.json")
    )
    for logging_path in logging_paths[1:]:
        test_metrics = load_json_file(os.path.join(logging_path, "test_metrics.json"))
        overall_metrics = merge_dicts(overall_metrics, test_metrics)
    dump_json_file(overall_metrics, save_path)
    logger.info("==================Overall Results================")
    logger.info(overall_metrics)

    metrics_df = {}
    for k, v in overall_metrics.items():
        if k != "labels_mapping":
            avg_metric, std_metric = np.mean(v), np.std(v)
            logger.info("%s %1.2f +- %1.2f", k, avg_metric, std_metric)
            metrics_df[k] = f"{avg_metric:1.2f}+-{std_metric:1.2f}"

    metrics = pd.DataFrame.from_dict(metrics_df, orient="index")
    metrics.to_csv(
        os.path.join(os.path.dirname(logging_paths[0]), "cross_val_overall_results.csv")
    )


if __name__ == "__main__":
    LOGO_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGO_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
