import os
import pandas as pd
from copy import deepcopy
from typing import Optional, List
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from thor_magni_actions.data_modeling.models import (
    LightDiscriminativePredictor,
    LightMultiTaskPredictor,
    LightBaseActPredictor,
)
from thor_magni_actions.data_modeling.datasets import (
    DatasetObjects,
    get_scalers_stats,
)
from thor_magni_actions.io import load_yaml_file


def get_trainer_objects_kfold_cv(
    all_trajectories: pd.DataFrame,
    train_index: List[int],
    validation_index: List[int],
    cfg: dict,
    accelerator: str = "cpu",
    subdir: Optional[str] = None,
):
    """used for k-fold cross validation"""
    model_name = cfg["model"]
    data_cfg = cfg["data"]
    network_cfg = cfg["network"]
    hyperparameters_cfg = cfg["hyperparameters"]
    save_cfg = cfg["save"]
    dataset_name = data_cfg["dataset"]
    test_dataset = data_cfg["test_dataset"]
    save_path = (
        os.path.join(save_cfg["path"], dataset_name, test_dataset)
        if test_dataset
        else os.path.join(save_cfg["path"], dataset_name)
    )
    visual_feature_extractor_cfg = cfg["visual_feature_extractor"]
    vis_features_cfg = None
    if visual_feature_extractor_cfg["use"]:
        vis_features_cfg = load_yaml_file(visual_feature_extractor_cfg["inherit_from"])
        visuals_path = vis_features_cfg["data_dir"]
        visual_window_size = vis_features_cfg["window_size"]
        data_cfg.update(dict(visuals_path=visuals_path, window_size=visual_window_size))

    train_trajectories = pd.concat(
        [all_trajectories[train_i] for train_i in train_index]
    )
    validation_trajectories = pd.concat(
        [all_trajectories[val_i] for val_i in validation_index]
    )
    train_objects = DatasetObjects(
        set_type="train",
        data_cfg=data_cfg,
        dataset_dataframe=train_trajectories,
    )
    val_objects = DatasetObjects(
        set_type="val",
        data_cfg=data_cfg,
        dataset_dataframe=validation_trajectories,
    )
    train_ds = train_objects.load_dataset()
    val_objects.features_scaler = deepcopy(train_objects.features_scaler)

    val_ds = val_objects.load_dataset()

    features_scalers_stats = get_scalers_stats(
        train_features_scaler=train_objects.features_scaler,
        val_features_scaler=val_objects.features_scaler,
    )
    if data_cfg["dataset"] in ["thor_magni", "synthetic"]:
        mapping_agent_types = {
            **val_ds.mapping_agent_types,
            **train_ds.mapping_agent_types,
        }
        mapping_actions = {
            **val_ds.mapping_actions,
            **train_ds.mapping_actions,
        }
        train_ds.mapping_agent_types = mapping_agent_types
        val_ds.mapping_agent_types = mapping_agent_types
        train_ds.mapping_actions = mapping_actions
        val_ds.mapping_actions = mapping_actions
        data_cfg.update(dict(supervised_labels=train_ds.mapping_agent_types))
        data_cfg.update(dict(actions=train_ds.mapping_actions))

    train_dl = DataLoader(train_ds, batch_size=hyperparameters_cfg["bs"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=hyperparameters_cfg["bs"], shuffle=False)

    subdir_model_name = model_name
    if subdir_model_name.startswith(("cond", "mtl", "base_act_pred")):
        subdir_model_name = (
            subdir_model_name
            + ",agent_"
            + str(network_cfg["conditions"]["agent_type"]["use"])
            + ",act_"
            + str(network_cfg["conditions"]["action"]["use"])
        )
        if subdir_model_name.startswith("base_act_pred") and data_cfg["features_in"]:
            subdir_model_name = subdir_model_name + ",motion_cues"
    subdir = os.path.join(subdir_model_name, subdir) if subdir else subdir_model_name
    logger = TensorBoardLogger(save_path, name=subdir, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, filename="{epoch}-{val_loss:.2f}"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=hyperparameters_cfg["patience"],
        verbose=False,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]
    if model_name in ["rnn", "cond_rnn", "tf", "cond_tf"]:
        lightning_module = LightDiscriminativePredictor(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=visual_feature_extractor_cfg,
            features_scalers_stats=features_scalers_stats,
        )
    elif model_name.startswith("mtl"):
        lightning_module = LightMultiTaskPredictor(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=visual_feature_extractor_cfg,
            features_scalers_stats=features_scalers_stats,
        )
    elif model_name.startswith("base_act_pred"):
        lightning_module = LightBaseActPredictor(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=visual_feature_extractor_cfg,
            features_scalers_stats=features_scalers_stats,
        )
    else:
        raise NotImplementedError(cfg["model"])
    trainer = pl.Trainer(
        default_root_dir=save_cfg["path"],
        logger=logger,
        accelerator="cpu",
        callbacks=callbacks,
        max_epochs=hyperparameters_cfg["max_epochs"],
        check_val_every_n_epoch=hyperparameters_cfg["val_freq"],
    )
    return trainer, lightning_module, train_dl, val_dl, val_dl


def run_trainer(trainer_options):
    trainer, lightning_module, train_dl, val_dl, test_dl = trainer_options
    trainer.fit(lightning_module, train_dl, val_dl)
    if test_dl:
        trainer.test(ckpt_path="best", dataloaders=test_dl)
    return trainer.logger.log_dir
