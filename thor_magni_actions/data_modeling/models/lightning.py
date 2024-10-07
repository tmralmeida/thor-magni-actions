import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

from thor_magni_actions.data_modeling.datasets import (
    build_features_scaler_from_stats,
)
from thor_magni_actions.data_modeling.evaluation.common_metrics import (
    AverageDisplacementError,
    FinalDisplacementError,
)
from thor_magni_actions.io import dump_json_file
from thor_magni_actions.data_modeling.datasets.scalers import TrajectoriesNormalizer
from .transformer import (
    TransformerEncMLP,
    AgentSemanticCondTransformerEncMLP,
    MultiTaskAgentSemanticCondTransformer,
)
from .lstm import RecurrentNetwork, AgentSemanticCondRNN, MultiTaskAgentSemanticCondRNN


class LightDiscriminativePredictor(pl.LightningModule):
    """LSTM, GRU, and Transformers lightning modules"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        model_name = kwargs["model_name"]
        data_cfg = kwargs["data_cfg"]
        network_cfg = kwargs["network_cfg"]
        hyperparameters_cfg = kwargs["hyperparameters_cfg"]
        visual_feature_cfg = kwargs["visual_feature_cfg"]
        features_scalers_stats = kwargs["features_scalers_stats"]
        saved_hyperparams = dict(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=visual_feature_cfg,
            features_scalers_stats=features_scalers_stats,
        )
        self.save_hyperparameters(saved_hyperparams)
        features_in = data_cfg["features_in"]
        if model_name == "tf":
            self.model = TransformerEncMLP(
                cfg=network_cfg,
                input_type=features_in,
            )
        elif model_name == "cond_tf":
            self.model = AgentSemanticCondTransformerEncMLP(
                cfg=network_cfg,
                input_type=features_in,
            )
        elif model_name == "rnn":
            self.model = RecurrentNetwork(cfg=network_cfg, input_type=features_in)
        elif model_name == "cond_rnn":
            self.model = AgentSemanticCondRNN(cfg=network_cfg, input_type=features_in)
        else:
            raise NotImplementedError(model_name)
        self.hyperparameters_cfg = hyperparameters_cfg
        loss_type = self.hyperparameters_cfg["loss_type"]
        if loss_type == "mse":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(loss_type)
        self.output_type = data_cfg["features_out"]
        if not self.output_type.startswith("translated"):
            self.scaler = build_features_scaler_from_stats(
                features_scalers_stats=features_scalers_stats,
                output_type=self.output_type,
            )
        self.metrics_per_class = (
            {} if data_cfg["dataset"] in ["synthetic", "thor_magni"] else None
        )
        if self.metrics_per_class is not None:
            self.sup_labels_mapping = data_cfg["supervised_labels"]
            self.n_sup_labels = max(self.sup_labels_mapping.values()) + 1
        self.observation_len = data_cfg["observation_len"]

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=float(self.hyperparameters_cfg["lr"]),
            weight_decay=1e-4,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.hyperparameters_cfg["scheduler_patience"], min_lr=1e-6
        )
        return [opt], [
            dict(scheduler=lr_scheduler, interval="epoch", monitor="train_loss")
        ]

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(train_batch)
        loss = self.loss(y_hat_unscaled, y_gt).mean()
        log_dict = dict(train_loss=loss)
        self.log_dict(log_dict, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(val_batch)
        self.update_metrics(y_hat_unscaled, y_gt)
        val_loss = self.loss(y_hat_unscaled, y_gt).mean()
        log_dict = dict(val_loss=val_loss)
        self.log_dict(log_dict, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch: dict, batch_idx: int) -> torch.Tensor:
        y_gt, y_hat_unscaled = self.common_step(test_batch)
        self.update_metrics(y_hat_unscaled, y_gt)
        if self.metrics_per_class is not None:
            self.update_metrics_per_class(
                y_hat_unscaled, y_gt, test_batch["agent_type"]
            )

    def on_validation_start(self) -> None:
        self.eval_metrics = dict(
            ade=AverageDisplacementError().to(self.device),
            fde=FinalDisplacementError().to(self.device),
        )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()

    def on_test_start(self) -> None:
        self.eval_metrics = dict(
            ade=AverageDisplacementError().to(self.device),
            fde=FinalDisplacementError().to(self.device),
        )
        if self.metrics_per_class is not None:
            for i in range(self.n_sup_labels):
                self.metrics_per_class[f"ADE_c{i}"] = AverageDisplacementError().to(
                    self.device
                )
                self.metrics_per_class[f"FDE_c{i}"] = FinalDisplacementError().to(
                    self.device
                )

    def on_test_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "test_metrics.json")
        test_metrics = self.compute_metrics()
        # if self.metrics_per_class is not None:
        #     test_metrics.update(labels_mapping=self.sup_labels_mapping)
        dump_json_file(test_metrics, save_path)
        self.reset_metrics()

    def predict_step(self, predict_batch: dict, batch_idx: int) -> torch.Tensor:
        _, y_hat_unscaled = self.common_step(predict_batch)
        return dict(
            gt=predict_batch["gt_pred"].detach(),
            y_hat=[y_hat_unscaled.detach()],
        )

    def get_unscaled_prediction(
        self, scaled_prediction: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        gt_observation = batch["gt_obs"]
        if self.output_type in ["translated_origin", "translated_rotated"]:
            y_hat_unscaled = TrajectoriesNormalizer.denormalize(
                raw_dataset=gt_observation,
                norm_dataset=scaled_prediction,
                ori=True,
                rot=self.output_type == "translated_rotated",
                sca=False,
            )
        else:
            predictions = torch.cat(
                [torch.zeros_like(gt_observation), scaled_prediction], dim=1
            )
            y_hat_unscaled = self.scaler.descale(predictions)[
                :, self.observation_len:, :
            ]
            if self.output_type == "velocities":
                delta_time = batch["delta_time"].view(-1, 1, 1)
                y_hat_unscaled = y_hat_unscaled * delta_time
                last_observed_points = gt_observation[:, -1, :]
                y_hat_unscaled[:, 0, :] += last_observed_points
                y_hat_unscaled = torch.cumsum(y_hat_unscaled, dim=1)
        return y_hat_unscaled

    def common_step(self, batch: dict, **kwargs):
        y_gt = batch["gt_pred"]
        y_pred = self(batch)
        y_hat = y_pred.clone()
        y_hat_unscaled = self.get_unscaled_prediction(
            scaled_prediction=y_hat, batch=batch
        )
        return y_gt, y_hat_unscaled

    def update_metrics(self, y_hat: torch.Tensor, y_gt: torch.Tensor):
        for _, metric in self.eval_metrics.items():
            metric.update(preds=y_hat, target=y_gt)

    def update_metrics_per_class(
        self, y_hat: torch.Tensor, y_gt: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        for i, label in enumerate(labels):
            cl_idx = int(label[0].item() if label.size(0) != 1 else label.item())
            self.metrics_per_class[f"ADE_c{cl_idx}"].update(
                preds=y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"FDE_c{cl_idx}"].update(
                preds=y_hat[i].unsqueeze(dim=0), target=y_gt[i].unsqueeze(dim=0)
            )

    def compute_metrics(self) -> dict:
        final_metrics = {
            met_name: met.compute().item()
            for met_name, met in self.eval_metrics.items()
        }
        if self.metrics_per_class is not None:
            final_labels_metrics = {
                met_name: met.compute().item()
                for met_name, met in self.metrics_per_class.items()
            }
            final_metrics.update(final_labels_metrics)
        return final_metrics

    def reset_metrics(self) -> None:
        for _, metric in self.eval_metrics.items():
            metric.reset()
        if self.metrics_per_class is not None:
            for _, metric in self.metrics_per_class.items():
                metric.reset()


class LightMultiTaskPredictor(pl.LightningModule):
    """LSTM, GRU, and Transformers lightning modules for trajectory prediction and action
    labels prediction"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        model_name = kwargs["model_name"]
        data_cfg = kwargs["data_cfg"]
        network_cfg = kwargs["network_cfg"]
        hyperparameters_cfg = kwargs["hyperparameters_cfg"]
        visual_feature_cfg = kwargs["visual_feature_cfg"]
        features_scalers_stats = kwargs["features_scalers_stats"]
        saved_hyperparams = dict(
            model_name=model_name,
            data_cfg=data_cfg,
            network_cfg=network_cfg,
            hyperparameters_cfg=hyperparameters_cfg,
            visual_feature_cfg=visual_feature_cfg,
            features_scalers_stats=features_scalers_stats,
        )
        self.save_hyperparameters(saved_hyperparams)
        features_in = data_cfg["features_in"]
        if model_name == "mtl_tf":
            self.model = MultiTaskAgentSemanticCondTransformer(
                cfg=network_cfg, input_type=features_in
            )
        elif model_name == "mtl_rnn":
            self.model = MultiTaskAgentSemanticCondRNN(
                cfg=network_cfg, input_type=features_in
            )
        else:
            raise NotImplementedError(model_name)
        self.hyperparameters_cfg = hyperparameters_cfg
        prediction_loss_type = self.hyperparameters_cfg["loss_type"]
        if prediction_loss_type == "mse":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError(prediction_loss_type)
        self.act_loss = nn.CrossEntropyLoss()
        self.weight_act_loss = hyperparameters_cfg["weight_act_loss"]
        self.prediction_output_type = data_cfg["features_out"]
        if not self.prediction_output_type.startswith("translated"):
            self.scaler = build_features_scaler_from_stats(
                features_scalers_stats=features_scalers_stats,
                output_type=self.prediction_output_type,
            )
        self.metrics_per_class = (
            {} if data_cfg["dataset"] in ["synthetic", "thor_magni"] else None
        )
        if self.metrics_per_class is not None:
            self.agents_mapping = data_cfg["supervised_labels"]
            self.actions_mapping = data_cfg["actions"]
            self.n_agents = max(self.agents_mapping.values()) + 1
            self.n_actions = max(self.actions_mapping.values()) + 1
        self.observation_len = data_cfg["observation_len"]

    def forward(self, x):
        traj_pred, act_pred = self.model(x)
        return traj_pred, act_pred

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=float(self.hyperparameters_cfg["lr"]),
            weight_decay=1e-4,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.hyperparameters_cfg["scheduler_patience"], min_lr=1e-6
        )
        return [opt], [
            dict(scheduler=lr_scheduler, interval="epoch", monitor="train_loss")
        ]

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.common_step(train_batch)
        traj_pred_gt, traj_pred_unscaled = outputs[:2]
        act_pred_gt, act_pred = outputs[2:]
        traj_loss = self.loss(traj_pred_unscaled, traj_pred_gt).mean()
        act_loss = self.act_loss(
            act_pred.contiguous().view(-1, act_pred.size(-1)),
            act_pred_gt.contiguous().view(-1),
        ).mean()
        loss = traj_loss + self.weight_act_loss * act_loss
        log_dict = {"traj_loss": traj_loss, "act_loss": act_loss, "train_loss": loss}
        self.log_dict(log_dict, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.common_step(val_batch)
        traj_pred_gt, traj_pred_unscaled = outputs[:2]
        act_pred_gt, act_pred = outputs[2:]
        val_traj_loss = self.loss(traj_pred_unscaled, traj_pred_gt).mean()
        val_act_loss = self.act_loss(
            act_pred.contiguous().view(-1, act_pred.size(-1)),
            act_pred_gt.contiguous().view(-1),
        ).mean()
        val_loss = val_traj_loss + self.weight_act_loss * val_act_loss
        act_pred = torch.nn.functional.softmax(act_pred, dim=-1)
        self.update_metrics(traj_pred_unscaled, traj_pred_gt, act_pred, act_pred_gt)

        log_dict = {
            "val_traj_loss": val_traj_loss,
            "val_act_loss": val_act_loss,
            "val_loss": val_loss,
        }
        log_dict = dict(val_loss=val_loss)
        self.log_dict(log_dict, on_epoch=True, prog_bar=True)

    def test_step(self, test_batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.common_step(test_batch)
        traj_pred_gt, traj_pred_unscaled = outputs[:2]
        act_pred_gt, act_pred = outputs[2:]
        act_pred = torch.nn.functional.softmax(act_pred, dim=-1)

        self.update_metrics(traj_pred_unscaled, traj_pred_gt, act_pred, act_pred_gt)
        if self.metrics_per_class is not None:
            self.update_metrics_per_class(
                traj_pred=traj_pred_unscaled,
                traj_gt=traj_pred_gt,
                act_pred=act_pred,
                act_gt=act_pred_gt,
                agent_type=test_batch["agent_type"],
            )

    def on_validation_start(self) -> None:
        self.eval_metrics = dict(
            accuracy=Accuracy(
                task="multiclass",
                num_classes=self.n_actions,
                average="micro",
            ),
            f1_score=F1Score(
                task="multiclass",
                num_classes=self.n_actions,
                average="weighted",
            ),
            ade=AverageDisplacementError().to(self.device),
            fde=FinalDisplacementError().to(self.device),
        )

    def on_validation_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "val_metrics.json")
        val_metrics = self.compute_metrics()
        dump_json_file(val_metrics, save_path)
        self.reset_metrics()

    def on_test_start(self) -> None:
        self.eval_metrics = dict(
            accuracy=Accuracy(
                task="multiclass",
                num_classes=self.n_actions,
                average="micro",
            ),
            f1_score=F1Score(
                task="multiclass",
                num_classes=self.n_actions,
                average="weighted",
            ),
            ade=AverageDisplacementError().to(self.device),
            fde=FinalDisplacementError().to(self.device),
        )
        for i in range(self.n_actions):
            self.metrics_per_class[f"accuracy_c{i}"] = Accuracy(
                task="multiclass",
                num_classes=self.n_actions,
                average="micro",
            ).to(self.device)
            self.metrics_per_class[f"f1_score_c{i}"] = F1Score(
                task="multiclass",
                num_classes=self.n_actions,
                average="weighted",
            ).to(self.device)
        for i in range(self.n_agents):
            self.metrics_per_class[f"ADE_c{i}"] = AverageDisplacementError().to(
                self.device
            )
            self.metrics_per_class[f"FDE_c{i}"] = FinalDisplacementError().to(
                self.device
            )

    def on_test_end(self) -> None:
        save_path = os.path.join(self.logger.log_dir, "test_metrics.json")
        eval_metrics = self.compute_metrics()
        if hasattr(self, "agents_mapping"):
            eval_metrics.update(self.agents_mapping)
        if hasattr(self, "actions_mapping"):
            eval_metrics.update(self.actions_mapping)
        dump_json_file(eval_metrics, save_path)
        self.reset_metrics()

    def predict_step(self, predict_batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.common_step(predict_batch)
        traj_pred_gt, traj_pred_unscaled = outputs[:2]
        act_pred_gt, act_pred = outputs[2:]
        act_pred = torch.nn.functional.softmax(act_pred, dim=-1)
        return dict(
            traj_pred_gt=traj_pred_gt.detach(),
            traj_pred_unscaled=[traj_pred_unscaled.detach()],
            act_pred_gt=act_pred_gt,
            act_pred=act_pred,
        )

    def update_metrics(
        self,
        traj_pred: torch.Tensor,
        traj_gt: torch.Tensor,
        act_pred: torch.Tensor,
        act_gt: torch.Tensor,
    ):
        act_pred_v = act_pred.contiguous().view(-1, act_pred.size(-1))
        act_gt_v = act_gt.contiguous().view(-1)
        self.eval_metrics["accuracy"].update(preds=act_pred_v, target=act_gt_v)
        self.eval_metrics["f1_score"].update(preds=act_pred_v, target=act_gt_v)
        self.eval_metrics["ade"].update(preds=traj_pred, target=traj_gt)
        self.eval_metrics["fde"].update(preds=traj_pred, target=traj_gt)

    def update_metrics_per_class(
        self,
        traj_pred: torch.Tensor,
        traj_gt: torch.Tensor,
        act_pred: torch.Tensor,
        act_gt: torch.Tensor,
        agent_type: torch.Tensor,
    ) -> torch.Tensor:
        for i, label in enumerate(agent_type.squeeze(dim=1)):
            cl_idx = int(label[0].item() if label.size(0) != 1 else label.item())
            self.metrics_per_class[f"ADE_c{cl_idx}"].update(
                preds=traj_pred[i].unsqueeze(dim=0), target=traj_gt[i].unsqueeze(dim=0)
            )
            self.metrics_per_class[f"FDE_c{cl_idx}"].update(
                preds=traj_pred[i].unsqueeze(dim=0), target=traj_gt[i].unsqueeze(dim=0)
            )
        for i, action_lbl in enumerate(act_gt):
            act_pred_v = act_pred[i]  # ts, n_acts
            seq_len = action_lbl.shape[0]  # ts
            for ts in range(seq_len):
                cl_idx = action_lbl[ts]
                self.metrics_per_class[f"accuracy_c{cl_idx.item()}"].update(
                    preds=act_pred_v[ts].unsqueeze(dim=0), target=cl_idx.unsqueeze(dim=0)
                )
                self.metrics_per_class[f"f1_score_c{cl_idx.item()}"].update(
                    preds=act_pred_v[ts].unsqueeze(dim=0), target=cl_idx.unsqueeze(dim=0)
                )

    def compute_metrics(self) -> dict:
        final_metrics = {
            met_name: met.compute().item()
            for met_name, met in self.eval_metrics.items()
        }
        final_act_metrics = {
            met_name: met.compute().item()
            for met_name, met in self.metrics_per_class.items()
        }
        final_metrics.update(final_act_metrics)
        return final_metrics

    def reset_metrics(self) -> None:
        for _, metric in self.eval_metrics.items():
            metric.reset()
        for _, metric in self.metrics_per_class.items():
            metric.reset()

    def get_unscaled_prediction(
        self, scaled_prediction: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        gt_observation = batch["gt_obs"]
        if self.prediction_output_type in ["translated_origin", "translated_rotated"]:
            y_hat_unscaled = TrajectoriesNormalizer.denormalize(
                raw_dataset=gt_observation,
                norm_dataset=scaled_prediction,
                ori=True,
                rot=self.prediction_output_type == "translated_rotated",
                sca=False,
            )
        else:
            predictions = torch.cat(
                [torch.zeros_like(gt_observation), scaled_prediction], dim=1
            )
            y_hat_unscaled = self.scaler.descale(predictions)[
                :, self.observation_len:, :
            ]
            if self.prediction_output_type == "velocities":
                delta_time = batch["delta_time"].contiguous().view(-1, 1, 1)
                y_hat_unscaled = y_hat_unscaled * delta_time
                last_observed_points = gt_observation[:, -1, :]
                y_hat_unscaled[:, 0, :] += last_observed_points
                y_hat_unscaled = torch.cumsum(y_hat_unscaled, dim=1)
        return y_hat_unscaled

    def common_step(self, batch: dict, **kwargs):
        traj_pred_gt = batch["gt_pred"]
        act_pred_gt = batch["action"][:, self.observation_len:].long()
        traj_pred, act_pred = self(batch)
        traj_pred = traj_pred.clone()
        traj_pred_unscaled = self.get_unscaled_prediction(
            scaled_prediction=traj_pred, batch=batch
        )
        return traj_pred_gt, traj_pred_unscaled, act_pred_gt, act_pred
