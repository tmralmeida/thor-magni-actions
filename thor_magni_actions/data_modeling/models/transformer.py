from typing import Union, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from .tf_modules import PositionalEncoding, TransformerEncoder
from .modules import LatentEmbedding, cat_class_emb


class TransformerEncMLP(nn.Module):
    def __init__(
        self,
        cfg: dict,
        input_type: Union[str, List[str]],
    ) -> None:
        super().__init__()
        self.d_model = cfg["d_model"]
        input_type = input_type if isinstance(input_type, list) else [input_type]
        input_dims = sum([2 for _ in input_type])
        self.emb_net = nn.Sequential(
            nn.Linear(input_dims, self.d_model), nn.Dropout(cfg["dropout"])
        )
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = TransformerEncoder(
            num_layers=cfg["num_layers"],
            input_dim=self.d_model,
            dim_feedforward=2 * self.d_model,
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )
        intermediate_size = self.d_model // 2
        self.decoder = nn.Sequential(
            nn.Linear(
                self.d_model,
                intermediate_size,
            ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(
                intermediate_size,
                2 * cfg["prediction_len"],
            ),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x: dict, mask: Optional[torch.Tensor] = None):
        inputs = torch.cat([_in["scl_obs"] for _in in x["features"].values()], dim=-1)
        bs = inputs.size(0)
        x = self.emb_net(inputs)
        x = self.positional_encoding(x)
        features = self.transformer_encoder(x, mask=mask)
        features = torch.mean(features, dim=1)
        out = self.decoder(features)
        out = out.view(bs, -1, 2)
        return out


class AgentSemanticCondTransformerEncMLP(nn.Module):
    def __init__(self, cfg: dict, input_type: str | List[str]) -> None:
        super().__init__()
        cfg_cond_agent = cfg["conditions"]["agent_type"]
        cfg_cond_act = cfg["conditions"]["action"]
        self.agent_classes, self.agent_cond_type, self.agent_emb_layer = (
            self._build_condition_info(cfg_cond_agent)
        )

        self.act_classes, self.act_cond_type, self.act_emb_layer = (
            self._build_condition_info(cfg_cond_act)
        )

        self.d_model = cfg["d_model"]
        input_type = input_type if isinstance(input_type, list) else [input_type]
        input_dims = sum([2 for _ in input_type])
        if self.act_emb_layer:
            input_dims += cfg_cond_act["embedding_dim"]
        self.emb_net = nn.Sequential(
            nn.Linear(input_dims, self.d_model), nn.Dropout(cfg["dropout"])
        )
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)
        self.transformer_encoder = TransformerEncoder(
            num_layers=cfg["num_layers"],
            input_dim=self.d_model,
            dim_feedforward=2 * self.d_model,
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )
        intermediate_dim = (
            cfg_cond_agent["embedding_dim"] if self.agent_emb_layer else 0
        )

        intermediate_size = (self.d_model + intermediate_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(
                (
                    self.d_model + intermediate_dim
                    if self.agent_emb_layer
                    else self.d_model
                ),
                intermediate_size,
            ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(
                intermediate_size,
                2 * cfg["prediction_len"],
            ),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _build_condition_info(self, cfg_cond: dict) -> Tuple[int, str, nn.Module]:
        n_classes, cond_type, emb_layer = cfg_cond["n_labels"], None, None
        if cfg_cond["use"]:
            cond_type = cfg_cond["name"]
            if cond_type not in ["embedding", "one_hot"]:
                raise NotImplementedError(cond_type)
            class_emb_dim = (
                cfg_cond["embedding_dim"] if cond_type == "embedding" else n_classes
            )
            emb_layer = (
                LatentEmbedding(n_classes, class_emb_dim)
                if cond_type == "embedding"
                else None
            )
        return n_classes, cond_type, emb_layer

    def forward_agent_embeddings(self, x: dict) -> torch.Tensor:
        agent_labels = x["agent_type"][:, 0].long()
        agent_labels = (
            self.agent_emb_layer(agent_labels)
            if self.agent_cond_type == "embedding"
            else F.one_hot(agent_labels, num_classes=self.agent_classes).float()
        )
        return agent_labels

    def forward_actions_embeddings(
        self, x: dict, inputs: torch.Tensor
    ) -> torch.Tensor:
        actions_labels = x["action"]
        seq_len = inputs.shape[1]
        temporal_embeddings = []
        for ts in range(seq_len):
            ts_act = actions_labels[:, ts].long()
            temporal_embeddings += [
                (
                    self.act_emb_layer(ts_act)
                    if self.act_cond_type == "embedding"
                    else F.one_hot(ts_act, num_classes=self.act_classes).float()
                )
            ]
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)
        input_cat = torch.cat([inputs, temporal_embeddings], dim=-1)
        return input_cat

    def forward(self, x: dict, mask: Optional[torch.Tensor] = None):
        inputs = torch.cat([_in["scl_obs"] for _in in x["features"].values()], dim=-1)
        if self.act_emb_layer:
            inputs = self.forward_actions_embeddings(x, inputs)

        if self.agent_emb_layer:
            agent_labels = self.forward_agent_embeddings(x)
        bs = inputs.size(0)
        x = self.emb_net(inputs)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        hn = torch.mean(x, dim=1)
        if self.agent_emb_layer:
            hn = cat_class_emb(hn, agent_labels)
        x = self.decoder(hn)
        return x.view(bs, -1, 2)


class MultiTaskAgentSemanticCondTransformer(AgentSemanticCondTransformerEncMLP):
    def __init__(self, cfg: dict, input_type: str | List[str]) -> None:
        super().__init__(cfg, input_type)
        cfg_cond_agent = cfg["conditions"]["agent_type"]
        intermediate_dim = (
            cfg_cond_agent["embedding_dim"] if self.agent_emb_layer else 0
        )
        intermediate_size = (self.d_model + intermediate_dim) // 2
        self.decoder_actions = nn.Sequential(
            nn.Linear(
                (
                    self.d_model + intermediate_dim
                    if self.agent_emb_layer
                    else self.d_model
                ),
                intermediate_size,
            ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(
                intermediate_size,
                self.act_classes * cfg["prediction_len"],
            ),
        )
        self._reset_parameters()

    def forward(self, x: dict, mask: Optional[torch.Tensor] = None):
        inputs = torch.cat([_in["scl_obs"] for _in in x["features"].values()], dim=-1)
        if self.act_emb_layer:
            inputs = self.forward_actions_embeddings(x, inputs)

        if self.agent_emb_layer:
            agent_labels = self.forward_agent_embeddings(x)
        bs = inputs.size(0)
        x = self.emb_net(inputs)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        hn = torch.mean(x, dim=1)
        if self.agent_emb_layer:
            hn = cat_class_emb(hn, agent_labels)

        traj_pred = self.decoder(hn)
        traj_pred = traj_pred.view(bs, -1, 2)
        act_pred = self.decoder_actions(hn)
        act_pred = act_pred.view(bs, -1, self.act_classes)
        return traj_pred, act_pred
