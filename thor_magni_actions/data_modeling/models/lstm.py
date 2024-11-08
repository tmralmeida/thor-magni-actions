from typing import Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Encoder, LatentEmbedding, cat_class_emb


class RecurrentNetwork(nn.Module):
    """Recurrent networks based models: GRU LSTM. Similar to RED predictor but with visual feature
    encoder if that's available"""

    def __init__(
        self,
        cfg: dict,
        input_type: Union[str, List[str]],
    ) -> None:
        super().__init__()
        self.hid_dim = cfg["hidden_dim"]
        self.emb_dim = cfg["embedding_dim"]
        self.network_type = cfg["type"]
        self.hidden_state = True if cfg["state"] == "hidden" else False
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.input_dims = sum([2 for _ in self.input_type])
        self.encoder = Encoder(
            inp_dim=self.input_dims,
            emb_dim=self.emb_dim,
            hid_dim=self.hid_dim,
            class_emb_dim=0,
            network_type=self.network_type,
        )
        intermediate_size = (self.hid_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(
                self.hid_dim,
                intermediate_size,
            ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(
                intermediate_size,
                2 * cfg["prediction_len"],
            ),
        )

    def forward(self, x: dict) -> torch.Tensor:
        inputs_cat = torch.cat(
            [_in["scl_obs"] for _in in x["features"].values()], dim=-1
        )

        bs = inputs_cat.size(0)
        hidden_cell_init = (
            (
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
            )
            if self.network_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim)
        )

        # encoder
        inp_lstm = self.encoder.input_embedding(
            inputs_cat.contiguous().view(-1, self.input_dims)
        )
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, hidden_cell = self.encoder.temp_feat(inp_lstm, hidden_cell_init)
        hn = (
            hidden_cell[0 if self.hidden_state else 1]
            if self.network_type == "lstm"
            else hidden_cell
        )
        hn = hn.view(-1, self.hid_dim)
        out = self.decoder(hn)
        out = out.view(bs, -1, 2)
        return out


class AgentSemanticCondRNN(nn.Module):
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
        self.hid_dim = cfg["hidden_dim"]
        self.emb_dim = cfg["embedding_dim"]
        self.network_type = cfg["type"]
        self.hidden_state = True if cfg["state"] == "hidden" else False
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.input_dims = sum([2 for _ in self.input_type])
        if self.act_emb_layer:
            self.input_dims += cfg_cond_act["embedding_dim"]
        self.encoder = Encoder(
            inp_dim=self.input_dims,
            emb_dim=self.emb_dim,
            hid_dim=self.hid_dim,
            class_emb_dim=0,
            network_type=self.network_type,
        )

        intermediate_dim = (
            cfg_cond_agent["embedding_dim"] if self.agent_emb_layer else 0
        )

        intermediate_size = (self.hid_dim + intermediate_dim) // 2
        self.decoder = nn.Sequential(
            nn.Linear(
                (self.hid_dim + intermediate_dim),
                intermediate_size,
            ),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(
                intermediate_size,
                2 * cfg["prediction_len"],
            ),
        )

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

    def forward(self, x: dict):
        inputs = torch.cat([_in["scl_obs"] for _in in x["features"].values()], dim=-1)
        if self.act_emb_layer:
            inputs = self.forward_actions_embeddings(x, inputs)

        if self.agent_classes:
            agent_labels = self.forward_agent_embeddings(x)

        bs = inputs.size(0)
        hidden_cell_init = (
            (
                (torch.zeros(1, bs, self.hid_dim)).to(inputs),
                (torch.zeros(1, bs, self.hid_dim)).to(inputs),
            )
            if self.network_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim)
        )

        # encoder
        inp_lstm = self.encoder.input_embedding(
            inputs.contiguous().view(-1, self.input_dims)
        )
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, hidden_cell = self.encoder.temp_feat(inp_lstm, hidden_cell_init)
        hn = (
            hidden_cell[0 if self.hidden_state else 1]
            if self.network_type == "lstm"
            else hidden_cell
        )
        hn = hn.view(-1, self.hid_dim)
        if self.agent_emb_layer:
            hn = cat_class_emb(hn, agent_labels)
        # decoder
        out = self.decoder(hn)
        out = out.view(bs, -1, 2)
        return out


class MultiTaskAgentSemanticCondRNN(AgentSemanticCondRNN):
    def __init__(self, cfg: dict, input_type: str | List[str]) -> None:
        super().__init__(cfg, input_type)
        cfg_cond_agent = cfg["conditions"]["agent_type"]
        intermediate_dim = (
            cfg_cond_agent["embedding_dim"] if self.agent_emb_layer else 0
        )

        intermediate_size = (self.hid_dim + intermediate_dim) // 2
        self.decoder_actions = nn.Sequential(
            nn.Linear(
                (
                    self.hid_dim + intermediate_dim
                    if self.agent_emb_layer
                    else self.hid_dim
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

    def forward(self, x: dict):
        inputs = torch.cat([_in["scl_obs"] for _in in x["features"].values()], dim=-1)
        if self.act_emb_layer:
            inputs = self.forward_actions_embeddings(x, inputs)

        if self.agent_classes:
            agent_labels = self.forward_agent_embeddings(x)

        bs = inputs.size(0)
        hidden_cell_init = (
            (
                (torch.zeros(1, bs, self.hid_dim)).to(inputs),
                (torch.zeros(1, bs, self.hid_dim)).to(inputs),
            )
            if self.network_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim)
        )

        # encoder
        inp_lstm = self.encoder.input_embedding(
            inputs.contiguous().view(-1, self.input_dims)
        )
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, hidden_cell = self.encoder.temp_feat(inp_lstm, hidden_cell_init)
        hn = (
            hidden_cell[0 if self.hidden_state else 1]
            if self.network_type == "lstm"
            else hidden_cell
        )
        hn = hn.view(-1, self.hid_dim)
        if self.agent_emb_layer:
            hn = cat_class_emb(hn, agent_labels)
        # decoder
        traj_pred = self.decoder(hn)
        traj_pred = traj_pred.view(bs, -1, 2)
        act_pred = self.decoder_actions(hn)
        act_pred = act_pred.view(bs, -1, self.act_classes)
        return traj_pred, act_pred
