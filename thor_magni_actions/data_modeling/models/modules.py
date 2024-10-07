from copy import copy
from collections import OrderedDict
from typing import List, Union, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from .tf_modules import (
    PositionalEncoding,
    TransformerEncoder,
)


def cat_class_emb(x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """concatenate input a class tensors -> for conditioned models"""
    if labels.dim() != x.dim():
        one_hot = labels.unsqueeze(dim=1)
        class_emb_inp = one_hot.repeat(1, x.size(1), 1)
    else:
        class_emb_inp = labels
    return torch.cat([x, class_emb_inp], dim=-1)


def make_mlp(dim_list: List[int], activation, batch_norm=False, dropout=0):
    """
    Generates MLP network:
    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)
    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        elif activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation == "prelu":
            layers.append(nn.PReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class Encoder(nn.Sequential):
    """Encoder object: input embedding + temporal features extractor"""

    def __init__(
        self,
        inp_dim: int,
        emb_dim: int,
        hid_dim: int,
        class_emb_dim: int,
        concat_start=False,
        network_type="lstm",
    ):
        emb_layers = [
            nn.Linear(inp_dim + class_emb_dim if concat_start else inp_dim, emb_dim),
            nn.PReLU(),
        ]
        input_embedding = nn.Sequential(*emb_layers)
        temporal_features_extractor = None
        if network_type:
            temporal_features_extractor = (
                nn.LSTM(
                    input_size=emb_dim + class_emb_dim if not concat_start else emb_dim,
                    hidden_size=hid_dim,
                    batch_first=True,
                )
                if network_type == "lstm"
                else nn.GRU(
                    input_size=emb_dim + class_emb_dim if not concat_start else emb_dim,
                    hidden_size=hid_dim,
                    batch_first=True,
                )
            )
        super().__init__(
            OrderedDict(
                [
                    ("input_embedding", input_embedding),
                    ("temp_feat", temporal_features_extractor),
                ]
            )
        )


class RNNEncoder(nn.Module):
    def __init__(
        self, cfg: dict, input_type: str | List[str], cfg_condition: dict | None
    ) -> None:
        super().__init__()
        self.hid_dim = cfg["hidden_dim"]
        self.emb_dim = cfg["embedding_dim"]
        self.network_type = cfg["type"]
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.input_dims = sum(
            [1 if inp == "straightness_index" else 2 for inp in self.input_type]
        )
        self.class_emb_dim = 0
        if cfg_condition is not None:
            self.cond_type = cfg_condition["name"]
            if self.cond_type not in ["embedding", "one_hot"]:
                raise NotImplementedError(self.cond_type)
            self.n_classes = cfg_condition["n_labels"]
            self.class_emb_dim = (
                cfg_condition["embedding_dim"]
                if self.cond_type == "embedding"
                else self.n_classes
            )
            self.emb_layer = (
                LatentEmbedding(self.n_classes, self.class_emb_dim)
                if self.cond_type == "embedding"
                else None
            )
        self.encoder = Encoder(
            inp_dim=self.input_dims,
            emb_dim=self.emb_dim,
            hid_dim=self.hid_dim,
            class_emb_dim=self.class_emb_dim,
            concat_start=True,
            network_type=self.network_type,
        )

    def _reset_hidden_layers(self, inputs_cat: torch.Tensor, inputs: torch.Tensor):
        bs = inputs.size(0)
        hidden_cell_init = (
            (
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
                (torch.zeros(1, bs, self.hid_dim)).to(inputs_cat),
            )
            if self.network_type == "lstm"
            else torch.zeros(1, bs, self.hid_dim).to(inputs)
        )
        return hidden_cell_init

    def forward(self, x: Union[dict, torch.Tensor], **kwargs) -> torch.Tensor:
        inputs = x
        if isinstance(x, dict):
            inputs_cat = []
            for feature_name, inputs in x.items():
                if feature_name in self.input_type:
                    inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                    inputs_cat.append(inputs)
            inputs_cat = torch.cat(inputs_cat, dim=-1)
        else:
            inputs_cat = inputs
        bs = inputs.size(0)
        hidden_cell_init = self._reset_hidden_layers(
            inputs_cat=inputs_cat, inputs=inputs
        )
        if self.class_emb_dim > 0:
            labels = (
                x["data_label"][:, 0].long()
                if isinstance(x, dict)
                else kwargs["labels"][:, 0].long()
            )
            labels = (
                self.emb_layer(labels)
                if self.cond_type == "embedding"
                else F.one_hot(labels, num_classes=self.n_classes).float()
            )
            inputs_cat = cat_class_emb(inputs_cat, labels)

        # encoder
        inp_lstm = self.encoder.input_embedding(
            inputs_cat.contiguous().view(-1, self.input_dims + self.class_emb_dim)
        )
        inp_lstm = inp_lstm.view(bs, -1, self.emb_dim)
        _, enc_hidden_cell = self.encoder.temp_feat(inp_lstm, hidden_cell_init)
        return enc_hidden_cell


class TFEncoder(nn.Module):
    def __init__(
        self,
        cfg: dict,
        input_type: str | List[str],
    ) -> None:
        super().__init__()
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.input_dims = sum(
            [1 if inp == "straightness_index" else 2 for inp in self.input_type]
        )
        d_model = cfg["d_model"]
        self.emb_net = nn.Sequential(
            nn.Linear(self.input_dims, d_model), nn.Dropout(cfg["dropout"])
        )
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.transformer_encoder = TransformerEncoder(
            num_layers=cfg["num_layers"],
            input_dim=d_model,
            dim_feedforward=2 * d_model,
            num_heads=cfg["num_heads"],
            dropout=cfg["dropout"],
        )

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x: Union[dict, torch.Tensor], **kwargs) -> torch.Tensor:
        inputs_cat = x
        if isinstance(x, dict):
            inputs_cat = []
            for feature_name, inputs in x.items():
                if feature_name in self.input_type:
                    inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                    inputs_cat.append(inputs)
            inputs_cat = torch.cat(inputs_cat, dim=-1)

        x = self.emb_net(inputs_cat)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=kwargs["mask"])
        if kwargs["get_features"]:
            return self.transformer_encoder.get_attention_maps(x, mask=kwargs["mask"])
        return x


class Decoder(nn.Sequential):
    """Decoder object: lstm + linear layer"""

    def __init__(
        self,
        inp_dim: int,
        emb_dim: int,
        hid_dim: int,
        out_dim: int,
        class_emb_dim: int,
        network_type="lstm",
    ):
        emb_layers = [
            nn.Linear(inp_dim + class_emb_dim, emb_dim),
            nn.PReLU(),
        ]
        input_embedding = nn.Sequential(*emb_layers)
        temp_feat = (
            nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, batch_first=True)
            if network_type == "lstm"
            else nn.GRU(input_size=emb_dim, hidden_size=hid_dim, batch_first=True)
        )
        predictor = nn.Linear(hid_dim, out_dim)
        super().__init__(
            OrderedDict(
                [
                    ("input_embedding", input_embedding),
                    ("temp_feat", temp_feat),
                    ("predictor", predictor),
                ]
            )
        )


class RNNDecoder(nn.Module):
    def __init__(
        self,
        cfg: dict,
        visual_feature_cfg: dict,
        input_type: str | List[str],
        prediction_len: int,
        cfg_condition: dict | None,
    ) -> None:
        super().__init__()
        self.network_type = cfg["type"]
        self.hidden_state = True if cfg["state"] == "hidden" else False
        self.hid_dim = cfg["hidden_dim"]
        self.input_type = input_type if isinstance(input_type, list) else [input_type]
        self.input_dims = sum(
            [1 if inp == "straightness_index" else 2 for inp in self.input_type]
        )
        self.prediction_len = prediction_len

        # visual feature encoder
        self.visual_out_size = 0
        if visual_feature_cfg:
            visual_feature_cfg = visual_feature_cfg["cnn_feature_extractor"]
            self.visual_feature_encoder = CNNFeatureExtractor(visual_feature_cfg)
            self.visual_out_size = visual_feature_cfg["linear_block"]["out_size"]

        self.class_emb_dim = 0
        if cfg_condition:
            self.cond_type = cfg_condition["name"]
            if self.cond_type not in ["embedding", "one_hot"]:
                raise NotImplementedError(self.cond_type)
            self.n_classes = cfg_condition["n_labels"]
            self.class_emb_dim = (
                cfg_condition["embedding_dim"]
                if self.cond_type == "embedding"
                else self.n_classes
            )
            self.emb_layer = (
                LatentEmbedding(self.n_classes, self.class_emb_dim)
                if self.cond_type == "embedding"
                else None
            )

        self.decoder = Decoder(
            inp_dim=self.input_dims + self.visual_out_size,
            emb_dim=cfg["embedding_dim"],
            hid_dim=cfg["hidden_dim"],
            out_dim=2,
            class_emb_dim=self.class_emb_dim,
            network_type=cfg["type"],
        )

    def forward(  # noqa C901
        self,
        x: dict,
        hidden_cell_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        scaler,
    ) -> torch.Tensor:
        inputs_cat = []
        for feature_name, inputs in x.items():
            if feature_name in self.input_type:
                inputs = inputs if inputs.dim() == 3 else inputs.unsqueeze(dim=-1)
                inputs_cat.append(inputs)
        inputs_cat = torch.cat(inputs_cat, dim=-1)
        bs = inputs.size(0)

        # decoder
        last_pt = inputs_cat[:, -1, :]
        last_pose = last_pt.view(bs, 1, -1)
        if self.visual_out_size > 0:
            out_features = self.visual_feature_encoder(x["img"]).unsqueeze(dim=1)
            last_pose = torch.cat([last_pose, out_features], dim=-1)
        if self.class_emb_dim > 0:
            labels = x["data_label"][:, 0].long()
            labels = (
                self.emb_layer(labels)
                if self.cond_type == "embedding"
                else F.one_hot(labels, num_classes=self.n_classes).float()
            )
            last_pose = cat_class_emb(last_pose, labels)
        decoder_input = self.decoder.input_embedding(last_pose)
        ts_poses = []
        for _ in range(self.prediction_len):
            _, hidden_cell = self.decoder.temp_feat(decoder_input, hidden_cell_state)
            hn = (
                hidden_cell[0 if self.hidden_state else 1]
                if self.network_type == "lstm"
                else hidden_cell
            )
            hn = hn.view(-1, self.hid_dim)
            decoder_out = self.decoder.predictor(hn)
            pred_pt = decoder_out.view(bs, 1, -1)
            ts_poses.append(pred_pt)
            if self.input_dims > 2:
                if "trajectories" in self.input_type:
                    add_features = scaler.inv_transform_speeds(pred_pt, x)
                elif "polars" in self.input_type:
                    unscaled_speeds = scaler.inv_scale_outputs(pred_pt, "speeds")
                    period, _ = x["period"].median(dim=1)
                    displacements = unscaled_speeds * period
                    r = torch.sqrt(
                        displacements[:, :, 0] ** 2 + displacements[:, :, 1] ** 2
                    )
                    theta = torch.atan2(displacements[:, :, 1], displacements[:, :, 0])
                    polar = torch.stack([r, theta], dim=-1)
                    add_features = scaler.scale_inputs(polar, mode="polar")
                pred_pt = torch.cat([add_features, pred_pt], dim=-1)
            if self.visual_out_size > 0:
                pred_pt = torch.cat([pred_pt, out_features], dim=-1)
            if self.class_emb_dim > 0:
                pred_pt = cat_class_emb(pred_pt, labels)
            decoder_input = self.decoder.input_embedding(pred_pt)

        return torch.cat(ts_poses, dim=1)


class MLPDecoder(nn.Module):
    def __init__(
        self,
        cfg: dict,
        visual_feature_cfg: dict,
        cfg_condition: dict | None,
        encoder_hidden_dimension: int,
        prediction_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        predictor_dims = copy(cfg["mlp_dims"])
        if prediction_len:
            predictor_dims.append(prediction_len * 2)
        # visual feature encoder
        self.visual_out_size = 0
        if visual_feature_cfg:
            visual_feature_cfg = visual_feature_cfg["cnn_feature_extractor"]
            self.visual_feature_encoder = CNNFeatureExtractor(visual_feature_cfg)
            self.visual_out_size = visual_feature_cfg["linear_block"]["out_size"]
        self.class_emb_dim = 0
        if cfg_condition is not None:
            self.cond_type = cfg_condition["name"]
            if self.cond_type not in ["embedding", "one_hot"]:
                raise NotImplementedError(self.cond_type)
            self.n_classes = cfg_condition["n_labels"]
            self.class_emb_dim = (
                cfg_condition["embedding_dim"]
                if self.cond_type == "embedding"
                else self.n_classes
            )
            self.emb_layer = (
                LatentEmbedding(self.n_classes, self.class_emb_dim)
                if self.cond_type == "embedding"
                else None
            )
        predictor_dims.insert(
            0, encoder_hidden_dimension + self.visual_out_size + self.class_emb_dim
        )

        self.decoder = make_mlp(
            dim_list=predictor_dims,
            activation=cfg["activation"],
            dropout=cfg["dropout"],
        )

    def forward(
        self,
        hidden_cell_state: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        if isinstance(hidden_cell_state, Tuple):
            hidden_cell_state = hidden_cell_state[0].squeeze(dim=0)
        hidden_cell_state = hidden_cell_state.flatten(start_dim=1)
        if self.class_emb_dim > 0:
            labels = (
                kwargs["x"]["data_label"][:, 0].long()
                if "x" in kwargs.keys()
                else kwargs["labels"][:, 0].long()
            )
            labels = (
                self.emb_layer(labels)
                if self.cond_type == "embedding"
                else F.one_hot(labels, num_classes=self.n_classes).float()
            )
            hidden_cell_state = cat_class_emb(hidden_cell_state, labels)
        out = self.decoder(hidden_cell_state)
        return out


class LatentEmbedding(nn.Module):
    """projects class embedding onto hypersphere and returns the concat of the latent and the class
    embedding"""

    def __init__(self, nlabels, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)

    def forward(self, y):
        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        return yembed


class ConvBnReLU2d(nn.Sequential):
    r"""The standard conv+bn+relu started in the VGG models
    and used in almost all modern network architectures.
    As usual, the convolution operation includes the bias term and
    the relu operation is performed inplace.
    The arguments are the same as in the convolution operation.
    See :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
    ):
        super(ConvBnReLU2d, self).__init__(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            stride=stride,
                            dilation=dilation,
                            groups=groups,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )


class CNNFeatureExtractor(nn.Sequential):
    def __init__(self, cfg: dict) -> None:
        kernel_size = cfg["conv_block"]["kernel_size"]
        out_size = cfg["linear_block"]["out_size"]

        encoder = nn.Sequential(
            OrderedDict(
                [
                    ("block1", ConvBnReLU2d(3, 32, kernel_size)),
                    ("mpool1", nn.MaxPool2d(3, stride=2, padding=1)),
                    ("block2", ConvBnReLU2d(32, 64, kernel_size)),
                    ("mpool2", nn.MaxPool2d(3, stride=2, padding=1)),
                    ("block3", ConvBnReLU2d(64, 128, kernel_size)),
                    ("mpool3", nn.MaxPool2d(3, stride=2, padding=1)),
                    ("block4", ConvBnReLU2d(128, 256, kernel_size)),
                    ("mpool4", nn.MaxPool2d(3, stride=2, padding=1)),
                ]
            )
        )
        linear = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 42 * 42, out_size), nn.Tanh()
        )
        super().__init__(
            OrderedDict(
                [
                    ("features", encoder),
                    ("linear", linear),
                ]
            )
        )
