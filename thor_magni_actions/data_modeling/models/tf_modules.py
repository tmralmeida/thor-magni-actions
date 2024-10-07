import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """scaled dot product function: softmax(Q @ K / sqrt(d)) @ V"""
    d_k = queries.size()[-1]
    attn_logits = torch.matmul(queries, keys.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, values)
    return values, attention


class MultiHeadAttention(nn.Module):
    """Multi Head Attention Layer"""

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Emebedding dim must be 0 modulo number of heads!")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], ret_attention: bool = False
    ):
        bs, traj_len, embed_dim = x.shape
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(bs, traj_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(queries, keys, values, mask=mask)
        values = values.permute(0, 2, 1, 3)  # batch, traj_len, head, dims
        values = values.reshape(bs, traj_len, embed_dim)

        out = self.o_proj(values)
        if ret_attention:
            return out, attention
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self, input_dim: int, num_heads: int, dim_feedforward: int, dropout: float = 0.0
    ) -> None:
        """Encoder block with pre layer normalization"""
        super().__init__()

        # attention layer
        self.self_attention = MultiHeadAttention(input_dim, input_dim, num_heads)

        # 2-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x_norm1 = self.norm1(x)
        attn_out = self.self_attention(x_norm1, mask=mask)
        x = x + self.dropout(attn_out)

        # MLP part
        x_norm2 = self.norm2(x)
        linear_out = self.linear_net(x_norm2)
        x = x + self.dropout(linear_out)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
