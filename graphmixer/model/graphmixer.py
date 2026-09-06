# -*- coding: utf-8 -*-
"""
GraphMixer temporal-graph baseline for binary link prediction.

Follows Cong et al., "Do We Really Need Complicated Model Architectures for
Temporal Networks?" (ICLR 2023), adapted to the bike-sharing dataset.

Three parts:
  1. link encoder    MLP-Mixer over a node's last K edge events, using a fixed
                     (not learned) time encoding.
  2. node encoder    mean of the node features of the most recent neighbours.
  3. classifier      MLP over the combined (u, v) embeddings -> link score.

Input:  ml_citibike.csv / .npy / _node.npy from "prepared Data".
Output: u, i, bin_idx, score -> shared_eval.SharedLinkEval.score_binary

Node indexing: the ml_citibike files are 1-indexed (row 0 is padding) and this
module keeps that convention internally. export_predictions() subtracts 1 to
get back to the canonical 0-indexing that shared_eval expects.

Needs PyTorch. A GPU is optional, the dataset is small enough for CPU.
"""

from __future__ import annotations
import os
import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --- configuration ---------------------------------------------------------
@dataclass
class GMConfig:
    # paths, relative to this file
    prep_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "prepared Data")
    # model
    num_neighbors: int = 20        # K: most recent edges per node
    time_dim: int = 100            # width of the fixed time encoding
    mixer_layers: int = 2
    hidden_dim: int = 128
    node_emb_dim: int = 100        # output width of the node encoder
    dropout: float = 0.1
    # training
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    # Training pairs come from shared_eval, so training and evaluation see the
    # same class balance. Capped because each pair needs two neighbourhood
    # packs; the full 374k candidates would triple the epoch time.
    max_train_pairs: int = 150_000
    seed: int = 42
    # split boundaries in seconds from the start of the window, as in shared_eval
    bin_minutes: int = 30
    train_days: int = 21
    val_days: int = 4

    @property
    def train_end_s(self) -> int:
        return self.train_days * 24 * 3600

    @property
    def val_end_s(self) -> int:
        return (self.train_days + self.val_days) * 24 * 3600

    @property
    def bin_seconds(self) -> int:
        return self.bin_minutes * 60


# --- fixed time encoding ---------------------------------------------------
class FixedTimeEncoder(nn.Module):
    """Maps a time difference in seconds to a vector.

    The encoding is fixed rather than learned; the paper reports that learnable
    time encoders destabilise training. Cosine with log-spaced frequencies, as
    in positional encoding: slow frequencies carry "long ago", fast ones
    "just now".
    """

    def __init__(self, dim: int):
        super().__init__()
        # fixed frequencies, deliberately not registered as parameters
        freqs = 1.0 / (10000 ** (np.arange(0, dim) / dim))
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: (...,)  ->  (..., dim)
        return torch.cos(dt.unsqueeze(-1) * self.freqs)


# --- MLP-Mixer, the core of the link encoder --------------------------------
class MixerBlock(nn.Module):
    """One MLP-Mixer block: token mixing over the K events, then channel
    mixing over the feature dimension. Both with a residual connection."""

    def __init__(self, num_tokens: int, num_channels: int, hidden: int, dropout: float):
        super().__init__()
        self.norm_token = nn.LayerNorm(num_channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_tokens), nn.Dropout(dropout),
        )
        self.norm_channel = nn.LayerNorm(num_channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_channels, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_channels), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_tokens=K, num_channels)
        # token mixing: across the K events
        y = self.norm_token(x).transpose(1, 2)          # (B, C, K)
        y = self.token_mlp(y).transpose(1, 2)           # (B, K, C)
        x = x + y
        # channel mixing: across the feature dimension
        z = self.norm_channel(x)
        z = self.channel_mlp(z)
        return x + z


class LinkEncoder(nn.Module):
    """Compresses a node's last K edge events into one vector.

    Input per node: a (K, d_edge + d_time) matrix of
      [edge feature || fixed time encoding of (t_query - t_event)].
    Output: one embedding per node, the mean token representation.
    """

    def __init__(self, cfg: GMConfig, edge_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        in_dim = edge_feat_dim + cfg.time_dim
        self.input_proj = nn.Linear(in_dim, in_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(cfg.num_neighbors, in_dim, cfg.hidden_dim, cfg.dropout)
            for _ in range(cfg.mixer_layers)
        ])
        self.norm = nn.LayerNorm(in_dim)
        self.out_dim = in_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch, K, d_edge + d_time)
        x = self.input_proj(tokens)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=1)          # mean over the K events -> (batch, in_dim)


# --- node encoder ----------------------------------------------------------
class NodeEncoder(nn.Module):
    """Describes a node through its own features plus the mean features of its
    most recent neighbours."""

    def __init__(self, cfg: GMConfig, node_feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feat_dim * 2, cfg.hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.node_emb_dim),
        )
        self.out_dim = cfg.node_emb_dim

    def forward(self, own_feat: torch.Tensor, neigh_mean_feat: torch.Tensor) -> torch.Tensor:
        # own_feat / neigh_mean_feat: (batch, node_feat_dim)
        return self.mlp(torch.cat([own_feat, neigh_mean_feat], dim=-1))


# --- full model ------------------------------------------------------------
class GraphMixer(nn.Module):
    """Combines link and node encoder for u and v into a link score."""

    def __init__(self, cfg: GMConfig, edge_feat_dim: int, node_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.time_enc = FixedTimeEncoder(cfg.time_dim)
        self.link_enc = LinkEncoder(cfg, edge_feat_dim)
        self.node_enc = NodeEncoder(cfg, node_feat_dim)
        pair_dim = 2 * (self.link_enc.out_dim + self.node_enc.out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def encode_node(self, link_tokens, own_feat, neigh_mean_feat):
        """Embedding for a batch of nodes, from the link and node encoders."""
        return torch.cat([self.link_enc(link_tokens),
                          self.node_enc(own_feat, neigh_mean_feat)], dim=-1)

    def forward(self, u_pack, v_pack) -> torch.Tensor:
        """u_pack / v_pack = (link_tokens, own_feat, neigh_mean_feat).
        Returns one logit per pair, shape (batch,), before the sigmoid."""
        hu = self.encode_node(*u_pack)
        hv = self.encode_node(*v_pack)
        return self.classifier(torch.cat([hu, hv], dim=-1)).squeeze(-1)
