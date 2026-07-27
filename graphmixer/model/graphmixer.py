# -*- coding: utf-8 -*-
"""
GraphMixer – temporal-graph baseline (binary link prediction)
=============================================================

Reference implementation after Cong et al., "Do We Really Need Complicated
Model Architectures for Temporal Networks?" (ICLR 2023), adapted to this
project's bike-sharing dataset.

The model has three building blocks (see GraphMixer - Grundlagen.md):
  1. Link encoder   : MLP-Mixer over a node's last K edge events
                      (with a FIXED, non-learned time encoding).
  2. Node encoder   : mean of the node features of the most recent neighbors.
  3. Link classifier: MLP over the combined (u, v) embeddings -> link score.

Input : the prepared files from  prepared/  (ml_citibike.csv/.npy/_node.npy)
Output: predictions in the shared eval format
        (columns: u, i, bin_idx, score) -> shared_eval.SharedLinkEval.score_binary

NOTE on node indexing
  The ml_citibike files are 1-indexed (row 0 = padding). This module works
  1-indexed internally too. On EXPORT for shared_eval we shift back to the
  canonical 0-indexing (-1) (see export_predictions()).

Needs PyTorch (GPU optional; CPU is fine, the dataset is small).
"""

from __future__ import annotations
import os
import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===========================================================================
# 1) CONFIG
# ===========================================================================
@dataclass
class GMConfig:
    # paths (relative to this script)
    prep_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "prepared Data")
    # model hyperparameters
    num_neighbors: int = 20        # K – number of most recent edges per node (link encoder)
    time_dim: int = 100            # dimension of the fixed time encoding
    mixer_layers: int = 2          # number of MLP-Mixer blocks
    hidden_dim: int = 128          # hidden dimension of the MLPs
    node_emb_dim: int = 100        # output dimension of the node encoder
    dropout: float = 0.1
    # training parameters
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    neg_per_pos: int = 1           # negatives per positive during training
    seed: int = 42
    # time bounds (in seconds since window start) – identical to shared_eval
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


# ===========================================================================
# 2) FIXED TIME ENCODING
# ===========================================================================
class FixedTimeEncoder(nn.Module):
    """Maps a time difference dt (seconds) to a vector.

    GraphMixer deliberately uses a FIXED (non-learned) encoding, because
    learnable time encoders tend to destabilize training. We use a cosine
    encoding with log-spaced frequencies (like positional encoding): slow
    frequencies capture "long ago", fast frequencies "just now".
    """

    def __init__(self, dim: int):
        super().__init__()
        # fix the frequencies and do NOT register them as parameters
        freqs = 1.0 / (10000 ** (np.arange(0, dim) / dim))
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: (...,)  ->  (..., dim)
        return torch.cos(dt.unsqueeze(-1) * self.freqs)


# ===========================================================================
# 3) MLP-MIXER (heart of the link encoder)
# ===========================================================================
class MixerBlock(nn.Module):
    """One MLP-Mixer block: first token mixing (across the K events),
    then channel mixing (across the feature dimension). Both with residuals."""

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
        # --- token mixing: mix information ACROSS the K events ---
        y = self.norm_token(x).transpose(1, 2)          # (B, C, K)
        y = self.token_mlp(y).transpose(1, 2)           # (B, K, C)
        x = x + y
        # --- channel mixing: mix information ACROSS the features ---
        z = self.norm_channel(x)
        z = self.channel_mlp(z)
        return x + z


class LinkEncoder(nn.Module):
    """Condenses a node's last K edge events into a vector.

    Input per node: a matrix (K, d_edge + d_time) of
      [edge feature  ||  fixed time encoding of (t_query - t_event)].
    Output: one embedding per node (the mean token representation).
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


# ===========================================================================
# 4) NODE ENCODER
# ===========================================================================
class NodeEncoder(nn.Module):
    """Describes a node's identity / recent activity via the mean of the node
    features of its most recent neighbors, plus its own features."""

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


# ===========================================================================
# 5) FULL MODEL
# ===========================================================================
class GraphMixer(nn.Module):
    """Combines link and node encoder for u and v and predicts the link score."""

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
        """Embedding of a (set of) node(s) from the link and node encoders."""
        return torch.cat([self.link_enc(link_tokens),
                          self.node_enc(own_feat, neigh_mean_feat)], dim=-1)

    def forward(self, u_pack, v_pack) -> torch.Tensor:
        """u_pack / v_pack = (link_tokens, own_feat, neigh_mean_feat).
        Returns: logit (pre-sigmoid) per pair, shape (batch,)."""
        hu = self.encode_node(*u_pack)
        hv = self.encode_node(*v_pack)
        return self.classifier(torch.cat([hu, hv], dim=-1)).squeeze(-1)
