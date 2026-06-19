# -*- coding: utf-8 -*-
"""
GraphMixer – Temporal-Graph-Baseline (binäre Link-Vorhersage)
=============================================================

Referenz-Implementierung nach Cong et al., "Do We Really Need Complicated Model
Architectures for Temporal Networks?" (ICLR 2023), zugeschnitten auf den
Bike-Sharing-Datensatz dieses Projekts.

Das Modell besteht aus drei Bausteinen (siehe GraphMixer - Grundlagen.md):
  1. Link-Encoder   : MLP-Mixer über die letzten K Kanten-Ereignisse eines Knotens
                      (mit FESTER, nicht gelernter Zeit-Kodierung).
  2. Node-Encoder   : Mittelung der Knoten-Features der jüngsten Nachbarn.
  3. Link-Classifier: MLP über die kombinierten Embeddings von (u, v) -> Link-Score.

Eingabe : die aufbereiteten Dateien aus  prepared/  (ml_citibike.csv/.npy/_node.npy)
Ausgabe : Vorhersagen im Format des gemeinsamen Eval-Moduls
          (Spalten: u, i, bin_idx, score) -> shared_eval.SharedLinkEval.score_binary

HINWEIS Knoten-Indizierung
  Die ml_citibike-Dateien sind 1-indiziert (Zeile 0 = Padding). Intern rechnet
  dieses Modul ebenfalls 1-indiziert. Beim EXPORT für shared_eval wird auf die
  kanonische 0-Indizierung (-1) zurückgerechnet (siehe export_predictions()).

Lokal ist kein PyTorch installiert -> dieses Modul ist für Colab/GPU gedacht.
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
# 1) KONFIGURATION
# ===========================================================================
@dataclass
class GMConfig:
    # Pfade (relativ zu diesem Skript)
    prep_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "prepared Data")
    # Modell-Hyperparameter
    num_neighbors: int = 20        # K – Anzahl jüngster Kanten je Knoten (Link-Encoder)
    time_dim: int = 100            # Dimension der festen Zeit-Kodierung
    mixer_layers: int = 2          # Anzahl MLP-Mixer-Blöcke
    hidden_dim: int = 128          # versteckte Dimension der MLPs
    node_emb_dim: int = 100        # Ausgabedimension des Node-Encoders
    dropout: float = 0.1
    # Trainings-Parameter
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    neg_per_pos: int = 1           # Negative je Positive beim Training
    seed: int = 42
    # zeitliche Grenzen (in Sekunden seit Fensterbeginn) – identisch zu shared_eval
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
# 2) FESTE ZEIT-KODIERUNG
# ===========================================================================
class FixedTimeEncoder(nn.Module):
    """Bildet eine Zeitdifferenz Δt (Sekunden) auf einen Vektor ab.

    GraphMixer nutzt bewusst eine FESTE (nicht gelernte) Kodierung, weil
    lernbare Zeit-Encoder das Training destabilisieren. Wir verwenden eine
    Kosinus-Kodierung mit logarithmisch gestaffelten Frequenzen (wie bei der
    Positional Encoding): langsame Frequenzen erfassen "vor langer Zeit",
    schnelle Frequenzen "gerade eben".
    """

    def __init__(self, dim: int):
        super().__init__()
        # Frequenzen fest vorgeben und NICHT als Parameter registrieren
        freqs = 1.0 / (10000 ** (np.arange(0, dim) / dim))
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: (...,)  ->  (..., dim)
        return torch.cos(dt.unsqueeze(-1) * self.freqs)


# ===========================================================================
# 3) MLP-MIXER (Herzstück des Link-Encoders)
# ===========================================================================
class MixerBlock(nn.Module):
    """Ein MLP-Mixer-Block: erst Token-Mixing (über die K Ereignisse),
    dann Channel-Mixing (über die Feature-Dimension). Beide mit Residual."""

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
        # --- Token-Mixing: mischt Information ÜBER die K Ereignisse ---
        y = self.norm_token(x).transpose(1, 2)          # (B, C, K)
        y = self.token_mlp(y).transpose(1, 2)           # (B, K, C)
        x = x + y
        # --- Channel-Mixing: mischt Information ÜBER die Features ---
        z = self.norm_channel(x)
        z = self.channel_mlp(z)
        return x + z


class LinkEncoder(nn.Module):
    """Verdichtet die letzten K Kanten-Ereignisse eines Knotens zu einem Vektor.

    Eingabe pro Knoten: eine Matrix (K, d_edge + d_time) aus
      [Kanten-Feature  ||  feste Zeit-Kodierung von (t_query - t_event)].
    Ausgabe: ein Embedding je Knoten (mittlere Token-Repräsentation).
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
        return x.mean(dim=1)          # Mittelung über die K Ereignisse -> (batch, in_dim)


# ===========================================================================
# 4) NODE-ENCODER
# ===========================================================================
class NodeEncoder(nn.Module):
    """Beschreibt die Identität/jüngste Aktivität eines Knotens über den
    Mittelwert der Knoten-Features seiner jüngsten Nachbarn plus eigene Features."""

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
# 5) GESAMTMODELL
# ===========================================================================
class GraphMixer(nn.Module):
    """Kombiniert Link- und Node-Encoder für u und v und sagt den Link-Score voraus."""

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
        """Embedding eines (Satzes von) Knoten aus Link- und Node-Encoder."""
        return torch.cat([self.link_enc(link_tokens),
                          self.node_enc(own_feat, neigh_mean_feat)], dim=-1)

    def forward(self, u_pack, v_pack) -> torch.Tensor:
        """u_pack / v_pack = (link_tokens, own_feat, neigh_mean_feat).
        Rückgabe: Logit (vor Sigmoid) je Paar, Form (batch,)."""
        hu = self.encode_node(*u_pack)
        hv = self.encode_node(*v_pack)
        return self.classifier(torch.cat([hu, hv], dim=-1)).squeeze(-1)
