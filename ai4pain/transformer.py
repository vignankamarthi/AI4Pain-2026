"""Lightweight Transformer encoder classifier for AI4Pain 2026.

Family: `transformer`. Linear projection + sinusoidal positional encoding +
N-layer Transformer encoder + mean-pool over time + FC.

Spec hyperparams (FRAMEWORK §9.4 seed 3):
  - d_model: embedding dim (default 64)
  - num_heads: attention heads (default 4)
  - num_layers: encoder layers (default 2)
  - ff_dim: feedforward hidden (default 128)
  - dropout: dropout rate (default 0.1)

Input shape: (B, T, C). C is projected to d_model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from ai4pain.baselines import run_pytorch_model


def _sinusoidal_positional_encoding(T: int, d_model: int,
                                     device: torch.device | None = None
                                     ) -> torch.Tensor:
    """Standard sinusoidal positional encoding (Vaswani et al. 2017).
    Returns (T, d_model)."""
    pe = torch.zeros(T, d_model, device=device)
    position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class LightweightTransformer(nn.Module):
    """Transformer encoder classifier. (B, T, C) -> logits (B, num_classes)."""

    def __init__(self, in_channels: int = 4, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 ff_dim: int = 128, dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.input_proj(x)  # (B, T, d_model)
        T = x.shape[1]
        pe = _sinusoidal_positional_encoding(T, self.d_model, device=x.device)
        x = x + pe.unsqueeze(0)
        x = self.encoder(x)        # (B, T, d_model)
        x = x.mean(dim=1)          # (B, d_model)
        return self.fc(x)


def _transformer_factory(in_channels: int, T_max: int, model_cfg: dict,
                          num_classes: int) -> nn.Module:
    return LightweightTransformer(
        in_channels=in_channels,
        d_model=int(model_cfg.get("d_model", 64)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        ff_dim=int(model_cfg.get("ff_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_classes=num_classes,
    )


def train_transformer(spec: dict, data_root: Path, out_dir: Path) -> dict:
    return run_pytorch_model(_transformer_factory, spec, data_root, out_dir,
                              name_tag="transformer")


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    return train_transformer(spec, data_root=Path(data_root), out_dir=run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
