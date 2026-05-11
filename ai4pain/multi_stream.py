"""Multi-stream BiGRU classifier with late-concat fusion.

Family: `multi_stream_bigru`. Per-channel BiGRU encoder + mean-pool over time
+ concat across channels + dropout + FC.

Spec hyperparams (FRAMEWORK §9.4 seed 4):
  - per_channel_hidden: hidden_size of each per-channel BiGRU (default 32)
  - per_channel_layers: BiGRU depth per channel (default 1)
  - fusion: how to combine channels ('late_concat' supported; default)
  - fusion_dropout: dropout before final FC (default 0.2)

Input shape: (B, T, C). Each channel becomes its own univariate BiGRU input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ai4pain.baselines import run_pytorch_model


class MultiStreamBiGRU(nn.Module):
    """Per-channel BiGRU + late-concat fusion. (B, T, C) -> logits."""

    def __init__(self, in_channels: int = 4, per_channel_hidden: int = 32,
                 per_channel_layers: int = 1,
                 fusion: str = "late_concat",
                 fusion_dropout: float = 0.2,
                 num_classes: int = 3):
        super().__init__()
        if fusion != "late_concat":
            raise ValueError(f"unsupported fusion {fusion!r}; only 'late_concat'")
        self.in_channels = in_channels
        self.per_channel_hidden = per_channel_hidden
        # One BiGRU per input channel
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=1, hidden_size=per_channel_hidden,
                   num_layers=per_channel_layers, batch_first=True,
                   bidirectional=True,
                   dropout=fusion_dropout if per_channel_layers > 1 else 0.0)
            for _ in range(in_channels)
        ])
        fused_dim = in_channels * per_channel_hidden * 2  # bidir
        self.dropout = nn.Dropout(fusion_dropout)
        self.fc = nn.Linear(fused_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        per_channel = []
        for c, enc in enumerate(self.encoders):
            xc = x[:, :, c:c + 1]  # (B, T, 1)
            out, _ = enc(xc)        # (B, T, 2 * hidden)
            pooled = out.mean(dim=1)  # (B, 2 * hidden)
            per_channel.append(pooled)
        fused = torch.cat(per_channel, dim=1)  # (B, C * 2 * hidden)
        return self.fc(self.dropout(fused))


def _multi_stream_factory(in_channels: int, T_max: int, model_cfg: dict,
                           num_classes: int) -> nn.Module:
    return MultiStreamBiGRU(
        in_channels=in_channels,
        per_channel_hidden=int(model_cfg.get("per_channel_hidden", 32)),
        per_channel_layers=int(model_cfg.get("per_channel_layers", 1)),
        fusion=model_cfg.get("fusion", "late_concat"),
        fusion_dropout=float(model_cfg.get("fusion_dropout", 0.2)),
        num_classes=num_classes,
    )


def train_multi_stream(spec: dict, data_root: Path, out_dir: Path) -> dict:
    return run_pytorch_model(_multi_stream_factory, spec, data_root, out_dir,
                              name_tag="multi_stream_bigru")


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    return train_multi_stream(spec, data_root=Path(data_root), out_dir=run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
