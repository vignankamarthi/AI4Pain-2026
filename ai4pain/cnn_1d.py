"""1D-CNN ResNet-style classifier for the AI4Pain 2026 case study.

Family: `1d_cnn`. Stem conv + N residual blocks + global avg pool + linear.
Spec hyperparams (FRAMEWORK §9.4 seed 1):
  - depth: number of residual blocks (default 4)
  - base_channels: stem output channels (default 32)
  - kernel_size: conv kernel size (default 7)
  - use_residual: whether to include skip connections (default True)

Input shape: (B, T, C). Internally permuted to (B, C, T) for Conv1d.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ai4pain.baselines import run_pytorch_model


class _ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, use_residual: bool):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_residual:
            out = out + identity
        return self.act(out)


class CNN1DResNet(nn.Module):
    """1D ResNet-style classifier. Input (B, T, C) -> logits (B, num_classes)."""

    def __init__(self, in_channels: int = 4, depth: int = 4,
                 base_channels: int = 32, kernel_size: int = 7,
                 use_residual: bool = True, num_classes: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            _ResBlock(base_channels, kernel_size, use_residual)
            for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def _cnn1d_factory(in_channels: int, T_max: int, model_cfg: dict,
                    num_classes: int) -> nn.Module:
    return CNN1DResNet(
        in_channels=in_channels,
        depth=int(model_cfg.get("depth", 4)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        kernel_size=int(model_cfg.get("kernel_size", 7)),
        use_residual=bool(model_cfg.get("use_residual", True)),
        num_classes=num_classes,
    )


def train_cnn1d(spec: dict, data_root: Path, out_dir: Path) -> dict:
    return run_pytorch_model(_cnn1d_factory, spec, data_root, out_dir,
                              name_tag="cnn1d_resnet")


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    from ai4pain.multiseed import run_multiseed
    return run_multiseed(train_cnn1d, spec, Path(data_root), run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()
    run_from_dir(args.run_dir, args.data_root)
