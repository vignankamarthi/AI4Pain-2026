"""Deep learning model definitions for AI4Pain 2026.

Three architectures, all taking multi-channel raw signal windows:

  NN #1 (ResNet1D): baseline, ported from Blood-Pressure-Inference-with-BVP.
      Output head adapted from regression (1 unit) to 3-class logits.
      Input channels configurable per ablation config (2, 3, or 4).

  NN #2 (NN2SOTA): current SOTA for 3-class pain localization from multimodal
      physiological signals. Architecture selection requires the Phase 4
      literature review and HIP-5 approval before implementation.

  NN #3 (NN3Novel): novel homemade framework. The paper's central novelty
      claim. Design driven by Phase 5 gap analysis. Requires HIP-6 approval.

BP inference's ResNet-BiGRU is deliberately NOT ported. See scaffolding plan
Phase 3 for the rationale.

Scaffolding state:
  - ResNet1D is a structural stub (matches BP inference's shape, 3-class head)
  - NN2SOTA is an empty placeholder class, populated in Phase 4
  - NN3Novel is an empty placeholder class, populated in Phase 5
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """1D residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _make_layer(in_channels: int, out_channels: int, n_blocks: int, stride: int = 1):
    layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, n_blocks):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResNet1D(nn.Module):
    """1D ResNet for multi-channel raw physiological signals.

    Ported from Blood-Pressure-Inference-with-BVP/src/dl_models.py::ResNet1D.
    The original produced a single regression output (BP); this version
    produces 3-class logits (NP / HP / AP) and accepts a configurable number
    of input channels (2, 3, or 4) to match the ablation configs.

    Parameters
    ----------
    in_channels : int
        Number of raw signal channels fed to the network.
    num_classes : int
        Output dimensionality of the classification head. Defaults to 3.
    dropout : float
        Dropout rate in the classification head.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = _make_layer(64, 64, n_blocks=2)
        self.layer2 = _make_layer(64, 128, n_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, n_blocks=2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


class NN2SOTA(nn.Module):
    """Current-SOTA architecture for 3-class pain localization.

    TBD Phase 4. Selection requires the `research/NN2_SOTA_RESEARCH.md`
    literature review and HIP-5 approval before implementation. Candidate
    directions from the scaffolding plan include multimodal fusion transformers,
    state-space models (Mamba / S4 / S5) for long physiological sequences, and
    self-supervised pretraining adapted to the small-label regime (without
    violating the NO EXTERNAL DATA rule).

    Raises ``NotImplementedError`` at construction time so ``build_model('nn2', ...)``
    fails with a clear, early error rather than crashing later in the
    optimizer or forward pass.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 3):
        raise NotImplementedError(
            "NN2SOTA: TBD Phase 4. Complete research/NN2_SOTA_RESEARCH.md and "
            "get HIP-5 approval before instantiating this class."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("NN2SOTA: TBD Phase 4.")


class NN3Novel(nn.Module):
    """Novel homemade framework for 3-class pain localization.

    TBD Phase 5. Design driven by the `research/NN3_NOVEL_FRAMEWORK.md` gap
    analysis and requires HIP-6 approval. This is the paper's central novelty
    claim and must be defensible under reviewer scrutiny. Candidate directions
    from the scaffolding plan include hierarchical detection-then-localization,
    cross-modal attention with physiological priors, frequency-aware branches
    for autonomic bands, and phase-aware cross-signal fusion.

    Raises ``NotImplementedError`` at construction time so ``build_model('nn3', ...)``
    fails with a clear, early error rather than crashing later in the
    optimizer or forward pass.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 3):
        raise NotImplementedError(
            "NN3Novel: TBD Phase 5. Complete research/NN3_NOVEL_FRAMEWORK.md and "
            "get HIP-6 approval before instantiating this class."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError("NN3Novel: TBD Phase 5.")


ARCH_REGISTRY = {
    "cnn1d": ResNet1D,
    "nn2": NN2SOTA,
    "nn3": NN3Novel,
}


def build_model(arch: str, in_channels: int, num_classes: int = 3) -> nn.Module:
    """Factory for the three supported architectures."""
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Expected one of {list(ARCH_REGISTRY)}.")
    return ARCH_REGISTRY[arch](in_channels=in_channels, num_classes=num_classes)
