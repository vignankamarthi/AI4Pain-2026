#!/bin/bash
#SBATCH --job-name=ai4pain-dl-cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl_cnn_%j.out
#SBATCH --error=logs/dl_cnn_%j.err

# AI4Pain 2026 ResNet-1D training (NN #1 baseline). Phase 3 complete.
#
# Required env vars:
#   CONFIG -- one of bvp_eda, bvp_eda_resp, all_four
#
# Optional:
#   ARCH       -- cnn1d (default), nn2 (post-Phase 4), nn3 (post-Phase 5)
#   EPOCHS     -- default 100
#   BATCH_SIZE -- default 32
#   LR         -- default 1e-3
#   PATIENCE   -- default 20
#   RESUME     -- set to "--resume" to continue from latest.pt
#
# Hardware: 1x H200, 8 hr wall time, 12 GB RAM (per CLAUDE.md hardware constraints).

set -euo pipefail

: "${CONFIG:?CONFIG is required (bvp_eda|bvp_eda_resp|all_four)}"
: "${ARCH:=cnn1d}"
: "${EPOCHS:=100}"
: "${BATCH_SIZE:=32}"
: "${LR:=1e-3}"
: "${PATIENCE:=20}"
: "${RESUME:=}"

mkdir -p logs

python scripts/train_dl_models.py \
    --arch "${ARCH}" \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --patience "${PATIENCE}" \
    ${RESUME}

echo "[train_dl_cnn] Done: ${ARCH} / ${CONFIG}"
