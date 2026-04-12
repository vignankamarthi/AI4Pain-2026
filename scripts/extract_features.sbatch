#!/bin/bash
#SBATCH --job-name=ai4pain-extract
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# AI4Pain 2026 Rust feature extraction.
#
# Invokes the forked extractor (Phase 1 complete) for one (split, signal) pair.
# To extract features for all four signals across train + validation, submit
# this script 8 times with different SPLIT and SIGNAL environment variables,
# or wrap it in a chain script.
#
# Required env vars:
#   SPLIT  -- one of train, validation, test
#   SIGNAL -- one of bvp, eda, resp, spo2
#
# Optional:
#   RESUME -- set to "--resume" to restart from checkpoint
#   FRAMEWORKS -- comma-separated subset (default: all three)
#
# Output: data/features/results_${SPLIT}_${SIGNAL}.csv
# Checkpoint: data/features/checkpoint_${SPLIT}_${SIGNAL}.json (atomic, resumable)

set -euo pipefail

: "${SPLIT:?SPLIT is required (train|validation|test)}"
: "${SIGNAL:?SIGNAL is required (bvp|eda|resp|spo2)}"
: "${RESUME:=}"
: "${FRAMEWORKS:=catch22,entropy,stats}"

# Capitalize signal name for the AI4Pain release directory layout
case "${SIGNAL}" in
    bvp)  SIGNAL_DIR="Bvp" ;;
    eda)  SIGNAL_DIR="Eda" ;;
    resp) SIGNAL_DIR="Resp" ;;
    spo2) SIGNAL_DIR="SpO2" ;;
    *)    echo "Unknown SIGNAL=${SIGNAL}" >&2; exit 1 ;;
esac

INPUT_DIR="data/raw/${SPLIT}/${SIGNAL_DIR}"
OUTPUT_DIR="data/features"

mkdir -p "${OUTPUT_DIR}" logs

cd feature-extraction-rust
cargo build --release
cd ..

./feature-extraction-rust/target/release/ai4pain_feature_extraction extract \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --signal-type "${SIGNAL}" \
    --frameworks "${FRAMEWORKS}" \
    --workers 8 \
    ${RESUME}

echo "[extract_features] Done: ${SPLIT}/${SIGNAL} -> ${OUTPUT_DIR}/results_${SPLIT}_${SIGNAL}.csv"
