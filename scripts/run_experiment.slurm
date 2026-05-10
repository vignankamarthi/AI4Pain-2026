#!/usr/bin/env bash
#SBATCH --job-name=ai4pain-evolve
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/%x-%j.out
#SBATCH --error=experiments/%x-%j.err

# Manual SLURM template. ANTIPATTERNS rule 10: framework Python NEVER calls
# sbatch. Vignan submits this manually at HIP-E.
#
# Usage on Explorer:
#   cd ~/AI4Pain-2026
#   sbatch --export=ALL,RUN_ID=<run_id> scripts/run_experiment.slurm

set -euo pipefail

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID environment variable is required" >&2
  exit 2
fi

RUN_DIR="experiments/${RUN_ID}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "no such run directory: ${RUN_DIR}" >&2
  exit 1
fi

module load python/3.11
module load cuda/12.4
source .venv/bin/activate

python -u "${RUN_DIR}/run.py" --run-dir "${RUN_DIR}"

echo "run ${RUN_ID} complete, result at ${RUN_DIR}/result.json"
