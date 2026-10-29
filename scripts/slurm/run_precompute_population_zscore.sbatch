#!/bin/bash
#SBATCH --job-name=w4_precompute
#SBATCH --account=icb
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/w4_population_zscore/%x.%j.out
#SBATCH --error=logs/w4_population_zscore/%x.%j.err

# Fit per-gene population mean/SD caches for both W4 scales
# (dataset-wide and dataset x cell type). Scoring CLIs read these
# caches and will refuse to start without them.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/icb/olga.novitskaia/venv/bin/python
DATA="${DATA:-/lustre/groups/ml01/workspace/olga.novitskaia/data_updated}"
CACHE="${CACHE:-${REPO}/results/w4_population_zscore_stats}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-2}}"
SCOPE="${SCOPE:-both}"

cd "$REPO"
mkdir -p logs/w4_population_zscore "$CACHE"

echo "[info] job=${SLURM_JOB_ID:-none} node=$(hostname) workers=${WORKERS} scope=${SCOPE}"
echo "[info] data=${DATA} cache=${CACHE}"

"$PY" scripts/precompute_population_zscore.py \
  --dataset-dir "l1000_phase1=${DATA}/l1000_phase1/deg_data/group_rep/full/qc_false/filter_min_cells_0/results" \
  --dataset-dir "l1000_phase2=${DATA}/l1000_phase2/deg_data/group_rep/full/qc_false/filter_min_cells_0/results" \
  --dataset-dir "tahoe=${DATA}/tahoe/deg_data/group_rep/full/qc_false/filter_min_cells_50/results" \
  --dataset-dir "sciplex=${DATA}/sciplex/deg_data/group_rep/full/qc_false/filter_min_cells_10/results" \
  --dataset-dir "cigs_mce=${DATA}/cigs_mce/deg_data/group_rep/full/qc_false/filter_min_cells_0/results" \
  --dataset-dir "cigs_tcm=${DATA}/cigs_tcm/deg_data/group_rep/full/qc_false/filter_min_cells_0/results" \
  --scope "${SCOPE}" \
  --row-chunk-size 1024 \
  --workers "${WORKERS}" \
  --cache-root "${CACHE}" \
  --qc-output "${CACHE}/precompute_qc.tsv"
