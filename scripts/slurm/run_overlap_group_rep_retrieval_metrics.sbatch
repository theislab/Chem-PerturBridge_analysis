#!/bin/bash
#SBATCH --job-name=retrieval_t9
#SBATCH --account=icb
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Table 9: one representation per array task, with all three similarities.
# Tasks: 0=raw, 1=direction, 2=moderated-t, 3=z-score-dataset,
#        4=z-score-dataset-cell-type. Select subsets with sbatch --array.
# Submit from the repository root, or export REPO explicitly.
# Z-score tasks require existing population caches; see reproducibility/table9.md.

set -euo pipefail

if [[ -n "${COMPUTE:-}" ]]; then
  echo "Select representations with sbatch --array=0-4 instead of COMPUTE." >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --compute|--compute=*)
      echo "Select representations with sbatch --array=0-4 instead of --compute." >&2
      exit 2
      ;;
    --run-tag|--run-tag=*)
      echo "Use RUN_TAG=... before sbatch instead of passing --run-tag." >&2
      exit 2
      ;;
  esac
done

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PY="${PY:-/home/icb/olga.novitskaia/venv/bin/python}"
cd "$REPO"

# Preserve the Table 9 dataset order and the runner's metric-specific peer caps.
DATASETS="${DATASETS:-l1000_phase1,l1000_phase2,tahoe,cigs_mce,cigs_tcm,sciplex}"
computations=(raw direction moderated-t z-score-dataset z-score-dataset-cell-type)
COMPUTE="${computations[${SLURM_ARRAY_TASK_ID:-0}]:?Use sbatch --array=0-4}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
Z_SCORE_STATS_ROOT="${Z_SCORE_STATS_ROOT:-results/w4_population_zscore_stats}"
if [[ -n "${RUN_TAG:-}" && "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ]]; then
  RUN_TAG="${RUN_TAG}_${COMPUTE}"
fi
RUN_TAG="${RUN_TAG:-${COMPUTE}}"

args=(
  --datasets "$DATASETS"
  --compute "$COMPUTE"
  --z-score-stats-root "$Z_SCORE_STATS_ROOT"
  --run-tag "$RUN_TAG"
  --workers "$WORKERS"
)
if [[ -n "${MAX_BASELINE_PEERS:-}" ]]; then
  args+=(--max-baseline-peers "$MAX_BASELINE_PEERS")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

echo "[info] job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-0} workers=${WORKERS} repo=${REPO}"
echo "[info] datasets=${DATASETS} compute=${COMPUTE} run_tag=${RUN_TAG} max_baseline_peers=${MAX_BASELINE_PEERS:-automatic}"

exec "$PY" scripts/run_overlap_group_rep_retrieval_metrics.py "${args[@]}" "$@"
