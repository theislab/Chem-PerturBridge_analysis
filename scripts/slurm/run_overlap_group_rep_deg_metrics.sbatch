#!/bin/bash
#SBATCH --job-name=deg_t4
#SBATCH --account=icb
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Table 4: consolidated raw, population z-score, and moderated-t launchers.
# Submit from the repository root, or export REPO explicitly.
# Array tasks: 0=raw, 1=z-score-dataset, 2=z-score-dataset-cell-type, 3=moderated-t.
# Select a subset with sbatch --array=1,2. Z-score modes require population caches.
# See reproducibility/table4_6.md for selected run tags and input requirements.

set -euo pipefail

# Keep each array task's computation and output directory distinct.
if [[ -n "${COMPUTE:-}" ]]; then
  echo "Select representations with sbatch --array=0-3 instead of COMPUTE." >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --compute|--compute=*)
      echo "Select representations with sbatch --array=0-3 instead of --compute." >&2
      exit 2
      ;;
    --run-tag|--run-tag=*)
      echo "Use RUN_TAG=... before sbatch instead of passing --run-tag." >&2
      exit 2
      ;;
  esac
done

# Slurm runs a spooled copy, so BASH_SOURCE is only a local-execution fallback.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
PY="${PY:-/home/icb/olga.novitskaia/venv/bin/python}"
cd "$REPO"

DATASETS="${DATASETS:-l1000_phase1,l1000_phase2,tahoe,sciplex,cigs_mce,cigs_tcm}"
computations=(raw z-score-dataset z-score-dataset-cell-type moderated-t)
COMPUTE="${computations[${SLURM_ARRAY_TASK_ID:-0}]:?Use sbatch --array=0-3}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
Z_SCORE_STATS_ROOT="${Z_SCORE_STATS_ROOT:-results/w4_population_zscore_stats}"
# Match the selected LaTeX table runs for every representation.
MAX_BASELINE_PEERS="${MAX_BASELINE_PEERS:-256}"
if [[ -n "${RUN_TAG:-}" && "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ]]; then
  RUN_TAG="${RUN_TAG}_${COMPUTE}"
fi
RUN_TAG="${RUN_TAG:-${COMPUTE}_${MAX_BASELINE_PEERS}}"

args=(
  --datasets "$DATASETS"
  --compute "$COMPUTE"
  --z-score-stats-root "$Z_SCORE_STATS_ROOT"
  --run-tag "$RUN_TAG"
  --max-baseline-peers "$MAX_BASELINE_PEERS"
  --workers "$WORKERS"
)
# Otherwise let the Python runner select the appropriate analysis directory.
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

echo "[info] job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-0} workers=${WORKERS} repo=${REPO}"
echo "[info] datasets=${DATASETS} compute=${COMPUTE} run_tag=${RUN_TAG} max_baseline_peers=${MAX_BASELINE_PEERS}"

exec "$PY" scripts/run_overlap_group_rep_deg_metrics.py "${args[@]}" "$@"
