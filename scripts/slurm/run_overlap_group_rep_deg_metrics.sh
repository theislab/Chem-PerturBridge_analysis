#!/bin/bash

# Table 6: consolidated raw, population z-score, and moderated-t launchers.
# Run with bash; the launcher submits the array and waits for completion.
# Array tasks: 0=raw, 1=z-score-dataset, 2=z-score-dataset-cell-type, 3=moderated-t.
# Select a subset with ARRAY=1,2 bash scripts/slurm/run_overlap_group_rep_deg_metrics.sh. Z-score modes require population caches.
# See reproducibility/table6_8.md for selected run tags and input requirements.

set -euo pipefail

# Keep each array task's computation and output directory distinct.
if [[ -n "${COMPUTE:-}" ]]; then
  echo "Select representations with ARRAY=0-3 bash scripts/slurm/run_overlap_group_rep_deg_metrics.sh instead of COMPUTE." >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --compute|--compute=*)
      echo "Select representations with ARRAY=0-3 bash scripts/slurm/run_overlap_group_rep_deg_metrics.sh instead of --compute." >&2
      exit 2
      ;;
    --run-tag|--run-tag=*)
      echo "Use RUN_TAG=... before bash instead of passing --run-tag." >&2
      exit 2
      ;;
  esac
done

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(cd "${REPO:-$(dirname "${SCRIPT_PATH}")/../..}" && pwd)"
export REPO

# sbatch --wrap uses /bin/sh; preserve Bash arrays and literal CLI arguments.
quote_command() {
    local arg
    for arg in "$@"; do
        printf "'%s' " "${arg//\'/\'\\\'\'}"
    done
}

WORKER_SCRIPT="$(cat <<'BASH'
set -euo pipefail
UV_BIN="${UV_BIN:-uv}"
cd "$REPO"

DATASETS="${DATASETS:-l1000_phase1,l1000_phase2,tahoe,sciplex,cigs_mce,cigs_tcm}"
computations=(raw z-score-dataset z-score-dataset-cell-type moderated-t)
COMPUTE="${computations[${SLURM_ARRAY_TASK_ID:-0}]:?Use ARRAY=0-3 bash scripts/slurm/run_overlap_group_rep_deg_metrics.sh}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
Z_SCORE_STATS_ROOT="${Z_SCORE_STATS_ROOT:-results/population_zscore_stats}"
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

exec "${UV_BIN}" run python scripts/run_overlap_group_rep_deg_metrics.py "${args[@]}" "$@"
BASH
)"

# Resource overrides use the same defaults as the previous array job.
SLURM_ARGS=(
  -W
  --job-name=deg_t6
  --account="${ACCOUNT:-icb}"
  --partition="${PARTITION:-cpu_p}"
  --qos="${QOS:-cpu_normal}"
  --cpus-per-task="${CPUS_PER_TASK:-8}"
  --mem="${MEM:-500G}"
  --time="${TIME:-1-00:00:00}"
  --array="${ARRAY:-0-3}"
  --output=slurm-%x-%A_%a.out
  --error=slurm-%x-%A_%a.err
  --export=ALL
  --chdir="${REPO}"
)

sbatch "${SLURM_ARGS[@]}" --wrap="$(quote_command bash -c "${WORKER_SCRIPT}" -- "$@")"
