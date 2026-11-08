#!/bin/bash

# Tables 12/13: one representation per array task, with all three similarities.
# Tasks: 0=raw, 1=direction, 2=moderated-t, 3=z-score-dataset,
#        4=z-score-dataset-cell-type. Select subsets with ARRAY=1,3 bash scripts/slurm/run_overlap_group_rep_retrieval_metrics.sh.
# Run with bash; the launcher submits the array and waits for completion.
# Z-score tasks require existing population caches; see reproducibility/table12_13.md.

set -euo pipefail

if [[ -n "${COMPUTE:-}" ]]; then
  echo "Select representations with ARRAY=0-4 bash scripts/slurm/run_overlap_group_rep_retrieval_metrics.sh instead of COMPUTE." >&2
  exit 2
fi
for argument in "$@"; do
  case "$argument" in
    --compute|--compute=*)
      echo "Select representations with ARRAY=0-4 bash scripts/slurm/run_overlap_group_rep_retrieval_metrics.sh instead of --compute." >&2
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

# Preserve the historical retrieval dataset order and the runner's metric-specific peer caps.
DATASETS="${DATASETS:-l1000_phase1,l1000_phase2,tahoe,cigs_mce,cigs_tcm,sciplex}"
computations=(raw direction moderated-t z-score-dataset z-score-dataset-cell-type)
COMPUTE="${computations[${SLURM_ARRAY_TASK_ID:-0}]:?Use ARRAY=0-4 bash scripts/slurm/run_overlap_group_rep_retrieval_metrics.sh}"
# One worker reproduces the paper bitwise; more workers are faster but score with one BLAS thread each.
WORKERS="${WORKERS:-1}"
Z_SCORE_STATS_ROOT="${Z_SCORE_STATS_ROOT:-results/population_zscore_stats}"
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

exec "${UV_BIN}" run python scripts/run_overlap_group_rep_retrieval_metrics.py "${args[@]}" "$@"
BASH
)"

# Resource overrides use the same defaults as the previous array job.
SLURM_ARGS=(
  -W
  --job-name=retrieval_t12_13
  --account="${ACCOUNT:-icb}"
  --partition="${PARTITION:-cpu_p}"
  --qos="${QOS:-cpu_normal}"
  --cpus-per-task="${CPUS_PER_TASK:-8}"
  --mem="${MEM:-500G}"
  --time="${TIME:-1-00:00:00}"
  --array="${ARRAY:-0-4}"
  --output=slurm-%x-%A_%a.out
  --error=slurm-%x-%A_%a.err
  --export=ALL
  --chdir="${REPO}"
)

sbatch "${SLURM_ARGS[@]}" --wrap="$(quote_command bash -c "${WORKER_SCRIPT}" -- "$@")"
