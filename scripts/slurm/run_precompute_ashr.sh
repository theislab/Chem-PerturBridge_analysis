#!/bin/bash
# Precompute ASHR sidecars in a Slurm array and wait for completion.
# Set N_SHARDS for a different split; ARRAY can select tasks to retry without
# changing that split (for example, ARRAY=23,32 keeps the default 40 shards).
set -euo pipefail

REPO_ROOT="$(cd "${REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "${REPO_ROOT}"

ACCOUNT="${ACCOUNT:-icb}"
QOS="${QOS:-cpu_normal}"
PARTITION="${PARTITION:-cpu_p}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
TIME="${TIME:-24:00:00}"
MEM="${MEM:-64G}"
N_SHARDS="${N_SHARDS:-40}"
ARRAY="${ARRAY:-1-${N_SHARDS}}"

UV_BIN="${UV_BIN:-uv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/ashr_sidecars}"
LOGS_DIR="${LOGS_DIR:-./logs/ashr_precompute}"
DATASETS="${DATASETS:-all}"
CELL_TYPES="${CELL_TYPES:-}"
PRIOR_GENES="${PRIOR_GENES:-all}"
RSCRIPT="${RSCRIPT:-/home/icb/olga.novitskaia/deg_venv/lib/R/bin/Rscript}"
OVERWRITE="${OVERWRITE:-0}"

# ASHR's R handoff can exhaust node-local /tmp; keep scratch on lustre.
SCRATCH_ROOT="${SCRATCH_ROOT:-/lustre/groups/ml01/workspace/olga.novitskaia/tmp/ashr_precompute}"
mkdir -p "${OUTPUT_ROOT}" "${LOGS_DIR}" "${SCRATCH_ROOT}"

# sbatch --wrap uses /bin/sh; keep every argument literal, including paths.
quote_command() {
    local arg
    for arg in "$@"; do
        printf "'%s' " "${arg//\'/\'\\\'\'}"
    done
}

# Resolve the zero-based shard on the allocated node, after Slurm sets its ID.
SHARD_COMMAND='task_id=${SLURM_ARRAY_TASK_ID:?Slurm array task ID required}
case "$task_id" in
    *[!0-9]*) echo "ERROR: array task ID must be an integer." >&2; exit 1 ;;
esac
if [ "$task_id" -lt 1 ] || [ "$task_id" -gt "$1" ]; then
    echo "ERROR: array task $task_id is outside 1-$1." >&2
    exit 1
fi
shift
exec "$@" --shard "$((task_id - 1))"'
CMD=(bash -c "${SHARD_COMMAND}" -- "${N_SHARDS}"
    env "TMPDIR=${SCRATCH_ROOT}"
    "${UV_BIN}" run python scripts/precompute_ashr_layers.py
    --output-root "${OUTPUT_ROOT}"
    --datasets "${DATASETS}"
    --prior-genes "${PRIOR_GENES}"
    --rscript "${RSCRIPT}"
    --n-shards "${N_SHARDS}"
    --verbose)
if [ -n "${CELL_TYPES}" ]; then
    CMD+=(--cell-types "${CELL_TYPES}")
fi
if [ "${OVERWRITE}" = "1" ]; then
    CMD+=(--overwrite)
fi

echo "> Precomputing ashr sidecars into ${OUTPUT_ROOT} across ${N_SHARDS} shards"
ARRAY_JOB_ID="$(sbatch --parsable -J ashr_precompute \
    -t "${TIME}" \
    -n 1 \
    --array="${ARRAY}" \
    --account="${ACCOUNT}" \
    --qos="${QOS}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --export=ALL \
    --chdir="${REPO_ROOT}" \
    -e "${LOGS_DIR}/ashr_precompute.%A_%a.err" \
    -o "${LOGS_DIR}/ashr_precompute.%A_%a.out" \
    --wrap="$(quote_command "${CMD[@]}")")"
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"
echo "> Submitted array ${ARRAY_JOB_ID}; waiting for all shards"

# sbatch -W on an array can return before every task ends, so wait on a
# dependency. Slurm cancels this job if any task fails, but sbatch -W still
# exits 0 for it, so task states are checked afterwards.
sbatch -W -J "wait_${ARRAY_JOB_ID}" --account="${ACCOUNT}" --partition="${PARTITION}" \
    --qos="${QOS}" --cpus-per-task=1 --mem=100M --time=00:05:00 --output=/dev/null \
    --dependency=afterok:"${ARRAY_JOB_ID}" --kill-on-invalid-dep=yes --wrap=true > /dev/null || true
failed_tasks="$(sacct -j "${ARRAY_JOB_ID}" -X -n -P -o JobID,State | awk -F'|' '$2 != "COMPLETED"')"
if [ -n "${failed_tasks}" ]; then
    echo "> ERROR: shards in array ${ARRAY_JOB_ID} did not complete:" >&2
    echo "${failed_tasks}" >&2
    echo "> Check ${LOGS_DIR}/ashr_precompute.${ARRAY_JOB_ID}_*.err" >&2
    exit 1
fi

echo "> Done. Sidecars in ${OUTPUT_ROOT}/<dataset>/<cell_type>.h5"
