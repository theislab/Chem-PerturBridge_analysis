#!/bin/bash
# Precompute ashr sidecar layers for the six retrieval datasets.
#
# One Slurm array shard per chunk of cell lines. Shards are independent, so a
# failed shard can be resubmitted on its own; already-written sidecars are
# skipped unless OVERWRITE=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

UV_BIN="${UV_BIN:-uv}"
QOS="${QOS:-cpu_normal}"
PARTITION="${PARTITION:-cpu_p}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
TIME="${TIME:-24:00:00}"
MEM="${MEM:-64G}"
N_SHARDS="${N_SHARDS:-40}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./results/ashr_sidecars}"
LOGS_DIR="${LOGS_DIR:-./logs/ashr_precompute}"
DATASETS="${DATASETS:-all}"
CELL_TYPES="${CELL_TYPES:-}"
PRIOR_GENES="${PRIOR_GENES:-all}"
RSCRIPT="${RSCRIPT:-/home/icb/olga.novitskaia/deg_venv/lib/R/bin/Rscript}"
OVERWRITE="${OVERWRITE:-0}"

# The R handoff writes two float64 copies of every finite estimate, which for a
# tahoe cell line is ~500MB per call. Node-local /tmp is only ~9GB and is shared
# by every co-scheduled shard, which is what exhausted it on the first run, so
# keep scratch on lustre instead.
SCRATCH_ROOT="${SCRATCH_ROOT:-/lustre/groups/ml01/workspace/olga.novitskaia/tmp/ashr_precompute}"

mkdir -p "${OUTPUT_ROOT}" "${LOGS_DIR}" "${SCRATCH_ROOT}"

CMD="export TMPDIR=${SCRATCH_ROOT} && ${UV_BIN} run python scripts/precompute_ashr_layers.py \
    --output-root ${OUTPUT_ROOT} \
    --datasets ${DATASETS} \
    --prior-genes ${PRIOR_GENES} \
    --rscript ${RSCRIPT} \
    --shard \$((SLURM_ARRAY_TASK_ID - 1)) \
    --n-shards ${N_SHARDS} \
    --verbose"
if [ -n "${CELL_TYPES}" ]; then
    CMD="${CMD} --cell-types ${CELL_TYPES}"
fi
if [ "${OVERWRITE}" = "1" ]; then
    CMD="${CMD} --overwrite"
fi

echo "> Precomputing ashr sidecars into ${OUTPUT_ROOT} across ${N_SHARDS} shards"
sbatch -W -J ashr_precompute \
    -t "${TIME}" \
    -n 1 \
    --array=1-"${N_SHARDS}" \
    --qos="${QOS}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    -e "${LOGS_DIR}/ashr_precompute.%A_%a.err" \
    -o "${LOGS_DIR}/ashr_precompute.%A_%a.out" \
    --wrap="${CMD}"

echo "> Done. Sidecars in ${OUTPUT_ROOT}/<dataset>/<cell_type>.h5"
