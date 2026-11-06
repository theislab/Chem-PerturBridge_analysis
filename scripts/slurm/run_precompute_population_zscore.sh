#!/bin/bash
# Fit population mean/SD caches for both z-score scopes in a Slurm job.
set -euo pipefail

REPO="$(cd "${REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "${REPO}"

ACCOUNT="${ACCOUNT:-icb}"
PARTITION="${PARTITION:-cpu_p}"
QOS="${QOS:-cpu_normal}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEM="${MEM:-500G}"
TIME="${TIME:-1-00:00:00}"
LOGS_DIR="${LOGS_DIR:-./logs/population_zscore}"

UV_BIN="${UV_BIN:-uv}"
DATA="${DATA:-/lustre/groups/ml01/workspace/olga.novitskaia/data_updated}"
CACHE="${CACHE:-${REPO}/results/population_zscore_stats}"
WORKERS="${WORKERS:-${CPUS_PER_TASK}}"
SCOPE="${SCOPE:-both}"
mkdir -p "${LOGS_DIR}" "${CACHE}"

# sbatch --wrap uses /bin/sh; keep every argument literal, including paths.
quote_command() {
    local arg
    for arg in "$@"; do
        printf "'%s' " "${arg//\'/\'\\\'\'}"
    done
}

DATASETS="${DATASETS:-l1000_phase1,l1000_phase2,tahoe,sciplex,cigs_mce,cigs_tcm}"
declare -A DATASET_SUBPATHS=(
    [l1000_phase1]=filter_min_cells_0 [l1000_phase2]=filter_min_cells_0
    [tahoe]=filter_min_cells_50 [sciplex]=filter_min_cells_10
    [cigs_mce]=filter_min_cells_0 [cigs_tcm]=filter_min_cells_0
)
DATASET_ARGS=()
IFS=, read -ra selected_datasets <<< "${DATASETS}"
for name in "${selected_datasets[@]}"; do
    if [ -z "${DATASET_SUBPATHS[${name}]:-}" ]; then
        echo "> ERROR: unknown dataset ${name}; choose from ${!DATASET_SUBPATHS[*]}" >&2
        exit 1
    fi
    DATASET_ARGS+=(--dataset-dir
        "${name}=${DATA}/${name}/deg_data/group_rep/full/qc_false/${DATASET_SUBPATHS[${name}]}/results")
done

CMD=("${UV_BIN}" run python scripts/precompute_population_zscore.py
    "${DATASET_ARGS[@]}"
    --scope "${SCOPE}"
    --row-chunk-size 1024
    --workers "${WORKERS}"
    --cache-root "${CACHE}"
    --qc-output "${CACHE}/precompute_qc.tsv")

echo "> Precomputing population z-score statistics: datasets=${DATASETS}, workers=${WORKERS}, scope=${SCOPE}"
sbatch -W -J population_zscore_precompute \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --export=ALL \
    --chdir="${REPO}" \
    --output="${LOGS_DIR}/%x.%j.out" \
    --error="${LOGS_DIR}/%x.%j.err" \
    --wrap="$(quote_command "${CMD[@]}")"
echo "> Done. Population statistics in ${CACHE}"
