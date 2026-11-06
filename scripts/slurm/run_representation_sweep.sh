#!/bin/bash
# Submit matching, per-representation scoring, then merged summaries and tables.
# Prepare ASHR sidecars and population z-score caches with their own launchers first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

UV_BIN="${UV_BIN:-uv}"
RUN_TAG="${RUN_TAG:-production}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/representation_sweep/${RUN_TAG}}"
TASK_OUTPUT_DIR="${TASK_OUTPUT_DIR:-${OUTPUT_DIR}/tasks}"
LOGS_DIR="${LOGS_DIR:-${REPO_ROOT}/logs/representation_sweep/${RUN_TAG}}"
LATEX_DIR="${LATEX_DIR:-${REPO_ROOT}/tables_and_figures/agreement_retrieval/results/default/representation_sweep}"
DATA_ROOT="${DATA_ROOT:-/lustre/groups/ml01/workspace/olga.novitskaia/data_updated}"
CIGS_SUBPATH="${CIGS_SUBPATH:-deg_data/group_rep/full/qc_false/filter_min_cells_0/results}"
ASHR_SIDECAR_ROOT="${ASHR_SIDECAR_ROOT:-${REPO_ROOT}/results/ashr_sidecars}"
Z_SCORE_STATS_ROOT="${Z_SCORE_STATS_ROOT:-${REPO_ROOT}/results/population_zscore_stats}"

QUERY_DATASETS="${QUERY_DATASETS:-sciplex,tahoe,l1000_phase1,l1000_phase2,cigs_mce,cigs_tcm}"
DB_DATASETS="${DB_DATASETS:-sciplex,tahoe,l1000_phase1,l1000_phase2,cigs_mce,cigs_tcm}"
REPRESENTATIONS="${REPRESENTATIONS:-B,P.Value,logFC,per_gene_population_zscore(logFC),sign(logFC),signed_-log10(adj.p)*sign(logFC),signed_-log10(p)*sign(logFC),t,z=sign(logFC)*Phi^-1(1-p/2),ashr_posterior_mean(logFC)_cell_line,per_gene_population_zscore_dataset}"
METRICS="${METRICS:-cosine,pearson,mrrmse}"

QOS="${QOS:-cpu_normal}"
PARTITION="${PARTITION:-cpu_p}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
NUMPY_THREADS="${NUMPY_THREADS:-${CPUS_PER_TASK}}"
PREP_TIME="${PREP_TIME:-24:00:00}"
TASK_TIME="${TASK_TIME:-24:00:00}"
MERGE_TIME="${MERGE_TIME:-24:00:00}"
PREP_MEM="${PREP_MEM:-200G}"
TASK_MEM="${TASK_MEM:-100G}"
MERGE_MEM="${MERGE_MEM:-200G}"

if [[ "${REPRESENTATIONS}" == all || "${REPRESENTATIONS}" == *ashr* ]]; then
    if [ ! -d "${ASHR_SIDECAR_ROOT}" ]; then
        echo "> ERROR: ASHR sidecars missing: ${ASHR_SIDECAR_ROOT}"
        echo "> Run bash scripts/slurm/run_precompute_ashr.sh first."
        exit 1
    fi
fi
if [[ "${REPRESENTATIONS}" == all || "${REPRESENTATIONS}" == *population_zscore* ]]; then
    for cache_dir in dataset_wide sciplex tahoe l1000_phase1 l1000_phase2 cigs_mce cigs_tcm; do
        if [ ! -d "${Z_SCORE_STATS_ROOT}/${cache_dir}" ]; then
            echo "> ERROR: population z-score caches missing: ${Z_SCORE_STATS_ROOT}/${cache_dir}"
            echo "> Run bash scripts/slurm/run_precompute_population_zscore.sh first."
            exit 1
        fi
    done
fi

# Keep the original dataset insertion order; overrides replace these paths.
DATASET_ARGS=(
    --dataset-path "sciplex=${DATA_ROOT}/sciplex/deg_data/group_rep/full/qc_false/filter_min_cells_10/results"
    --dataset-path "tahoe=${DATA_ROOT}/tahoe/deg_data/group_rep/full/qc_false/filter_min_cells_50/results"
    --dataset-path "l1000_phase1=${DATA_ROOT}/l1000_phase1/deg_data/group_rep/full/qc_false/filter_min_cells_0/results"
    --dataset-path "l1000_phase2=${DATA_ROOT}/l1000_phase2/deg_data/group_rep/full/qc_false/filter_min_cells_0/results"
    --dataset-path "cigs_mce=${DATA_ROOT}/cigs_mce/${CIGS_SUBPATH}"
    --dataset-path "cigs_tcm=${DATA_ROOT}/cigs_tcm/${CIGS_SUBPATH}"
)
if [ -n "${DATASET_PATH_ARGS:-}" ]; then
    # Parse quoted paths without executing shell expressions from this variable.
    parsed_overrides="$(python3 -c 'import shlex, sys; print("\n".join(shlex.split(sys.argv[1])))' "${DATASET_PATH_ARGS}")"
    mapfile -t override_args <<< "${parsed_overrides}"
    DATASET_ARGS+=("${override_args[@]}")
fi

# sbatch --wrap uses /bin/sh. Single-quote each static argument so parentheses,
# spaces and shell metacharacters in representation names or paths stay literal.
quote_command() {
    local arg
    for arg in "$@"; do
        printf "'%s' " "${arg//\'/\'\\\'\'}"
    done
}

CLI=("${UV_BIN}" run python -m chem_perturbridge_analysis.retrieval.parallel_cli)
CACHE_ENV=(env "ASHR_SIDECAR_ROOT=${ASHR_SIDECAR_ROOT}" "Z_SCORE_STATS_ROOT=${Z_SCORE_STATS_ROOT}")
TASK_FILE="${OUTPUT_DIR}/tasks.csv"
PRECOMPUTE_ARGS=(precompute --output-dir "${OUTPUT_DIR}" --output-prefix ""
    --query-datasets "${QUERY_DATASETS}" --db-datasets "${DB_DATASETS}"
    --representations "${REPRESENTATIONS}" "${DATASET_ARGS[@]}" --verbose)
RUN_TASK_ARGS=(run-task --task-file "${TASK_FILE}" --output-dir "${TASK_OUTPUT_DIR}"
    --output-prefix "" --metrics "${METRICS}" "${DATASET_ARGS[@]}" --verbose)
MERGE_ARGS=(merge --task-file "${TASK_FILE}" --task-output-dir "${TASK_OUTPUT_DIR}"
    --output-dir "${OUTPUT_DIR}" --output-prefix "" --verbose)
if [ -n "${SKIP_REPRESENTATIONS:-}" ]; then
    PRECOMPUTE_ARGS+=(--skip-representations "${SKIP_REPRESENTATIONS}")
fi
if [ -n "${CELL_TYPES:-}" ]; then
    PRECOMPUTE_ARGS+=(--cell-types "${CELL_TYPES}")
    RUN_TASK_ARGS+=(--cell-types "${CELL_TYPES}")
fi
if [ "${INCLUDE_SELF_DATASET:-0}" = 1 ]; then
    PRECOMPUTE_ARGS+=(--include-self-dataset)
fi
if [ "${SPLIT_BY_CELL_TYPE:-1}" = 1 ]; then
    PRECOMPUTE_ARGS+=(--split-by-cell-type)
fi
if [ "${STRICT_MISSING:-1}" = 1 ]; then
    MERGE_ARGS+=(--strict-missing)
fi

PRECOMPUTE_CMD="$(quote_command "${CACHE_ENV[@]}" "${CLI[@]}" "${PRECOMPUTE_ARGS[@]}")"
RUN_TASK_CMD="$(quote_command "${CACHE_ENV[@]}" "OMP_NUM_THREADS=${NUMPY_THREADS}" \
    "OPENBLAS_NUM_THREADS=${NUMPY_THREADS}" "MKL_NUM_THREADS=${NUMPY_THREADS}" \
    "NUMEXPR_NUM_THREADS=${NUMPY_THREADS}" "${CLI[@]}" "${RUN_TASK_ARGS[@]}")"' --task-id "${SLURM_ARRAY_TASK_ID}"'
MERGE_CMD="$(quote_command "${CLI[@]}" "${MERGE_ARGS[@]}") && $(quote_command \
    "${UV_BIN}" run python "${REPO_ROOT}/tables_and_figures/agreement_retrieval/scripts/summarize_representation_sweep.py" \
    --results-dir "${OUTPUT_DIR}" --latex-dir "${LATEX_DIR}")"

mkdir -p "${OUTPUT_DIR}" "${TASK_OUTPUT_DIR}" "${LOGS_DIR}"
SLURM_ARGS=(-W -n 1 --chdir="${REPO_ROOT}" --qos="${QOS}"
    --partition="${PARTITION}" --cpus-per-task="${CPUS_PER_TASK}")

echo "> Stage 1/3: Precompute true matches and task matrix"
sbatch "${SLURM_ARGS[@]}" -J representation_precompute -t "${PREP_TIME}" --mem="${PREP_MEM}" \
    -e "${LOGS_DIR}/precompute.%j.err" -o "${LOGS_DIR}/precompute.%j.out" \
    --wrap="${PRECOMPUTE_CMD}"

if [ ! -f "${TASK_FILE}" ]; then
    echo "> ERROR: task file not found: ${TASK_FILE}"
    exit 1
fi
N_TASKS=$(($(wc -l < "${TASK_FILE}") - 1))
if [ "${N_TASKS}" -le 0 ]; then
    echo "> No retrieval tasks to run (task matrix is empty)."
    exit 0
fi

echo "> Stage 2/3: Run ${N_TASKS} retrieval tasks in a Slurm array"
sbatch "${SLURM_ARGS[@]}" -J representation_task -t "${TASK_TIME}" --mem="${TASK_MEM}" \
    --array=1-"${N_TASKS}" -e "${LOGS_DIR}/task.%A_%a.err" -o "${LOGS_DIR}/task.%A_%a.out" \
    --wrap="${RUN_TASK_CMD}"

echo "> Stage 3/3: Merge task outputs and build ranking tables"
sbatch "${SLURM_ARGS[@]}" -J representation_merge -t "${MERGE_TIME}" --mem="${MERGE_MEM}" \
    -e "${LOGS_DIR}/merge.%j.err" -o "${LOGS_DIR}/merge.%j.out" --wrap="${MERGE_CMD}"
echo "> Done. Results: ${OUTPUT_DIR}; LaTeX: ${LATEX_DIR}"
