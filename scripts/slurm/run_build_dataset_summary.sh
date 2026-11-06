#!/bin/bash
# Build the dataset summary in a Slurm job and wait for completion.
# Pass Python CLI options directly, for example --datasets sciplex tahoe --jobs 2.
set -euo pipefail

REPO="$(cd "${REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
cd "${REPO}"

UV_BIN="${UV_BIN:-uv}"
ACCOUNT="${ACCOUNT:-icb}"
PARTITION="${PARTITION:-cpu_p}"
QOS="${QOS:-cpu_normal}"
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
MEM="${MEM:-64G}"
TIME="${TIME:-1-00:00:00}"
LOGS_DIR="${LOGS_DIR:-./logs/dataset_summary}"
mkdir -p "${LOGS_DIR}"

# sbatch --wrap uses /bin/sh; keep every argument literal, including paths.
quote_command() {
    local arg
    for arg in "$@"; do
        printf "'%s' " "${arg//\'/\'\\\'\'}"
    done
}

CMD=("${UV_BIN}" run python scripts/build_dataset_summary.py)
if [[ -n "${DATA_ROOT:-}" ]]; then
    CMD+=(--data-root "${DATA_ROOT}")
fi
CMD+=("$@")

sbatch -W -J dataset_summary -n 1 \
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
