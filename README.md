# Chem-PerturBridge_analysis

## Overview

Cross-dataset and replicate analyses of perturbation-response signatures.

### Repository branches

The original repository has three primary branches:

| Branch | Purpose |
|---|---|
| [main](https://github.com/theislab/Chem-PerturBridge_analysis/tree/main) | Repository entry point, cross-dataset analyses, summaries of replicate results, and manuscript tables/figures/supplementaries. |
| [replicates](https://github.com/theislab/Chem-PerturBridge_analysis/tree/replicates) | Replicate analyses, building on the fork at commit [46a72c5](https://github.com/elmella/Chem-PerturBridge_analysis/commit/46a72c504376d6168535ca2fed2c0d8baaeb875b). |
| [archive](https://github.com/theislab/Chem-PerturBridge_analysis/tree/archive) | Original analyses accompanying [arXiv v1](https://arxiv.org/abs/2605.31522v1). |

### Fork

The current analyses build on scripts from two branches of the [fork](https://github.com/elmella/Chem-PerturBridge_analysis):

| Branch | Purpose | Code reference |
|---|---|---|
| [reviewer-additions-baselines](https://github.com/elmella/Chem-PerturBridge_analysis/tree/reviewer-additions-baselines) | Peer-baseline comparisons and z-score representations. | [e323b36](https://github.com/elmella/Chem-PerturBridge_analysis/commit/e323b360280580093a50a910cdb0dc902755c57e) |
| [cigs-production-hardening](https://github.com/elmella/Chem-PerturBridge_analysis/tree/cigs-production-hardening) | Extended replicate analyses and optimized cross-dataset analyses. | [46a72c5](https://github.com/elmella/Chem-PerturBridge_analysis/commit/46a72c504376d6168535ca2fed2c0d8baaeb875b) |

## What Is Implemented

- Dataset loading and cross-dataset overlap matching for `l1000_phase1`, `l1000_phase2`, `tahoe`, `cigs_mce`, `novartis_batch_2500`, `vcpi_0001`, `cigs_tcm`, `vcpi_0002`, `gdpx2`, `sciplex`, `dilimap_train_val` and `op3`
- Cross-dataset DEG, direction and whole-signature agreement
- Per-cell-type retrieval with gene-overlap alignment for each query/database cell-type pair
- Global ranking against all database cell types in the unsplit workflow, or within matching cell types in the split workflow
- Representation-sweep ground-truth matching based on exact `cell_type` and `pubchem_cid`, with nearest `pert_time_h` and nearest `log(pert_dose_uM)` (falling back to linear dose distance for non-positive doses)
- Layer-wise scoring over shared layers, excluding:
  - `CI.L`, `CI.R`, `stdev.scaled`, `stdev.unscaled`, `AveExpr`
  - `adj.P.Value.across_all_contrasts`, `adj.P.Value.within_one_contrast`
- Representations, depending on the workflow:
  - logFC
  - Moderated t-statistics
  - Direction: `sign(logFC)`
  - Signed p-values: `-log10(p) * sign(logFC)`
  - P-value-derived z-scores: `sign(logFC) * Phi^-1(1 - p/2)`
  - Population z-scores at dataset and dataset × cell-type levels
  - ASHR posterior means
- Similarity and distance metrics, depending on the workflow: `cosine`, `pearson`, `spearman`, `negative_l2` and `mrrmse`
- Baseline comparisons:
  - Centroid baseline
  - Individual-peer baselines
- Bootstrap confidence intervals
- Normalized retrieval rank in `[0, 1]`, where 1 is the best
- Creation of summary tables and dose-sensitivity analyses

## Notebooks and tables

- [Notebooks](notebooks/README.md)
- [Tables and figures](tables_and_figures/README.md)
- [Supplementary files](supplementary/README.md)

## Environment

### Python

The reproducible Python environment is defined by `pyproject.toml`,
`uv.lock`, and `.python-version`. The checked-in lockfile recreates the
environment currently activated with:

```bash
source .venv/bin/activate
```

from the repository root. The `.venv/` directory itself is intentionally not
tracked.

To create or refresh the environment:

```bash
uv sync --locked
source .venv/bin/activate
python --version  # Python 3.9.18
```

If Python 3.9.18 is not available locally, install or point `uv` at a 3.9.18
interpreter first, then rerun `uv sync --locked`:

```bash
uv python install 3.9.18
uv sync --locked
```

Most batch scripts in this repository call `uv run`, so they can also be
launched without manually activating `.venv` as long as `uv sync --locked` has
been run from the repository root.

### R

ASHR preparation also requires R and `ashr`, installed separately from the
`uv` Python environment. Create the R environment once using the tested versions:

```bash
mamba create -n cpb-ashr --override-channels -c conda-forge r-base=4.2.3 r-ashr=2.2_63
```

## Run

Run the following commands from the repository root.

### Data preparation

Requires processed pseudobulk H5ADs and per-cell-type DEG results at the configured input paths.

Build the dataset summary and overlap-filtered inputs:

```bash
bash scripts/slurm/run_build_dataset_summary.sh
bash scripts/slurm/run_build_overlap_filtered_h5ads.sh
```

Prepare population z-score statistics for agreement, retrieval and the representation sweep:

```bash
bash scripts/slurm/run_precompute_population_zscore.sh
```

The statistics are stored in `results/population_zscore_stats/` and reused across these workflows.

### Agreement and retrieval

Run DEG agreement, whole-signature agreement and retrieval using logFC, moderated t-statistics and both z-score scopes. Retrieval also includes direction-based analysis.

```bash
bash scripts/slurm/run_overlap_group_rep_deg_metrics.sh
bash scripts/slurm/run_overlap_group_rep_signature_similarity.sh
bash scripts/slurm/run_overlap_group_rep_retrieval_metrics.sh
```

### Representation sweep

Compare 11 representations across six datasets using cosine, Pearson and MRRMSE.

**Preparation**

- Prepare population z-score statistics as described in [**Data preparation**](#data-preparation).
- Then prepare ASHR representations (see [Environment](#environment) for installation):

```bash
mamba activate cpb-ashr
export RSCRIPT="$CONDA_PREFIX/bin/Rscript"
bash scripts/slurm/run_precompute_ashr.sh
```

**Run**

```bash
bash scripts/slurm/run_representation_sweep.sh
```

The launcher runs matching, parallel scoring, merging and ranking-table generation.


## Slurm configuration

The launchers submit jobs with `sbatch` and wait for completion.

### Agreement and retrieval

Example: run logFC and moderated-t signature agreement:

```bash
ARRAY=0,3 CPUS_PER_TASK=8 MEM=500G \
  bash scripts/slurm/run_overlap_group_rep_signature_similarity.sh
```

Useful environment variables:

- `PARTITION`, `QOS`, `CPUS_PER_TASK`, `MEM`, `TIME`: job resources.
- `UV_BIN`: `uv` executable; default `uv`.
- `ARRAY`: select computations:
  - Agreement: `0` raw, `1` dataset z-score, `2` dataset × cell-type z-score, `3` moderated-t.
  - Retrieval: `0` raw, `1` direction, `2` moderated-t, `3` dataset z-score, `4` dataset × cell-type z-score.

### Representation sweep

`scripts/slurm/run_representation_sweep.sh` orchestrates matching, parallel scoring,
and merging with ranking-table generation. Stage 2 uses a Slurm array.

Example:

```bash
QOS=cpu_normal \
PARTITION=cpu_p \
UV_BIN=uv \
RUN_TAG=my_run \
bash scripts/slurm/run_representation_sweep.sh
```

Useful environment variables:

- `QUERY_DATASETS`, `DB_DATASETS`: comma-separated names or `all`.
- `REPRESENTATIONS`, `SKIP_REPRESENTATIONS`: representations to include or exclude.
- `METRICS`, `CELL_TYPES`: scoring metrics and cell types.
- `INCLUDE_SELF_DATASET`: `0`/`1`; default `0`.
- `RUN_TAG`: names the output folder under `results/representation_sweep/`.
- `DATASET_PATH_ARGS`: dataset-path overrides, e.g. `--dataset-path sciplex=/path`.
- `PREP_MEM`, `TASK_MEM`, `MERGE_MEM`: defaults `200G`, `100G`, `200G`.
- `PREP_TIME`, `TASK_TIME`, `MERGE_TIME`: stage time limits.


## Python CLI examples

Run from the repository root with the input data prepared.

### Retrieval

Run logFC and moderated-t retrieval using cosine and negative L2:

```bash
uv run python scripts/run_overlap_group_rep_retrieval_metrics.py \
  --datasets sciplex,tahoe \
  --compute raw,moderated-t \
  --similarities cosine negative_l2 \
  --workers 4 \
  --run-tag example
```

### Representation sweep

Run the same stages as the Slurm launcher with selected datasets and representations, scoring tasks sequentially:

```bash
set -e
SWEEP_DIR=results/representation_sweep/example

# 1. Prepare matches and tasks; select representations.
uv run python -m chem_perturbridge_analysis.retrieval.parallel_cli precompute \
  --query-datasets sciplex,tahoe \
  --db-datasets l1000_phase1,l1000_phase2 \
  --representations logFC,t,signed,z \
  --split-by-cell-type \
  --output-dir "$SWEEP_DIR" \
  --output-prefix ""

# 2. Score each task; select metrics.
SWEEP_TASK_IDS=$(uv run python -c \
  'import pandas as pd, sys; print(*pd.read_csv(sys.argv[1])["task_id"])' \
  "$SWEEP_DIR/tasks.csv")

for task_id in $SWEEP_TASK_IDS; do
  uv run python -m chem_perturbridge_analysis.retrieval.parallel_cli run-task \
    --task-file "$SWEEP_DIR/tasks.csv" \
    --task-id "$task_id" \
    --metrics cosine,mrrmse \
    --output-dir "$SWEEP_DIR/tasks" \
    --output-prefix ""
done

# 3. Merge results.
uv run python -m chem_perturbridge_analysis.retrieval.parallel_cli merge \
  --task-file "$SWEEP_DIR/tasks.csv" \
  --task-output-dir "$SWEEP_DIR/tasks" \
  --output-dir "$SWEEP_DIR" \
  --output-prefix "" \
  --strict-missing

# 4. Summarize results.
uv run python tables_and_figures/agreement_retrieval/scripts/summarize_representation_sweep.py \
  --results-dir "$SWEEP_DIR" \
  --latex-dir tables_and_figures/agreement_retrieval/results/example/representation_sweep
```
