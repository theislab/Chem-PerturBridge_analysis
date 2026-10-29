# Chem-PerturBridge_analysis

Cross-dataset retrieval benchmark for perturbation-response signatures.

## Reproducibility
### Environment

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

### Execution
1. `scripts/build_dataset_summary.py` for table 1 content
2. `scripts/overlap_filtered_h5ads.py` for getting pseudobulks of overlapping samples (excluding those overlapping only across l1000 phases as there are so many of these that they would dominate and differ from the other considered significantly)
3. `notebooks/plot_overlap_heatmaps.ipynb` for cross-dataset overlap plots in fig1. In each dataset pair, we skip contexts that have less than 10 compounds shared between the two.
4. `notebooks/overlap_group_rep_deg_metrics.ipynb`, `notebooks/overlap_group_rep_signature_similarity.ipynb` and `notebooks/overlap_group_rep_retrieval_metrics.ipynb` to plot cross-dataset agreement.
5. `SKIP_EXISTING_DATASETS=1 EXISTING_RESULTS_DIR=./results/replicate_signature_similarity_sep_rep OUTPUT_DIR=./results/replicate_signature_similarity_sep_rep_combined COMPUTE_BASELINE_METRICS=1 COMPUTE_DEG_METRICS=1 scripts/slurm/run_replicate_signature_similarity.sh`, `notebooks/replicate_signature_similarity.ipynb` and `notebooks/replicate_deg_metrics.ipynb` to compute the cross-replicate agreement.

## What Is Implemented

- Dataset loading for `sciplex`, `tahoe`, `l1000_phase1`, `l1000_phase2` using paths from `notebooks/load_data.ipynb`.
- Per-cell-type retrieval with gene-overlap alignment for each query/db cell-type pair.
- Global ranking against all db cell types (including non-matching cell types) for each query sample.
- Ground-truth matching based on exact `cell_type` + `pubchem_cid`, with nearest `pert_time_h` and nearest `log(pert_dose_uM)` (falling back to linear dose distance for non-positive doses).
- Layer-wise scoring over all shared layers except:
  - `CI.L`, `CI.R`, `stdev.scaled`, `stdev.unscaled`, `AveExpr`
  - `adj.P.Value.across_all_contrasts`, `adj.P.Value.within_one_contrast`
- Additional derived representations:
  - `-log10(p) * sign(logFC)`
  - `sign(logFC) * Phi^-1(1 - p/2)`
- Similarity/distance metrics:
  - `cosine`, `pearson`, `spearman`, `mrrmse`
- Rank normalization to `[0, 1]`:
  - `rank_normalized` (0 is best)
  - `retrieval_score` (1 is best)

## Run

```bash
python -m chem_perturbridge_analysis.retrieval.cli --verbose
```

Outputs are written to:

- `results/cross_dataset_retrieval_detail.csv`
- `results/cross_dataset_retrieval_summary_by_cell_type.csv`
- `results/cross_dataset_retrieval_summary_overall.csv`

Verbose mode now logs progress at pair, cell-type, and representation levels (including elapsed time and rows produced).

## Useful Options

```bash
python -m chem_perturbridge_analysis.retrieval.cli \
  --query-datasets sciplex,tahoe \
  --db-datasets l1000_phase1,l1000_phase2 \
  --cell-types CVCL_0002,CVCL_0063 \
  --representations logFC,P.Value,signed,z \
  --metrics cosine,mrrmse \
  --skip-representations P.Value \
  --output-prefix my_run \
  --verbose
```

Limit compute for faster runs:

```bash
python -m chem_perturbridge_analysis.retrieval.cli \
  --representations logFC,signed \
  --metrics cosine
```

Override data paths:

```bash
python -m chem_perturbridge_analysis.retrieval.cli \
  --dataset-path sciplex=/my/sciplex/results \
  --dataset-path l1000_phase1=/my/l1000_phase1/results
```

## Visualization

Use `notebooks/visualize_retrieval_results.ipynb` to visualize retrieval outputs (`*_detail.csv`, `*_summary_by_cell_type.csv`, `*_summary_overall.csv`) for any run prefix (default in notebook: `my_run_fast`).

## Parallel Slurm Workflow

For large runs, use the parallel pipeline:

1. Precompute true cell line-drug matches and generate task matrix (`pair x representation`).
2. Run one task per row of the matrix (ideal for Slurm array jobs).
3. Merge all task outputs into the final CSVs.

Precompute now also writes one AnnData per dataset pair with matches:

- `results/<run_dir>/<prefix>_pair_matches/<dataset_a>__<dataset_b>_matches.h5ad`
- each file contains matched samples from both datasets on shared genes and includes
  directional obs columns `<dataset_a>_<dataset_b>_match_id` and `<dataset_b>_<dataset_a>_match_id`.

### Python CLI Stages

```bash
# Stage 1: precompute truth matches + task matrix
python -m chem_perturbridge_analysis.retrieval.parallel_cli precompute \
  --output-dir results/full \
  --output-prefix my_run_fast \
  --query-datasets all \
  --db-datasets all \
  --representations all \
  --verbose

# Stage 2: run one task (task_id usually from Slurm array index)
python -m chem_perturbridge_analysis.retrieval.parallel_cli run-task \
  --task-file results/full/my_run_fast_tasks.csv \
  --task-id 1 \
  --output-dir results/full/tasks \
  --output-prefix my_run_fast \
  --metrics all \
  --verbose

# Stage 3: merge all task outputs
python -m chem_perturbridge_analysis.retrieval.parallel_cli merge \
  --task-file results/full/my_run_fast_tasks.csv \
  --task-output-dir results/full/tasks \
  --output-dir results/full \
  --output-prefix my_run_fast \
  --verbose
```

### Slurm Launcher

`scripts/slurm/run_retrieval_parallel.sh` orchestrates all three stages with `sbatch -W` and an array for stage 2.

Example:

```bash
QOS=cpu_normal \
PARTITION=cpu_p \
UV_BIN=uv \
OUTPUT_PREFIX=my_run_fast \
bash scripts/slurm/run_retrieval_parallel.sh
```

Useful environment variables:

- `QUERY_DATASETS`, `DB_DATASETS` (comma-separated names or `all`)
- `REPRESENTATIONS`, `SKIP_REPRESENTATIONS`
- `METRICS`
- `CELL_TYPES`
- `INCLUDE_SELF_DATASET` (`0`/`1`)
- `UV_BIN` (default: `uv`)
- `DATASET_PATH_ARGS` (e.g. `--dataset-path sciplex=/path --dataset-path tahoe=/path`)
- `TASK_MEM`, `PREP_MEM`, `MERGE_MEM` (defaults set to `200G`)
