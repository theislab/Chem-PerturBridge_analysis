# Agreement and retrieval

## Overview

Tables combine results computed locally with saved reviewer results from the fork’s [cigs-production-hardening](https://github.com/elmella/Chem-PerturBridge_analysis/tree/cigs-production-hardening) branch.

| Tables | Data source |
|---|---|
| **6 and 8** | Local DEG and signature jobs below. |
| **7** | Local direction-agreement notebook results, combined with peer results from the fork’s [table5_direction_agreement_cluster_bca_ci.tsv](https://github.com/elmella/Chem-PerturBridge_analysis/blob/90af32e2718e892f15c8bf02cf9ee068cb7fb2c7/results/parallel_cross_source/reviewer_final_summary/table5_direction_agreement_cluster_bca_ci.tsv) at commit `90af32e`, imported through the reviewer workbook. |
| **10, 11 and 14** | Replicate results from the [fork](https://github.com/elmella/Chem-PerturBridge_analysis/commit/46a72c504376d6168535ca2fed2c0d8baaeb875b) (`46a72c5`), imported through `reviewer_replicate_update.xlsx`. |
| **12 and 13** | Local retrieval results; the saved z-score observed/AUROC and peer results are imported from the fork’s [retrieval_combined_cluster_bca_ci.tsv](https://github.com/elmella/Chem-PerturBridge_analysis/blob/90af32e2718e892f15c8bf02cf9ee068cb7fb2c7/results/parallel_cross_source/reviewer_final_summary/retrieval_combined_cluster_bca_ci.tsv) at commit `90af32e`. Centroids come from local runs. |
| **9 (representation comparison)** | Local representation sweep. |

**NB!** Bootstrap confidence-interval bounds can differ between the local and fork analyses because they derive resampling seeds differently.

The mapping below relates the updated manuscript numbering to the original [arXiv paper (v1)](https://arxiv.org/abs/2605.31522v1). Repository identifiers follow the updated numbering.

| Analysis | Current manuscript table | Table in arXiv v1 | Current repository identifier |
|---|---|---|---|
| Dose mismatch and threshold sensitivity | 4, 5 | - | `dose_sensitivity/` |
| Cross-dataset DEG agreement | 6 | 4 | `table6` |
| Cross-dataset direction agreement | 7 | 5 | `table7` |
| Cross-dataset all-gene agreement | 8 | 6 | `table8` |
| Representation comparison | 9 | - | `representation_ranking_min_support` |
| Replicate DEG agreement | 10 | 7 | `table10` |
| Replicate direction agreement | 11 | 8 | `table11` |
| Cross-dataset retrieval: cosine | 12 | - | `table12_13` |
| Cross-dataset retrieval: negative L2 | 13 | 9 | `table12_13` |
| Replicate all-gene agreement | 14 | 10 | `table14` |


**NB!** Table 12 uses cosine retrieval and Table 13 uses negative L2. Their shared inputs and outputs are grouped under `table12_13/`.

## Run

### Reminder

To compute local results, run from the repository root with the input data and required caches prepared:

```bash
# Table 6
bash scripts/slurm/run_overlap_group_rep_deg_metrics.sh
# Table 8
bash scripts/slurm/run_overlap_group_rep_signature_similarity.sh
# Tables 12/13 (all five representations)
bash scripts/slurm/run_overlap_group_rep_retrieval_metrics.sh

# Representation rankings
bash scripts/slurm/run_representation_sweep.sh
```

### Format summaries

1. To reformat the saved tables into `tables_and_figures/agreement_retrieval/results/default/`:

```bash
# Selected tables, or a separate destination:
python tables_and_figures/agreement_retrieval/build.py --tables 6 8 12 13
python tables_and_figures/agreement_retrieval/build.py --out-dir /tmp/manuscript_latex
```

2. To rebuild the sweep summaries and ranking tables from saved runs in `results/representation_sweep/`:

```bash
python tables_and_figures/agreement_retrieval/scripts/summarize_representation_sweep.py --build-inputs
```

- the consolidated CSVs and ranking are written to `tables_and_figures/agreement_retrieval/data/representation_sweep/`
- formatted tables are written to `tables_and_figures/agreement_retrieval/results/default/representation_sweep/`

3. To plot heatmaps from the ready tables in `results/default/` (Tables 6, 7, 8, 12 and 13):

```bash
python tables_and_figures/agreement_retrieval/scripts/build_latex_heatmaps.py
```

Output figures and TSV values are saved in `tables_and_figures/agreement_retrieval/figures/heatmaps/`.
