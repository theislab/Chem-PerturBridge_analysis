# Dose sensitivity

Builds Tables 4-5 and the dose-agreement figure from raw DEG and combined
logFC/moderated-t signature results.

## Run

Run from the repository root with the input data prepared:

```bash
# Prepare inputs.
uv run python scripts/run_overlap_group_rep_deg_metrics.py \
  --compute raw --run-tag raw_256

uv run python scripts/run_overlap_group_rep_signature_similarity.py \
  --compute raw,moderated-t --run-tag raw_moderated_t_256

# Build tables and figure.
uv run python tables_and_figures/dose_sensitivity/build.py
```

The DEG run must include `deg_scored_metrics.tsv`, `task_manifest.tsv` and
`task_inputs/`. The signature run must contain `signature_scored_metrics.tsv`
with both raw and moderated-t results.

Use `--deg-run` and `--signature-run` to select existing runs, and `--out-dir`
to choose an output folder.

## Outputs

Two LaTeX tables, three TSV summaries and a PDF figure are written to
`tables_and_figures/dose_sensitivity/results/default/`.
