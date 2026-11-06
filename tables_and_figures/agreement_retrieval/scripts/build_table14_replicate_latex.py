#!/usr/bin/env python3
"""Build Table 14 (updated) – all-gene replicate similarity across representations.

Reads the historical "Table 10 Replicates" sheet from the reviewer Excel update and produces
a LaTeX table at
  tables_and_figures/agreement_retrieval/results/default/table14/table_14_replicate_allgene.tex
with four Spearman blocks (raw logFC, both z-score variants, and moderated t)
and the extended column set introduced in the rebuttal update:
Observed, Centroid, Obs.–Centroid, Peer mean, Obs.–Peer, Corrected percentile.

Usage
-----
  python tables_and_figures/agreement_retrieval/scripts/build_table14_replicate_latex.py

  # Override the Excel source or output path:
  python ... --excel path/to/other.xlsx --out path/to/output.tex

Requires: pandas, openpyxl
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from build_excel_table_latex import parse_args, render  # noqa: E402

# Workbook sheet names retain the original numbering.
SHEET_NAME = "Table 10 Replicates"

DEFAULT_OUT = (
    REPO_ROOT
    / "tables_and_figures" / "agreement_retrieval"
    / "results"
    / "default"
    / "table14"
    / "table_14_replicate_allgene.tex"
)

# Spearman only (cosine rows exist in the sheet but are not included here).
# Representation labels must match the Excel sheet exactly.
# Block order follows the cross-dataset tables (results/default/table6, results/default/table8).
REPRESENTATIONS = [
    "raw logFC, Spearman",
    "z-score (dataset), Spearman",
    "z-score (dataset x cell type), Spearman",
    "moderated t, Spearman",
]

METRIC = "Spearman"

LABEL = "tab:replicate_logfc_bootstrap_ci"


def main() -> None:
    render(
        parse_args(__doc__, DEFAULT_OUT, SHEET_NAME),
        REPRESENTATIONS,
        LABEL,
        METRIC,
    )


if __name__ == "__main__":
    main()
