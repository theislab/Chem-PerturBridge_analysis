#!/usr/bin/env python3
"""Build Table 10 (updated) – DEG-restricted replicate Spearman across representations.

Reads the historical "Table 7 Replicates" sheet from the reviewer Excel update and produces
a LaTeX table at
  tables_and_figures/agreement_retrieval/results/default/table10/table_10_replicate_deg_spearman.tex
with four representation blocks (raw logFC, z-score by dataset, z-score by
dataset × cell type, and moderated t) and the extended column set introduced
in the rebuttal update:
Observed, Centroid, Obs.–Centroid, Peer mean, Obs.–Peer, Corrected percentile.

Usage
-----
  python tables_and_figures/agreement_retrieval/scripts/build_table10_replicate_latex.py

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
SHEET_NAME = "Table 7 Replicates"

DEFAULT_OUT = (
    REPO_ROOT
    / "tables_and_figures" / "agreement_retrieval"
    / "results"
    / "default"
    / "table10"
    / "table_10_replicate_deg_spearman.tex"
)

# Representation labels must match the Excel sheet exactly.
# Block order follows the cross-dataset tables (results/default/table6, results/default/table8).
REPRESENTATIONS = [
    "raw logFC",
    "z-score (dataset)",
    "z-score (dataset x cell type)",
    "moderated t",
]

METRIC = "Spearman"

LABEL = "tab:replicate_deg_lfc_spearman_bootstrap_ci"


def main() -> None:
    render(
        parse_args(__doc__, DEFAULT_OUT, SHEET_NAME),
        REPRESENTATIONS,
        LABEL,
        METRIC,
    )


if __name__ == "__main__":
    main()
