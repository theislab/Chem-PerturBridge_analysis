#!/usr/bin/env python3
"""Build Table 11 (updated) – DEG-restricted replicate direction agreement.

Reads the historical "Table 8 Direction" sheet from the reviewer Excel update and produces
a LaTeX table at
  tables_and_figures/agreement_retrieval/results/default/table11/table_11_replicate_direction.tex
with the extended column set introduced in the rebuttal update:
Observed, Centroid, Obs.–Centroid, Peer mean, Obs.–Peer, Corrected percentile.

Direction agreement is kept on raw logFC sign only (a per-gene z-score subtracts
a population mean, which can flip signs, so z-score direction agreement no longer
reflects biological direction).

Usage
-----
  python tables_and_figures/agreement_retrieval/scripts/build_table11_direction_latex.py

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
SHEET_NAME = "Table 8 Direction"

DEFAULT_OUT = (
    REPO_ROOT
    / "tables_and_figures" / "agreement_retrieval"
    / "results"
    / "default"
    / "table11"
    / "table_11_replicate_direction.tex"
)

# Only raw logFC sign is reported for direction agreement.
REPRESENTATIONS = [
    "raw logFC sign",
]

METRIC = "direction"

LABEL = "tab:replicate_direction_agreement_bootstrap_ci"


def main() -> None:
    render(
        parse_args(__doc__, DEFAULT_OUT, SHEET_NAME),
        REPRESENTATIONS,
        LABEL,
        METRIC,
    )


if __name__ == "__main__":
    main()
