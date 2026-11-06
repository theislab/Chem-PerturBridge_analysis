#!/usr/bin/env python3
"""Render manuscript LaTeX tables from the saved inputs in this folder."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BUILDERS = {
    "6": ("build_table6_combined_latex.py", "table6/table_6_deg_spearman_combined.tex"),
    "7": ("build_table7_direction_latex.py", "table7/table_7_direction_combined.tex"),
    "8": ("build_table8_combined_latex.py", "table8/table_8_allgene_spearman_combined.tex"),
    "10": ("build_table10_replicate_latex.py", "table10/table_10_replicate_deg_spearman.tex"),
    "11": ("build_table11_direction_latex.py", "table11/table_11_replicate_direction.tex"),
    "12": ("build_table12_13_retrieval_latex.py", "table12_13"),
    "13": ("build_table12_13_retrieval_latex.py", "table12_13"),
    "14": ("build_table14_replicate_latex.py", "table14/table_14_replicate_allgene.tex"),
    "sweep": ("summarize_representation_sweep.py", "representation_sweep"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="+", choices=BUILDERS, default=list(BUILDERS))
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "default",
        help="Output directory (default: tables_and_figures/agreement_retrieval/results/default).",
    )
    parser.add_argument(
        "--sweep-results", type=Path, default=ROOT / "data" / "representation_sweep",
        help="Sweep folder containing summary_overall.csv and truth_summary.csv.",
    )
    args = parser.parse_args()
    for table in dict.fromkeys(args.tables):
        script, destination = BUILDERS[table]
        output = args.out_dir / destination
        command = [sys.executable, "-B", str(ROOT / "scripts" / script)]
        if table == "sweep":
            command += ["--results-dir", str(args.sweep_results), "--latex-dir", str(output)]
        elif table in ("12", "13"):
            similarity = "cosine" if table == "12" else "negative_l2"
            command += ["--out-dir", str(output), "--similarity", similarity]
        else:
            command += ["--out", str(output)]
        print(f"Building {table} → {output}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
