"""Peer-baseline heatmaps read straight from the LaTeX tables, into heatmaps/.

Companion to build_peer_baseline_heatmaps.py, which reads the intermediate TSVs
and workbooks. This one reads the compiled LaTeX of Tables 6, 7, 8, 12 and 13,
so a figure cannot disagree with the table the manuscript actually prints:

  table_6                            results/default/table6/table_6_deg_spearman_combined.tex
  table_7                            results/default/table7/table_7_direction_combined.tex
  table_8                            results/default/table8/table_8_allgene_spearman_combined.tex
  table_13_logfc_negative_l2          results/default/table12_13/table_13_retrieval_negative_l2_combined.tex
  table_12_logfc_cosine               results/default/table12_13/table_12_retrieval_cosine_combined.tex
  table_13_signed_logfc_negative_l2   the negative L2 file, sign(logFC) block
  table_12_signed_logfc_cosine        the cosine file, sign(logFC) block

Two differences from heatmaps_reference/:

  * Both retrieval similarities are drawn -- negative L2 (Table 13) and cosine
    (Table 12) --
    each in its own folder, with the filenames and panel titles of every other
    folder, so the two sets are read side by side rather than in one figure.
    Tables 6 and 8 have no similarity axis; they are Spearman correlations.
  * The CIGS-TCM column is dropped from the figures. Its row stays, so the
    CIGS-MCE x CIGS-TCM cell -- the only pair CIGS-TCM has -- is still drawn;
    under the masked upper triangle the column itself carries nothing. The
    matrix TSVs keep every dataset either way. --drop-columns '' keeps it.

Four panels per folder, as before: observed, the source and target peer
baselines, and the centroid baseline, as a symmetric dataset x dataset matrix
with the upper triangle masked. The colour scale is pooled over all four panels
of a block, as in heatmaps_reference/, and over both retrieval similarities, so the
two folders of an arm can be compared cell for cell. --scale panel pools each
panel alone, which the retrieval arms read better under: their centroid sits near
1.0 and the other three near chance, so one scale leaves the rest little
contrast.

The LaTeX carries three decimals, the annotation precision of the figures, so
the matrices written here are rounded where heatmaps_reference/ holds full precision.
--compare-existing reports the gap against heatmaps_reference/ for the arms it covers.

Only the raw logFC block is drawn unless --blocks says otherwise (sign(logFC) for
the signed arms); every block is written to the values TSV regardless.

  python tables_and_figures/agreement_retrieval/scripts/build_latex_heatmaps.py
  python tables_and_figures/agreement_retrieval/scripts/build_latex_heatmaps.py --arms table_6 --blocks all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Matches build_peer_baseline_heatmaps.py, whose figures these sit beside: the
# seaborn "notebook" context uses larger fonts than the matplotlib defaults.
sns.set_theme(style="whitegrid")

REPO_ROOT = Path(__file__).resolve().parents[3]
LATEX_DIR = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "results" / "default"
DEFAULT_OUT_DIR = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "figures" / "heatmaps"
EXISTING_DIR = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "figures" / "heatmaps_reference"

DISPLAY_LABELS = {
    "cigs_mce": "CIGS-MCE",
    "cigs_tcm": "CIGS-TCM",
    "l1000_phase1": "L1000 Phase I",
    "l1000_phase2": "L1000 Phase II",
    "sciplex": "sci-Plex",
    "tahoe": "Tahoe-100M",
}
DATASET_ORDER = [
    "l1000_phase1",
    "l1000_phase2",
    "tahoe",
    "cigs_mce",
    "cigs_tcm",
    "sciplex",
]
# The dataset labels contain single hyphens, so the doubled one is what makes
# splitting a pair label safe.
PAIR_SEPARATOR = "--"

# LaTeX header text -> canonical panel. The four are named identically in all
# input tables, so the header locates them without hard-coded column numbers.
PANEL_HEADERS = {
    "Observed": "observed",
    "Centroid": "centroid",
    "Source peer": "source_peer",
    "Target peer": "target_peer",
}
# canonical panel -> (file stem, figure title)
PANELS = {
    "observed": ("observed", "Observed"),
    "source_peer": ("source_peer_baseline", "Source peer baseline"),
    "target_peer": ("target_peer_baseline", "Target peer baseline"),
    # Titled "Source centroid baseline": the LaTeX "Centroid" column is the
    # source centroid in every cross-dataset table. The file stem keeps the old
    # name so the folders stay comparable file by file with heatmaps_reference/.
    "centroid": ("centroid_baseline", "Source centroid baseline"),
}
PANEL_ORDER = list(PANELS)

# Block header as typeset -> slug used in filenames and the values TSV.
BLOCK_SLUGS = {
    "raw logFC": "raw",
    "dataset $z$-score": "dataset_z",
    "dataset $\\times$ cell-line $z$-score": "dataset_cell_line_z",
    "moderated $t$": "moderated_t",
    "sign(logFC)": "sign_logfc",
}
# Table 7 is direction agreement on raw logFC alone, so it prints one unbroken
# run of dataset pairs with no block headers at all. Its rows belong here.
UNBLOCKED_BLOCK = "raw"

SIMILARITIES = ["negative_l2", "cosine"]
RETRIEVAL_TABLES = {"negative_l2": 13, "cosine": 12}
RETRIEVAL_TEX = {
    similarity: LATEX_DIR / "table12_13" / f"table_{RETRIEVAL_TABLES[similarity]}_retrieval_{similarity}_combined.tex"
    for similarity in SIMILARITIES
}

# One group per agreement table or retrieval representation, one arm per
# similarity within it. An arm is a folder; a group shares a colour scale.
# Tables 12 and 13 stay pooled together for each retrieval representation.
GROUPS = {
    "table_6": {
        "block": "raw",
        "arms": {"table_6": LATEX_DIR / "table6" / "table_6_deg_spearman_combined.tex"},
    },
    "table_7": {
        "block": "raw",
        "arms": {"table_7": LATEX_DIR / "table7" / "table_7_direction_combined.tex"},
    },
    "table_8": {
        "block": "raw",
        "arms": {"table_8": LATEX_DIR / "table8" / "table_8_allgene_spearman_combined.tex"},
    },
    "table_12_13_logfc": {
        "block": "raw",
        "arms": {
            f"table_{RETRIEVAL_TABLES[similarity]}_logfc_{similarity}": path
            for similarity, path in RETRIEVAL_TEX.items()
        },
    },
    "table_12_13_signed_logfc": {
        "block": "sign_logfc",
        "arms": {
            f"table_{RETRIEVAL_TABLES[similarity]}_signed_logfc_{similarity}": path
            for similarity, path in RETRIEVAL_TEX.items()
        },
    },
}
ARMS = {arm: group for group, spec in GROUPS.items() for arm in spec["arms"]}

# Historical heatmaps_reference/ folder and variant for --compare-existing.
# These saved reference files retain the previous manuscript numbering.
# Their old Table 9 folders are negative L2 only, so cosine has no counterpart.
EXISTING_ARMS = {
    "table_6": ("table_4", "raw"),
    "table_7": ("table_5", "raw"),
    "table_8": ("table_6", "raw"),
    "table_13_logfc_negative_l2": ("table_9_logfc", "negative_l2"),
    "table_13_signed_logfc_negative_l2": ("table_9_signed_logfc", "negative_l2"),
}

NUMBER = re.compile(r"-?\d+\.\d+")
BLOCK_HEADER = re.compile(r"\\multicolumn\{\d+\}\{l\}\{\\textit\{(.+?)\}\}")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def take_braced(text: str) -> tuple[str, str]:
    """Split ``{a{b}c}rest`` into ``a{b}c`` and ``rest``, counting nesting."""
    depth = 0
    for index, character in enumerate(text):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[1:index], text[index + 1 :]
    raise ValueError(f"unbalanced braces in {text!r}")


def cell_span(cell: str) -> tuple[int, str]:
    """Columns a header cell spans, and its text. Column specs nest braces
    several deep (``{c@{\\hspace{14pt}}}``), so they are walked, not matched."""
    if not cell.startswith("\\multicolumn"):
        return 1, cell
    rest = cell[len("\\multicolumn") :]
    args = []
    while rest.startswith("{") and len(args) < 3:
        argument, rest = take_braced(rest)
        args.append(argument)
    if len(args) != 3:
        raise ValueError(f"malformed \\multicolumn: {cell!r}")
    return int(args[0]), args[2].strip()


def split_row(line: str) -> list[str]:
    """Cells of one row. No cell here nests an ampersand, so this is safe."""
    return [cell.strip() for cell in line.rstrip().removesuffix("\\\\").split("&")]


def header_panel_columns(cells: list[str]) -> dict[str, int]:
    """Panel -> column index, expanding spans as it walks the header."""
    found: dict[str, int] = {}
    index = 0
    for cell in cells:
        span, text = cell_span(cell)
        if text in PANEL_HEADERS:
            found[PANEL_HEADERS[text]] = index
        index += span
    return found


def point_estimate(cell: str) -> float:
    """First number of a cell: the point estimate, ahead of its gray interval."""
    match = NUMBER.search(cell)
    return float(match.group()) if match else np.nan


def parse_latex_table(path: Path) -> pd.DataFrame:
    """Tidy one LaTeX table: a row per (block, dataset pair), a column per panel."""
    if not path.exists():
        raise SystemExit(f"missing {path}")

    reverse = {label: name for name, label in DISPLAY_LABELS.items()}
    columns: dict[str, int] = {}
    block = UNBLOCKED_BLOCK
    rows: list[dict[str, object]] = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("Dataset pair &"):
            columns = header_panel_columns(split_row(stripped))
            missing = [panel for panel in PANEL_ORDER if panel not in columns]
            if missing:
                raise SystemExit(f"{path} header lacks panels {missing}")
            continue

        header = BLOCK_HEADER.search(stripped)
        if header:
            label = header.group(1)
            if label not in BLOCK_SLUGS:
                raise SystemExit(f"{path} has an unmapped block {label!r}")
            block = BLOCK_SLUGS[label]
            continue

        # A data row: after the header, ends the LaTeX row, and opens with a pair.
        if not columns or not stripped.endswith("\\\\"):
            continue
        cells = split_row(stripped)
        pair = cells[0]
        parts = pair.split(PAIR_SEPARATOR)
        if len(parts) != 2 or any(part not in reverse for part in parts):
            continue

        row: dict[str, object] = {
            "block": block,
            "dataset_a": reverse[parts[0]],
            "dataset_b": reverse[parts[1]],
        }
        for panel in PANEL_ORDER:
            row[panel] = point_estimate(cells[columns[panel]])
        rows.append(row)

    if not rows:
        raise SystemExit(f"{path} yielded no data rows")
    frame = pd.DataFrame(rows)
    blank = int(frame[PANEL_ORDER].isna().to_numpy().sum())
    if blank:
        raise SystemExit(f"{path}: {blank} panel cell(s) held no point estimate")
    print(f"[read] {len(frame)} rows, {frame['block'].nunique()} blocks <- {relative(path)}")
    return frame


def load_group(group: str) -> dict[str, pd.DataFrame]:
    """Every arm of a group, parsed. All of them, even when only one is written,
    so the colour scale a folder is drawn on does not depend on that choice."""
    return {arm: parse_latex_table(path) for arm, path in GROUPS[group]["arms"].items()}


def symmetric_matrix(frame: pd.DataFrame, panel: str, order: list[str]) -> pd.DataFrame:
    """Dataset x dataset matrix, mirrored across the diagonal, in display labels."""
    matrix = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    for row in frame.itertuples(index=False):
        a, b = str(row.dataset_a), str(row.dataset_b)
        if a not in matrix.index or b not in matrix.columns:
            continue
        value = getattr(row, panel)
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value
    return matrix


def display_frame(
    matrix: pd.DataFrame, order: list[str], drop_columns: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Trim a symmetric matrix to the published lower-triangle layout.

    The first dataset loses its row and the last its column, as in the notebook
    figures, since both would be empty. Dropped columns are removed on top of
    that, so the mask is derived from each dataset's place in ``order`` rather
    than from the array shape, which no longer tracks it.
    """
    rows = [name for name in order[1:]]
    columns = [name for name in order[:-1] if name not in drop_columns]
    position = {name: index for index, name in enumerate(order)}
    mask = np.array(
        [[position[column] > position[row] for column in columns] for row in rows],
        dtype=bool,
    )
    display = matrix.loc[rows, columns].rename(index=DISPLAY_LABELS, columns=DISPLAY_LABELS)
    return display, mask


def draw_heatmap(
    display: pd.DataFrame,
    mask: np.ndarray,
    title: str,
    ax: plt.Axes,
    vmin: float,
    vmax: float,
    cbar: bool,
) -> None:
    sns.heatmap(
        display,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar=cbar,
        cbar_kws={"shrink": 0.85},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.grid(False)


def save(fig: plt.Figure, output_path: Path, formats: tuple[str, ...]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        path = output_path.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"[write] {relative(path)}")
    plt.close(fig)


def plot_panel(
    frame: pd.DataFrame,
    panel: str,
    title: str,
    order: list[str],
    drop_columns: list[str],
    output_path: Path,
    vmin: float,
    vmax: float,
    formats: tuple[str, ...],
) -> None:
    display, mask = display_frame(symmetric_matrix(frame, panel, order), order, drop_columns)
    width = max(5.5, 1.35 * len(display.columns))
    height = max(4.5, 1.15 * len(display.index))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    draw_heatmap(display, mask, title, ax, vmin, vmax, cbar=True)
    save(fig, output_path, formats)


def compare_existing(arm: str, frame: pd.DataFrame, block: str, tolerance: float) -> None:
    """Report the gap against the heatmaps_reference/ TSV for the same panel."""
    if arm not in EXISTING_ARMS:
        print(f"[compare] {arm}: no counterpart in {relative(EXISTING_DIR)}, skipping")
        return
    folder, variant = EXISTING_ARMS[arm]
    path = EXISTING_DIR / folder / f"{folder}_heatmap_values.tsv"
    if not path.exists():
        print(f"[compare] {arm}: {relative(path)} not found, skipping")
        return

    current = frame[frame["block"] == block]
    # Historical reference files name retrieval variants by similarity and
    # agreement variants by normalization, so one column covers both.
    old = pd.read_csv(path, sep="\t")
    old = old[old["variant"] == variant]
    if old.empty:
        print(f"[compare] {arm}: no {variant!r} rows in {relative(path)}, skipping")
        return

    merged = current.merge(old, on=["dataset_a", "dataset_b"], suffixes=("", "_old"))
    if merged.empty:
        print(f"[compare] {arm}: no shared dataset pairs with {relative(path)}")
        return
    worst = {
        panel: float((merged[panel] - merged[f"{panel}_old"]).abs().max())
        for panel in PANEL_ORDER
    }
    label = max(worst, key=worst.get)
    status = "ok" if worst[label] <= tolerance else "DIFFERS"
    print(
        f"[compare] {arm:32s} vs heatmaps_reference/{folder}: {len(merged)} pairs, "
        f"max |diff| = {worst[label]:.4f} ({label}) [{status}]"
    )
    if worst[label] > tolerance:
        for panel, gap in worst.items():
            print(f"           {panel:12s} max |diff| = {gap:.4f}")


def color_limits(frame: pd.DataFrame, panels: list[str]) -> tuple[float, float]:
    values = pd.concat([frame[panel] for panel in panels], ignore_index=True)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise SystemExit(f"no finite values over {panels}")
    return float(values.min()), float(values.max())


def write_arm(
    arm: str,
    frame: pd.DataFrame,
    blocks: list[str],
    limits: dict[tuple[str, str], tuple[float, float]],
    out_dir: Path,
    drop_columns: list[str],
    plot: bool,
    formats: tuple[str, ...],
) -> None:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    long_path = arm_dir / f"{arm}_heatmap_values.tsv"
    frame[["block", "dataset_a", "dataset_b", *PANEL_ORDER]].sort_values(
        ["block", "dataset_a", "dataset_b"]
    ).to_csv(long_path, sep="\t", index=False)
    print(f"[write] {relative(long_path)}")

    # The block only enters a filename when more than one is drawn, so default
    # panel filenames match the references and both retrieval similarities.
    single = len(blocks) == 1
    for block in blocks:
        group = frame[frame["block"] == block]
        if group.empty:
            raise SystemExit(f"{arm}: no rows for block {block!r}")
        order = [
            name
            for name in DATASET_ORDER
            if name in set(group["dataset_a"]) | set(group["dataset_b"])
        ]
        prefix = "" if single else f"{block}_"

        for panel, (stem, title) in PANELS.items():
            vmin, vmax = limits[(block, panel)]
            name = f"{prefix}{stem}_heatmap"
            symmetric_matrix(group, panel, order).rename(
                index=DISPLAY_LABELS, columns=DISPLAY_LABELS
            ).to_csv(arm_dir / f"{name}.tsv", sep="\t")
            print(f"[write] {relative(arm_dir / f'{name}.tsv')}")
            if plot:
                plot_panel(
                    group,
                    panel,
                    title,
                    order,
                    drop_columns,
                    arm_dir / f"{name}.pdf",
                    vmin,
                    vmax,
                    formats,
                )


def group_limits(
    group: str, frames: dict[str, pd.DataFrame], blocks: list[str], scale: str
) -> dict[tuple[str, str], tuple[float, float]]:
    """Colour limits per (block, panel), pooled over every arm of the group."""
    pooled = pd.concat(frames.values(), ignore_index=True)
    limits = {}
    for block in blocks:
        rows = pooled[pooled["block"] == block]
        if rows.empty:
            raise SystemExit(f"{group}: no rows for block {block!r}")
        for panel in PANEL_ORDER:
            panels = [panel] if scale == "panel" else PANEL_ORDER
            limits[(block, panel)] = color_limits(rows, panels)
        shown = sorted({limits[(block, panel)] for panel in PANEL_ORDER})
        printable = ", ".join(f"{low:.3f} .. {high:.3f}" for low, high in shown)
        print(f"[scale] {group} / {block}: {printable}")
    return limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arms", default="all", help=f"Comma-separated subset of {list(ARMS)}, or 'all'."
    )
    parser.add_argument(
        "--blocks",
        default="default",
        help=(
            "Comma-separated subset of "
            f"{sorted(set(BLOCK_SLUGS.values()))}, 'all', or 'default' for the arm's "
            "own block (raw logFC, or sign(logFC) for the signed arms)."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--drop-columns",
        default="cigs_tcm",
        help="Datasets to leave out of the figure columns. Their rows are kept.",
    )
    parser.add_argument(
        "--scale",
        choices=("block", "panel"),
        default="block",
        help=(
            "Pool the colour scale over all four panels of a block, or each panel "
            "alone. Either way it is pooled over both similarities of an arm."
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Write matrices only.")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument(
        "--compare-existing",
        action="store_true",
        help="Report the gap against tables_and_figures/agreement_retrieval/figures/heatmaps_reference/ for the arms it covers.",
    )
    parser.add_argument(
        "--compare-tol",
        type=float,
        default=5e-4,
        help="Gap treated as agreement; the LaTeX is printed to three decimals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    arms = (
        list(ARMS)
        if args.arms == "all"
        else [a.strip() for a in args.arms.split(",") if a.strip()]
    )
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")

    drop_columns = [name.strip() for name in args.drop_columns.split(",") if name.strip()]
    unknown = [name for name in drop_columns if name not in DISPLAY_LABELS]
    if unknown:
        raise SystemExit(f"unknown datasets to drop: {unknown}")
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    # A group is written at once, because its arms share a colour scale.
    for group in dict.fromkeys(ARMS[arm] for arm in arms):
        print(f"\n=== {group} ===")
        frames = load_group(group)
        available = list(dict.fromkeys(next(iter(frames.values()))["block"]))
        default = GROUPS[group]["block"]
        if args.blocks == "default":
            blocks = [default]
        elif args.blocks == "all":
            blocks = available
        else:
            blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
        missing = [block for block in blocks if block not in available]
        if missing:
            raise SystemExit(f"{group}: blocks {missing} not in {available}")

        limits = group_limits(group, frames, blocks, args.scale)
        for arm, frame in frames.items():
            if arm not in arms:
                continue
            if args.compare_existing:
                compare_existing(arm, frame, default, args.compare_tol)
            write_arm(
                arm,
                frame,
                blocks,
                limits,
                args.out_dir,
                drop_columns,
                not args.no_plot,
                formats,
            )


if __name__ == "__main__":
    main()
