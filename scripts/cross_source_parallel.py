#!/usr/bin/env python3
"""Shared local-process runner for the cross-source scoring commands.

The runner deliberately keeps process boundaries boring:

* the coordinator plans small deterministic TSV inputs;
* spawned workers open H5AD files themselves and write unique result shards;
* a JSON marker, written last, commits each completed shard; and
* the coordinator validates and merges committed shards in task order.

No AnnData object, HDF5 handle, or large NumPy matrix crosses a process boundary.
"""

from __future__ import annotations

# Prevent every process-pool worker from starting a second pool of BLAS threads. These
# variables must be set before NumPy/SciPy are imported in a spawned interpreter.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import fcntl
import hashlib
import importlib
import json
import multiprocessing
import re
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cross_source_core  # noqa: E402
from scripts.notebook_cache import (  # noqa: E402
    CACHE_STRING_COLUMNS,
    frame_identity_fingerprint,
    stable_json_fingerprint,
)
from scripts.population_zscore import (  # noqa: E402
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
    PopulationStatsCatalog,
    dataset_stats_cache_path,
    stats_cache_path,
)


DIAGNOSTIC_COLUMNS = (
    "task_id",
    "analysis",
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "left_dose_key",
    "right_dose_key",
    "query_obs_id",
    "left_obs_id",
    "right_obs_id",
    "stage",
    "reason",
    "error",
)
PROGRESS_LOG_NAME = "progress.log"


def add_computation_arguments(
    parser: argparse.ArgumentParser,
    *,
    computations: Mapping[str, str],
) -> None:
    """Add the shared repeatable computation-selection interface."""
    names = ", ".join(computations)
    parser.add_argument(
        "--compute",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help=(
            "Run only the named computation component. Repeat this option or "
            "supply a comma-separated list. Omit it to run the command's "
            f"complete default set. Available: {names}."
        ),
    )
    parser.add_argument(
        "--list-computations",
        action="store_true",
        help="Print available computation component names and exit.",
    )


def resolve_computations(
    values: Sequence[str],
    *,
    computations: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate selectors and return them in stable declared order."""
    requested = [
        item.strip()
        for value in values
        for item in str(value).split(",")
        if item.strip()
    ]
    if not requested:
        return tuple(computations)
    unknown = sorted(set(requested) - set(computations))
    if unknown:
        raise ValueError(
            f"Unknown --compute value(s): {unknown}; "
            f"available={list(computations)}"
        )
    return tuple(name for name in computations if name in set(requested))


def print_computations(
    computations: Mapping[str, str],
    *,
    stream=None,
) -> None:
    """Print a compact, script-friendly component catalog."""
    destination = sys.stdout if stream is None else stream
    for name, description in computations.items():
        print(f"{name}\t{description}", file=destination)


def append_progress_log(
    log_path: Path,
    *,
    analysis: str,
    message: str,
) -> None:
    """Append one process-safe, immediately flushed progress event."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(f"{timestamp}\t[{analysis}] {message}\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def report_task_progress(
    config: Mapping[str, Any],
    task: "TaskSpec",
    *,
    phase: str,
    detail: str = "",
) -> None:
    """Publish a coarse worker milestone without creating nested progress bars."""
    analysis = str(config.get("analysis", "scoring"))
    log_path = Path(
        config.get(
            "progress_log_path",
            Path(config["output_dir"]) / PROGRESS_LOG_NAME,
        )
    )
    message = f"task={task.task_id} phase={phase}"
    if detail:
        message = f"{message} {detail}"
    append_progress_log(
        log_path,
        analysis=analysis,
        message=message,
    )
    if bool(config.get("worker_progress_console", True)):
        print(f"[{analysis}] {message}", flush=True)


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    if normalized in {"", ".", ".."}:
        raise ValueError(f"Unsafe empty path component derived from {value!r}")
    return normalized


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_normalize_json(item) for item in value)
    if isinstance(value, np.generic):
        return _normalize_json(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return {"__float__": "nan"}
        if np.isposinf(value):
            return {"__float__": "inf"}
        if np.isneginf(value):
            return {"__float__": "-inf"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} as stable JSON")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        record["sha256"] = sha256_file(path)
    return record


def source_code_inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [
        file_record(path, content_hash=True)
        for path in sorted({Path(path).resolve() for path in paths}, key=str)
    ]


def _atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            _normalize_json(dict(payload)),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def output_directory_lock(output_dir: Path) -> Iterator[None]:
    """Hold one non-blocking coordinator lock for an analysis output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".coordinator.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another scoring coordinator is already using {output_dir}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class CommonPaths:
    repo_root: Path
    data_root: Path
    overlap_dir: Path
    output_dir: Path
    z_score_stats_root: Path


@dataclass
class PreparedScope:
    profile: str
    paths: CommonPaths
    dataset_order: list[str]
    source_dirs: dict[str, Path]
    overlap_frames: dict[str, pd.DataFrame]
    matched_pairs: pd.DataFrame
    retained_lines: dict[str, list[str]]
    matched_lines: dict[str, list[str]]
    matched_active_datasets: list[str]
    global_gene_lines: dict[str, list[str]]
    line_global_gene_keys: dict[str, np.ndarray]
    matched_pairs_fingerprint: str

    def worker_config(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "repo_root": str(self.paths.repo_root),
            "data_root": str(self.paths.data_root),
            "overlap_dir": str(self.paths.overlap_dir),
            "output_dir": str(self.paths.output_dir),
            "z_score_stats_root": str(self.paths.z_score_stats_root),
            "dataset_order": list(self.dataset_order),
            "source_dirs": {
                key: str(value) for key, value in self.source_dirs.items()
            },
            "retained_lines": {
                key: list(value) for key, value in self.retained_lines.items()
            },
            "matched_lines": {
                key: list(value) for key, value in self.matched_lines.items()
            },
            "matched_active_datasets": list(self.matched_active_datasets),
            "line_global_gene_keys": {
                key: np.asarray(value).astype(str).tolist()
                for key, value in self.line_global_gene_keys.items()
            },
        }


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    key: tuple[str, ...]
    input_file: str
    n_input_rows: int

    @property
    def filename_stem(self) -> str:
        return f"task-{self.task_id:06d}"


@dataclass(frozen=True)
class TaskExecutionRequest:
    analysis: str
    scorer_module: str
    fingerprint: str
    checkpoint_dir: str
    task: TaskSpec
    worker_config: dict[str, Any]


@dataclass(frozen=True)
class TaskCompletion:
    task_id: int
    n_metric_rows: int
    n_diagnostic_rows: int
    elapsed_seconds: float


def task_paths(
    checkpoint_dir: Path,
    task: TaskSpec,
) -> tuple[Path, Path, Path]:
    root = Path(checkpoint_dir)
    return (
        root / f"{task.filename_stem}.metrics.tsv",
        root / f"{task.filename_stem}.diagnostics.tsv",
        root / f"{task.filename_stem}.complete.json",
    )


def _read_tsv(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if columns is not None and len(columns) == 0:
        return pd.DataFrame()
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={column: str for column in CACHE_STRING_COLUMNS},
    )
    for column in CACHE_STRING_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").fillna("").astype(str)
    return frame


def _validate_one_frame(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    prefix: str,
    load: bool,
) -> Optional[pd.DataFrame]:
    columns = metadata.get(f"{prefix}_columns")
    if not isinstance(columns, list) or not path.is_file():
        return None
    if int(path.stat().st_size) != int(metadata.get(f"{prefix}_size", -1)):
        return None
    if sha256_file(path) != str(metadata.get(f"{prefix}_sha256", "")):
        return None
    if not load:
        return pd.DataFrame(columns=columns)
    try:
        frame = _read_tsv(path, columns)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if frame.columns.tolist() != columns:
        return None
    if len(frame) != int(metadata.get(f"n_{prefix}_rows", -1)):
        return None
    return frame


def validate_checkpoint(
    checkpoint_dir: Path,
    task: TaskSpec,
    *,
    analysis: str,
    fingerprint: str,
    load: bool = False,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    metrics_path, diagnostics_path, marker_path = task_paths(checkpoint_dir, task)
    try:
        metadata = json.loads(marker_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    expected = {
        "analysis": analysis,
        "fingerprint": fingerprint,
        "task_id": int(task.task_id),
        "task_key": list(task.key),
        "n_input_rows": int(task.n_input_rows),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    metrics = _validate_one_frame(
        metrics_path,
        metadata,
        prefix="metric",
        load=load,
    )
    diagnostics = _validate_one_frame(
        diagnostics_path,
        metadata,
        prefix="diagnostic",
        load=load,
    )
    if metrics is None or diagnostics is None:
        return None
    return metrics, diagnostics


def diagnostic_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)
    frame = pd.DataFrame(records)
    for column in DIAGNOSTIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame.loc[:, list(DIAGNOSTIC_COLUMNS)]


def execute_checkpoint_task(request: TaskExecutionRequest) -> TaskCompletion:
    """Spawn-safe worker wrapper that commits both task outputs atomically."""
    started = time.monotonic()
    module = importlib.import_module(request.scorer_module)
    score_task: Callable[
        [TaskSpec, Mapping[str, Any]], tuple[pd.DataFrame, pd.DataFrame]
    ] = getattr(module, "score_task")
    metrics, diagnostics = score_task(request.task, request.worker_config)
    if not isinstance(metrics, pd.DataFrame) or not isinstance(
        diagnostics, pd.DataFrame
    ):
        raise TypeError("score_task must return two pandas DataFrames")

    checkpoint_dir = Path(request.checkpoint_dir)
    metrics_path, diagnostics_path, marker_path = task_paths(
        checkpoint_dir,
        request.task,
    )
    atomic_write_frame(metrics_path, metrics)
    atomic_write_frame(diagnostics_path, diagnostics)
    metadata = {
        "analysis": request.analysis,
        "fingerprint": request.fingerprint,
        "task_id": int(request.task.task_id),
        "task_key": list(request.task.key),
        "n_input_rows": int(request.task.n_input_rows),
        "metric_columns": metrics.columns.tolist(),
        "diagnostic_columns": diagnostics.columns.tolist(),
        "n_metric_rows": int(len(metrics)),
        "n_diagnostic_rows": int(len(diagnostics)),
        "metric_size": int(metrics_path.stat().st_size),
        "diagnostic_size": int(diagnostics_path.stat().st_size),
        "metric_sha256": sha256_file(metrics_path),
        "diagnostic_sha256": sha256_file(diagnostics_path),
        "completed_at_unix": time.time(),
    }
    # The marker is the commit record and must always be published last.
    atomic_write_json(marker_path, metadata)
    return TaskCompletion(
        task_id=request.task.task_id,
        n_metric_rows=len(metrics),
        n_diagnostic_rows=len(diagnostics),
        elapsed_seconds=time.monotonic() - started,
    )


def _merge_checkpoint_kind(
    *,
    tasks: Sequence[TaskSpec],
    checkpoint_dir: Path,
    analysis: str,
    fingerprint: str,
    output_path: Path,
    kind: str,
) -> tuple[int, list[str]]:
    checkpoint_schemas: list[tuple[int, list[str]]] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        validated = validate_checkpoint(
            checkpoint_dir,
            task,
            analysis=analysis,
            fingerprint=fingerprint,
            load=False,
        )
        if validated is None:
            raise RuntimeError(
                f"Task {task.task_id} checkpoint became invalid during merge"
            )
        frame = validated[0 if kind == "metric" else 1]
        columns = frame.columns.tolist()
        if columns:
            checkpoint_schemas.append((task.task_id, columns))

    expected_columns: list[str] = []
    if checkpoint_schemas:
        schema_task_id, expected_columns = max(
            checkpoint_schemas,
            key=lambda item: len(item[1]),
        )
        if len(expected_columns) != len(set(expected_columns)):
            raise ValueError(
                f"Task {schema_task_id} {kind} schema contains duplicate columns"
            )
        expected_set = set(expected_columns)
        for task_id, columns in checkpoint_schemas:
            if len(columns) != len(set(columns)):
                raise ValueError(
                    f"Task {task_id} {kind} schema contains duplicate columns"
                )
            if not set(columns).issubset(expected_set):
                raise ValueError(
                    f"Task {task_id} {kind} schema is not a subset of the "
                    f"largest schema from task {schema_task_id}: {columns!r} "
                    f"versus {expected_columns!r}"
                )
            columns_set = set(columns)
            ordered_subset = [
                column
                for column in expected_columns
                if column in columns_set
            ]
            if ordered_subset != columns:
                raise ValueError(
                    f"Task {task_id} {kind} schema has an incompatible column "
                    f"order relative to task {schema_task_id}: {columns!r} "
                    f"versus {expected_columns!r}"
                )

    total_rows = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        wrote_header = False
        for task in sorted(tasks, key=lambda item: item.task_id):
            validated = validate_checkpoint(
                checkpoint_dir,
                task,
                analysis=analysis,
                fingerprint=fingerprint,
                load=True,
            )
            if validated is None:
                raise RuntimeError(
                    f"Task {task.task_id} checkpoint became invalid during merge"
                )
            frame = validated[0 if kind == "metric" else 1]
            columns = frame.columns.tolist()
            if not columns and len(frame) == 0:
                continue
            if columns != expected_columns:
                frame = frame.reindex(columns=expected_columns)
            frame.to_csv(
                temporary,
                sep="\t",
                index=False,
                mode="a",
                header=not wrote_header,
            )
            wrote_header = True
            total_rows += len(frame)
        if not wrote_header:
            pd.DataFrame().to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return total_rows, expected_columns


def merge_checkpointed_results(
    *,
    analysis: str,
    tasks: Sequence[TaskSpec],
    output_dir: Path,
    fingerprint: str,
    final_metrics_name: str,
    elapsed_seconds: float = 0.0,
    merge_only: bool = False,
) -> tuple[Path, Path, int, int]:
    """Validate and atomically merge an existing set of task checkpoints."""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints" / fingerprint
    invalid = [
        task.task_id
        for task in tasks
        if validate_checkpoint(
            checkpoint_dir,
            task,
            analysis=analysis,
            fingerprint=fingerprint,
            load=False,
        )
        is None
    ]
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} task checkpoints are incomplete or invalid: "
            f"{invalid[:10]}"
        )

    metrics_path = output_dir / final_metrics_name
    diagnostics_path = output_dir / "diagnostics.tsv"
    merge_token = f"{os.getpid()}-{time.monotonic_ns()}"
    staged_metrics_path = output_dir / (
        f".{metrics_path.name}.merge-{merge_token}"
    )
    staged_diagnostics_path = output_dir / (
        f".{diagnostics_path.name}.merge-{merge_token}"
    )
    try:
        n_metrics, metric_columns = _merge_checkpoint_kind(
            tasks=tasks,
            checkpoint_dir=checkpoint_dir,
            analysis=analysis,
            fingerprint=fingerprint,
            output_path=staged_metrics_path,
            kind="metric",
        )
        n_diagnostics, diagnostic_columns = _merge_checkpoint_kind(
            tasks=tasks,
            checkpoint_dir=checkpoint_dir,
            analysis=analysis,
            fingerprint=fingerprint,
            output_path=staged_diagnostics_path,
            kind="diagnostic",
        )
        if n_metrics == 0:
            raise ValueError(
                f"[{analysis}] every task completed, but no metric rows were scored"
            )
        # Publish the primary result last. If validation or either staged merge fails,
        # an existing final result remains untouched.
        os.replace(staged_diagnostics_path, diagnostics_path)
        os.replace(staged_metrics_path, metrics_path)
    finally:
        for staged_path in (staged_metrics_path, staged_diagnostics_path):
            if staged_path.exists():
                staged_path.unlink()
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "analysis": analysis,
            "fingerprint": fingerprint,
            "n_tasks": len(tasks),
            "n_metric_rows": n_metrics,
            "n_diagnostic_rows": n_diagnostics,
            "metric_columns": metric_columns,
            "diagnostic_columns": diagnostic_columns,
            "metrics_file": metrics_path.name,
            "diagnostics_file": diagnostics_path.name,
            "progress_log_file": PROGRESS_LOG_NAME,
            "elapsed_seconds": float(elapsed_seconds),
            "merge_only": bool(merge_only),
        },
    )
    return metrics_path, diagnostics_path, n_metrics, n_diagnostics


def run_checkpointed_tasks(
    *,
    analysis: str,
    scorer_module: str,
    tasks: Sequence[TaskSpec],
    worker_config: Mapping[str, Any],
    output_dir: Path,
    fingerprint: str,
    workers: int,
    force: bool,
    final_metrics_name: str,
    progress_mode: str = "auto",
) -> tuple[Path, Path]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    if progress_mode not in {"auto", "always", "off"}:
        raise ValueError(
            "--progress must be one of: auto, always, off"
        )
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints" / fingerprint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_dir / PROGRESS_LOG_NAME

    progress_bar = None

    def report(
        message: str,
        *,
        console: Optional[bool] = None,
    ) -> None:
        append_progress_log(
            progress_log_path,
            analysis=analysis,
            message=message,
        )
        if console is None:
            console = progress_mode != "off"
        if console:
            print(f"[{analysis}] {message}", flush=True)

    pending = [
        task
        for task in tasks
        if force
        or validate_checkpoint(
            checkpoint_dir,
            task,
            analysis=analysis,
            fingerprint=fingerprint,
            load=False,
        )
        is None
    ]
    cached_count = len(tasks) - len(pending)
    progress_disabled = (
        progress_mode == "off"
        or (progress_mode == "auto" and not sys.stderr.isatty())
    )
    progress_bar = tqdm(
        total=len(tasks),
        initial=cached_count,
        desc=f"{analysis} tasks",
        unit="task",
        dynamic_ncols=True,
        disable=progress_disabled,
    )
    report(
        f"run fingerprint={fingerprint} force={force} "
        f"progress={progress_mode}"
    )
    report(
        f"tasks={len(tasks):,}; cached={len(tasks) - len(pending):,}; "
        f"pending={len(pending):,}; workers={workers}"
    )
    request_worker_config = dict(worker_config)
    request_worker_config["analysis"] = analysis
    request_worker_config["progress_log_path"] = str(progress_log_path)
    request_worker_config["worker_progress_console"] = not progress_disabled
    requests = [
        TaskExecutionRequest(
            analysis=analysis,
            scorer_module=scorer_module,
            fingerprint=fingerprint,
            checkpoint_dir=str(checkpoint_dir),
            task=task,
            worker_config=request_worker_config,
        )
        for task in pending
    ]
    completed = 0
    started = time.monotonic()
    try:
        if workers == 1:
            for request in requests:
                try:
                    result = execute_checkpoint_task(request)
                except BaseException as exc:
                    report(
                        f"failed task={request.task.task_id} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    raise
                completed += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    task=result.task_id,
                    rows=result.n_metric_rows,
                    elapsed=f"{result.elapsed_seconds:.1f}s",
                    refresh=True,
                )
                report(
                    f"completed {completed}/{len(requests)} "
                    f"task={result.task_id} rows={result.n_metric_rows} "
                    f"elapsed={result.elapsed_seconds:.1f}s",
                    console=(
                        progress_disabled and progress_mode != "off"
                    ),
                )
        elif requests:
            worker_count = min(workers, len(requests), os.cpu_count() or workers)
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                futures = {
                    executor.submit(
                        execute_checkpoint_task,
                        request,
                    ): request.task.task_id
                    for request in requests
                }
                current_task_id = None
                try:
                    for future in as_completed(futures):
                        current_task_id = futures[future]
                        result = future.result()
                        completed += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            task=result.task_id,
                            rows=result.n_metric_rows,
                            elapsed=f"{result.elapsed_seconds:.1f}s",
                            refresh=True,
                        )
                        report(
                            f"completed {completed}/{len(requests)} "
                            f"task={result.task_id} rows={result.n_metric_rows} "
                            f"elapsed={result.elapsed_seconds:.1f}s",
                            console=(
                                progress_disabled and progress_mode != "off"
                            ),
                        )
                except BaseException as exc:
                    report(
                        f"failed task={current_task_id} "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    for future in futures:
                        future.cancel()
                    raise
    finally:
        if progress_bar is not None:
            progress_bar.close()

    (
        metrics_path,
        diagnostics_path,
        n_metrics,
        n_diagnostics,
    ) = merge_checkpointed_results(
        analysis=analysis,
        tasks=tasks,
        output_dir=output_dir,
        fingerprint=fingerprint,
        final_metrics_name=final_metrics_name,
        elapsed_seconds=time.monotonic() - started,
    )
    report(
        f"merged {n_metrics:,} rows -> {metrics_path}; "
        f"diagnostics={n_diagnostics:,}"
    )
    return metrics_path, diagnostics_path


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    analysis: str,
    include_z_score_scales: bool = True,
) -> None:
    parser.add_argument(
        "--datasets",
        default="l1000_phase1,l1000_phase2,tahoe,sciplex,cigs_mce,cigs_tcm",
        help="Comma-separated dataset names (default: manuscript roster), or 'all' for the full profile.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/lustre/groups/ml01/workspace/olga.novitskaia/data_updated"
        ),
        help="Root containing configured line-level grouped-result H5ADs.",
    )
    parser.add_argument(
        "--overlap-dir",
        type=Path,
        default=REPO_ROOT / "results" / "overlap_filtered_h5ads",
        help="Directory containing <dataset>_overlap_filtered.h5ad files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "parallel_cross_source" / analysis,
        help="Base output directory. A run tag or dataset slug is appended.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional output subdirectory name.",
    )
    parser.add_argument(
        "--z-score-stats-root",
        metavar="PATH",
        type=Path,
        default=REPO_ROOT / "results" / "w4_population_zscore_stats",
        help="Existing population z-score statistics cache root.",
    )
    if include_z_score_scales:
        parser.add_argument(
            "--z-score-scales",
            choices=("all", "dataset", "dataset-cell-type"),
            default="all",
            help=(
                "Population z-score scopes to score. 'dataset' is the reviewer-minimal primary "
                "analysis; 'all' preserves the full sensitivity bundle."
            ),
        )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Spawned scoring processes. Use 1 for serial debugging.",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "off"),
        default="auto",
        help=(
            "Task progress bar mode. 'auto' shows tqdm in an interactive "
            "terminal; progress.log is always written."
        ),
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=500,
        help="Maximum matched-pair rows per DEG/signature shard.",
    )
    parser.add_argument(
        "--max-baseline-peers",
        type=int,
        default=256,
        help=(
            "Maximum individual peers scored per query; use 0 for all. "
            "Centroids always use every eligible peer."
        ),
    )
    parser.add_argument(
        "--peer-sampling-seed",
        type=int,
        default=20260505,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every task even when compatible checkpoints exist.",
    )
    parser.add_argument(
        "--merge-only",
        nargs="?",
        const="planned",
        default=None,
        metavar="FINGERPRINT",
        help=(
            "Skip input validation and scoring, then merge an already completed "
            "checkpoint run. Without a value, use planned_run.json; otherwise "
            "merge the supplied fingerprint."
        ),
    )


def selected_z_score_scale_variants(value: str) -> tuple[str, ...]:
    value = str(value)
    if value == "all":
        return (
            PER_GENE_DATASET_CELL_TYPE_VARIANT,
            PER_GENE_DATASET_VARIANT,
        )
    if value == "dataset":
        return (PER_GENE_DATASET_VARIANT,)
    if value == "dataset-cell-type":
        return (PER_GENE_DATASET_CELL_TYPE_VARIANT,)
    raise ValueError(f"Unsupported z-score scope selection: {value!r}")


def selected_datasets(profile: str, value: str) -> list[str]:
    available = cross_source_core.production_dataset_order(profile)
    if str(value).strip().lower() == "all":
        return available
    selected = [
        item.strip() for item in str(value).split(",") if item.strip()
    ]
    if not selected:
        raise ValueError("--datasets must contain at least one dataset")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise KeyError(
            f"Datasets are not available in the {profile!r} profile: {unknown}; "
            f"available={available}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("--datasets contains duplicate names")
    if len(selected) < 2:
        raise ValueError("Cross-source scoring requires at least two datasets")
    return selected


def resolve_common_paths(
    args: argparse.Namespace,
    *,
    analysis: str,
    dataset_order: Sequence[str],
) -> CommonPaths:
    run_component = (
        _safe_component(args.run_tag)
        if str(args.run_tag).strip()
        else (
            "production"
            if list(dataset_order)
            == cross_source_core.production_dataset_order(analysis)
            else "__".join(_safe_component(name) for name in dataset_order)
        )
    )
    return CommonPaths(
        repo_root=REPO_ROOT,
        data_root=Path(args.data_root).resolve(),
        overlap_dir=Path(args.overlap_dir).resolve(),
        output_dir=(Path(args.output_dir).resolve() / run_component),
        z_score_stats_root=Path(args.z_score_stats_root).resolve(),
    )


def _required_layers(
    profile: str,
    *,
    workload: str = "full",
) -> tuple[str, ...]:
    if profile == "retrieval" and workload in {
        "reviewer-minimal",
        "peer-only",
    }:
        return ("logFC",)
    if profile == "signature":
        return ("logFC", "t")
    return ("logFC", "t")


def validate_line_sources(
    *,
    profile: str,
    workload: str = "full",
    source_catalog: cross_source_core.LineSourceCatalog,
    dataset_lines: Mapping[str, Sequence[str]],
    required_layers: Optional[Sequence[str]] = None,
    require_adj_p: Optional[bool] = None,
) -> None:
    errors: list[str] = []
    for dataset_name in sorted(dataset_lines):
        for cell_type in sorted(set(dataset_lines[dataset_name])):
            try:
                source = source_catalog.get_line_source(dataset_name, cell_type)
                missing = [
                    layer
                    for layer in (
                        tuple(required_layers)
                        if required_layers is not None
                        else _required_layers(
                            profile,
                            workload=workload,
                        )
                    )
                    if layer not in source.adata.layers
                ]
                adj_p_required = (
                    bool(require_adj_p)
                    if require_adj_p is not None
                    else (
                        profile == "deg"
                        or (
                            profile == "retrieval"
                            and workload
                            not in {"reviewer-minimal", "peer-only"}
                        )
                    )
                )
                if adj_p_required:
                    try:
                        cross_source_core.first_available_layer(
                            source,
                            (
                                "adj.P.Value.within_one_contrast",
                                "adj.P.Value.across_all_contrasts",
                            ),
                        )
                    except KeyError as exc:
                        errors.append(str(exc))
                if missing:
                    errors.append(
                        f"{source.path} is missing required layers {missing}"
                    )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError(
            "Line-source validation failed:\n- " + "\n- ".join(errors)
        )


def prepare_scope(
    args: argparse.Namespace,
    *,
    profile: str,
    require_z_scores: bool = True,
) -> PreparedScope:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.rows_per_shard < 1:
        raise ValueError("--rows-per-shard must be positive")
    if args.max_baseline_peers is not None and args.max_baseline_peers < 0:
        raise ValueError(
            "--max-baseline-peers cannot be negative; use 0 for all peers"
        )

    dataset_order = selected_datasets(profile, args.datasets)
    paths = resolve_common_paths(
        args,
        analysis=profile,
        dataset_order=dataset_order,
    )
    if not paths.overlap_dir.is_dir():
        raise FileNotFoundError(
            f"Overlap directory does not exist: {paths.overlap_dir}"
        )
    missing_overlap = [
        paths.overlap_dir / f"{dataset}_overlap_filtered.h5ad"
        for dataset in dataset_order
        if not (
            paths.overlap_dir / f"{dataset}_overlap_filtered.h5ad"
        ).is_file()
    ]
    if missing_overlap:
        raise FileNotFoundError(
            "Missing overlap-filtered H5AD inputs:\n- "
            + "\n- ".join(str(path) for path in missing_overlap)
        )

    source_dirs = cross_source_core.source_dataset_dirs(paths.data_root, profile)
    source_catalog = cross_source_core.LineSourceCatalog(source_dirs)
    try:
        scope = cross_source_core.prepare_cross_source_scope(
            dataset_order=dataset_order,
            overlap_dir=paths.overlap_dir,
            output_dir=paths.output_dir,
            source_catalog=source_catalog,
            settings=cross_source_core.DEFAULT_MATCH_SETTINGS,
        )
        validate_line_sources(
            profile=profile,
            workload=str(getattr(args, "workload", "full")),
            source_catalog=source_catalog,
            dataset_lines=scope.global_gene_lines,
            required_layers=getattr(
                args,
                "required_layers_override",
                None,
            ),
            require_adj_p=getattr(
                args,
                "require_adj_p_override",
                None,
            ),
        )
        if require_z_scores:
            try:
                # Construction performs the complete read-only cache readiness check.
                PopulationStatsCatalog(
                    source_dataset_dirs=source_dirs,
                    matched_lines=scope.matched_lines,
                    matched_dataset_names=scope.matched_active_datasets,
                    resolve_line_path=source_catalog.resolve_line_path,
                    cache_root=paths.z_score_stats_root,
                    precompute_mode="stop",
                )
            except (
                FileNotFoundError,
                RuntimeError,
                TimeoutError,
                ValueError,
            ) as exc:
                dataset_arguments = " ".join(
                    "--dataset-dir "
                    + shlex.quote(
                        f"{dataset_name}={source_dirs[dataset_name]}"
                    )
                    for dataset_name in scope.matched_active_datasets
                )
                command = " ".join(
                    [
                        "uv run python scripts/precompute_population_zscore.py",
                        dataset_arguments,
                        "--scope both --row-chunk-size 1024 --workers 2",
                        "--cache-root",
                        shlex.quote(str(paths.z_score_stats_root)),
                        "--qc-output",
                        shlex.quote(
                            str(paths.z_score_stats_root / "precompute_qc.tsv")
                        ),
                    ]
                )
                raise RuntimeError(
                    f"Required population z-score caches are not ready: {exc}\n"
                    f"Prepare them first with:\n{command}"
                ) from exc
        return PreparedScope(
            profile=profile,
            paths=paths,
            dataset_order=list(dataset_order),
            source_dirs=dict(source_dirs),
            overlap_frames={
                dataset_name: scope.dataset_indices[dataset_name][
                    "frame"
                ].copy()
                for dataset_name in dataset_order
            },
            matched_pairs=scope.matched_pairs.copy(),
            retained_lines={
                key: list(value) for key, value in scope.retained_lines.items()
            },
            matched_lines={
                key: list(value) for key, value in scope.matched_lines.items()
            },
            matched_active_datasets=list(scope.matched_active_datasets),
            global_gene_lines={
                key: list(value) for key, value in scope.global_gene_lines.items()
            },
            line_global_gene_keys={
                key: np.asarray(value).copy()
                for key, value in scope.line_global_gene_keys.items()
            },
            matched_pairs_fingerprint=scope.matched_pairs_fingerprint,
        )
    finally:
        source_catalog.close()


def make_worker_catalog(
    config: Mapping[str, Any],
) -> cross_source_core.LineSourceCatalog:
    catalog = cross_source_core.LineSourceCatalog(
        {
            key: Path(value)
            for key, value in config["source_dirs"].items()
        }
    )
    catalog.line_global_shared_gene_keys = {
        key: np.asarray(value, dtype=object)
        for key, value in config["line_global_gene_keys"].items()
    }
    catalog.global_gene_position_cache = {}
    return catalog


def make_worker_z_score_catalog(
    config: Mapping[str, Any],
    source_catalog: cross_source_core.LineSourceCatalog,
) -> PopulationStatsCatalog:
    return PopulationStatsCatalog(
        source_dataset_dirs={
            key: Path(value)
            for key, value in config["source_dirs"].items()
        },
        matched_lines={
            key: list(value) for key, value in config["matched_lines"].items()
        },
        matched_dataset_names=list(config["matched_active_datasets"]),
        resolve_line_path=source_catalog.resolve_line_path,
        cache_root=Path(config["z_score_stats_root"]),
        precompute_mode="stop",
    )


def _task_key(
    context_key: Sequence[Any],
    row_start: int,
    row_stop: int,
    *,
    include_range: bool,
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in context_key)
    if include_range:
        return (*normalized, f"rows={row_start}:{row_stop}")
    return normalized


def build_tasks(
    frame: pd.DataFrame,
    *,
    context_columns: Sequence[str],
    rows_per_shard: Optional[int],
) -> tuple[list[TaskSpec], dict[int, pd.DataFrame]]:
    missing = sorted(set(context_columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Task frame is missing context columns: {missing}")
    tasks: list[TaskSpec] = []
    task_frames: dict[int, pd.DataFrame] = {}
    next_task_id = 1
    grouped = frame.groupby(list(context_columns), sort=False, dropna=False)
    for raw_key, group in grouped:
        context_key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        group = group.copy()
        step = len(group) if rows_per_shard is None else int(rows_per_shard)
        for row_start in range(0, len(group), max(1, step)):
            row_stop = min(row_start + max(1, step), len(group))
            task_frame = group.iloc[row_start:row_stop].copy()
            task = TaskSpec(
                task_id=next_task_id,
                key=_task_key(
                    context_key,
                    row_start,
                    row_stop,
                    include_range=rows_per_shard is not None,
                ),
                input_file=f"task-{next_task_id:06d}.input.tsv",
                n_input_rows=len(task_frame),
            )
            tasks.append(task)
            task_frames[next_task_id] = task_frame
            next_task_id += 1
    return tasks, task_frames


def materialize_task_plan(
    *,
    tasks: Sequence[TaskSpec],
    task_frames: Mapping[int, pd.DataFrame],
    output_dir: Path,
) -> Path:
    input_dir = Path(output_dir) / "task_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for task in tasks:
        input_path = input_dir / task.input_file
        atomic_write_frame(input_path, task_frames[task.task_id])
        manifest_rows.append(
            {
                "task_id": task.task_id,
                "task_key": json.dumps(list(task.key), separators=(",", ":")),
                "input_file": str(input_path),
                "n_input_rows": task.n_input_rows,
                "input_sha256": sha256_file(input_path),
            }
        )
    manifest_path = Path(output_dir) / "task_manifest.tsv"
    atomic_write_frame(manifest_path, pd.DataFrame(manifest_rows))
    return manifest_path


def load_task_manifest(output_dir: Path) -> list[TaskSpec]:
    """Reload the deterministic task plan needed to merge old checkpoints."""
    manifest_path = Path(output_dir) / "task_manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cannot merge existing checkpoints without {manifest_path}"
        )
    frame = _read_tsv(manifest_path)
    required = {"task_id", "task_key", "input_file", "n_input_rows"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Task manifest {manifest_path} is missing columns: {missing}"
        )
    tasks: list[TaskSpec] = []
    for row in frame.to_dict(orient="records"):
        try:
            raw_key = json.loads(str(row["task_key"]))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Task manifest contains invalid task_key JSON: "
                f"{row['task_key']!r}"
            ) from exc
        if not isinstance(raw_key, list):
            raise ValueError(
                f"Task manifest task_key must be a JSON list: {raw_key!r}"
            )
        tasks.append(
            TaskSpec(
                task_id=int(row["task_id"]),
                key=tuple(str(value) for value in raw_key),
                input_file=Path(str(row["input_file"])).name,
                n_input_rows=int(row["n_input_rows"]),
            )
        )
    tasks.sort(key=lambda task: task.task_id)
    task_ids = [task.task_id for task in tasks]
    if not tasks or len(task_ids) != len(set(task_ids)):
        raise ValueError(
            f"Task manifest {manifest_path} is empty or has duplicate task IDs"
        )
    return tasks


def materialize_overlap_metadata(scope: PreparedScope) -> dict[str, str]:
    metadata_dir = scope.paths.output_dir / "overlap_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for dataset_name in scope.dataset_order:
        path = metadata_dir / f"{_safe_component(dataset_name)}.tsv"
        atomic_write_frame(path, scope.overlap_frames[dataset_name])
        paths[dataset_name] = str(path)
    return paths


def task_input_path(config: Mapping[str, Any], task: TaskSpec) -> Path:
    return Path(config["output_dir"]) / "task_inputs" / task.input_file


def read_task_input(config: Mapping[str, Any], task: TaskSpec) -> pd.DataFrame:
    frame = _read_tsv(task_input_path(config, task))
    if len(frame) != task.n_input_rows:
        raise RuntimeError(
            f"Task {task.task_id} expected {task.n_input_rows} input rows, "
            f"found {len(frame)}"
        )
    return frame


def read_overlap_metadata(
    config: Mapping[str, Any],
    dataset_name: str,
) -> pd.DataFrame:
    try:
        path = Path(config["overlap_metadata_files"][str(dataset_name)])
    except KeyError as exc:
        raise KeyError(
            f"No materialized overlap metadata for {dataset_name!r}"
        ) from exc
    return _read_tsv(path)


def z_score_cache_inventory(scope: PreparedScope) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for dataset_name in scope.matched_active_datasets:
        for cell_type in scope.matched_lines.get(dataset_name, []):
            path = stats_cache_path(
                scope.paths.z_score_stats_root,
                dataset_name,
                cell_type,
            )
            paths.add(path)
            paths.add(path.with_suffix(".cache.json"))
        dataset_path = dataset_stats_cache_path(
            scope.paths.z_score_stats_root,
            dataset_name,
        )
        paths.add(dataset_path)
        paths.add(dataset_path.with_suffix(".cache.json"))
    return [file_record(path) for path in sorted(paths, key=str)]


def run_fingerprint(
    *,
    analysis: str,
    scope: PreparedScope,
    tasks: Sequence[TaskSpec],
    settings: Mapping[str, Any],
    code_paths: Sequence[Path],
) -> str:
    source_catalog = cross_source_core.LineSourceCatalog(scope.source_dirs)
    source_paths = [
        Path(record["path"])
        for record in cross_source_core.line_source_inventory(
            scope.global_gene_lines,
            source_catalog.resolve_line_path,
        )
    ]
    overlap_paths = [
        scope.paths.overlap_dir / f"{dataset}_overlap_filtered.h5ad"
        for dataset in scope.dataset_order
    ]
    payload = {
        "analysis": analysis,
        "dataset_order": scope.dataset_order,
        "matched_pairs_fingerprint": scope.matched_pairs_fingerprint,
        "matched_pair_identity": frame_identity_fingerprint(
            scope.matched_pairs,
            cross_source_core.MATCH_PAIR_IDENTITY_COLUMNS,
            order_sensitive=True,
        ),
        "tasks": [
            {
                "task_id": task.task_id,
                "key": list(task.key),
                "n_input_rows": task.n_input_rows,
            }
            for task in tasks
        ],
        "settings": dict(settings),
        "overlap_inputs": [
            file_record(path) for path in sorted(overlap_paths, key=str)
        ],
        "line_inputs": [
            file_record(path) for path in sorted(set(source_paths), key=str)
        ],
        "z_score_caches": (
            z_score_cache_inventory(scope)
            if settings.get("z_score_scale_variants")
            else []
        ),
        "code": source_code_inventory(code_paths),
    }
    return stable_json_fingerprint(payload)


def finalize_worker_config(
    scope: PreparedScope,
    *,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    config = scope.worker_config()
    config["settings"] = _normalize_json(dict(settings))
    return config


def run_analysis(
    *,
    analysis: str,
    scorer_module: str,
    args: argparse.Namespace,
    context_columns: Sequence[str],
    rows_per_shard: Optional[int],
    final_metrics_name: str,
    settings: Mapping[str, Any],
    code_paths: Sequence[Path],
    task_frame_factory: Optional[
        Callable[[PreparedScope], pd.DataFrame]
    ] = None,
) -> tuple[Path, Path]:
    dataset_order = selected_datasets(analysis, args.datasets)
    paths = resolve_common_paths(
        args,
        analysis=analysis,
        dataset_order=dataset_order,
    )
    with output_directory_lock(paths.output_dir):
        if args.merge_only is not None:
            merge_started = time.monotonic()
            tasks = load_task_manifest(paths.output_dir)
            requested_fingerprint = str(args.merge_only)
            if requested_fingerprint == "planned":
                planned_path = paths.output_dir / "planned_run.json"
                try:
                    planned = json.loads(planned_path.read_text())
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"--merge-only requires the existing {planned_path}"
                    ) from exc
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Cannot parse planned run metadata in {planned_path}"
                    ) from exc
                if planned.get("analysis") != analysis:
                    raise ValueError(
                        f"Planned run analysis {planned.get('analysis')!r} "
                        f"does not match {analysis!r}"
                    )
                fingerprint = str(planned.get("fingerprint", ""))
                planned_task_count = int(planned.get("n_tasks", -1))
                if planned_task_count != len(tasks):
                    raise ValueError(
                        f"Planned run expects {planned_task_count} tasks, but "
                        f"the task manifest contains {len(tasks)}"
                    )
            else:
                fingerprint = requested_fingerprint
            if not re.fullmatch(r"[A-Za-z0-9._-]+", fingerprint):
                raise ValueError(
                    f"Unsafe or empty checkpoint fingerprint: {fingerprint!r}"
                )
            progress_log_path = paths.output_dir / PROGRESS_LOG_NAME
            append_progress_log(
                progress_log_path,
                analysis=analysis,
                message=(
                    f"merge-only fingerprint={fingerprint} "
                    f"tasks={len(tasks):,}"
                ),
            )
            (
                metrics_path,
                diagnostics_path,
                n_metrics,
                n_diagnostics,
            ) = merge_checkpointed_results(
                analysis=analysis,
                tasks=tasks,
                output_dir=paths.output_dir,
                fingerprint=fingerprint,
                final_metrics_name=final_metrics_name,
                elapsed_seconds=time.monotonic() - merge_started,
                merge_only=True,
            )
            message = (
                f"merge-only completed {n_metrics:,} rows -> {metrics_path}; "
                f"diagnostics={n_diagnostics:,}"
            )
            append_progress_log(
                progress_log_path,
                analysis=analysis,
                message=message,
            )
            if args.progress != "off":
                print(f"[{analysis}] {message}", flush=True)
            return metrics_path, diagnostics_path

        scope = prepare_scope(
            args,
            profile=analysis,
            require_z_scores=bool(settings.get("z_score_scale_variants")),
        )
        task_frame = (
            scope.matched_pairs
            if task_frame_factory is None
            else task_frame_factory(scope)
        )
        tasks, task_frames = build_tasks(
            task_frame,
            context_columns=context_columns,
            rows_per_shard=rows_per_shard,
        )
        if not tasks:
            raise ValueError("No scoring tasks were produced")
        fingerprint = run_fingerprint(
            analysis=analysis,
            scope=scope,
            tasks=tasks,
            settings=settings,
            code_paths=code_paths,
        )
        materialize_task_plan(
            tasks=tasks,
            task_frames=task_frames,
            output_dir=scope.paths.output_dir,
        )
        worker_config = finalize_worker_config(scope, settings=settings)
        worker_config["overlap_metadata_files"] = (
            materialize_overlap_metadata(scope)
        )
        worker_config["fingerprint"] = fingerprint
        atomic_write_json(
            scope.paths.output_dir / "planned_run.json",
            {
                "analysis": analysis,
                "fingerprint": fingerprint,
                "dataset_order": scope.dataset_order,
                "settings": dict(settings),
                "n_tasks": len(tasks),
                "n_matched_pairs": len(scope.matched_pairs),
            },
        )
        return run_checkpointed_tasks(
            analysis=analysis,
            scorer_module=scorer_module,
            tasks=tasks,
            worker_config=worker_config,
            output_dir=scope.paths.output_dir,
            fingerprint=fingerprint,
            workers=args.workers,
            force=args.force,
            final_metrics_name=final_metrics_name,
            progress_mode=args.progress,
        )


__all__ = [
    "CommonPaths",
    "DIAGNOSTIC_COLUMNS",
    "PreparedScope",
    "TaskCompletion",
    "TaskExecutionRequest",
    "TaskSpec",
    "add_computation_arguments",
    "add_common_arguments",
    "atomic_write_frame",
    "atomic_write_json",
    "build_tasks",
    "diagnostic_frame",
    "execute_checkpoint_task",
    "finalize_worker_config",
    "make_worker_catalog",
    "make_worker_z_score_catalog",
    "load_task_manifest",
    "materialize_overlap_metadata",
    "materialize_task_plan",
    "merge_checkpointed_results",
    "output_directory_lock",
    "prepare_scope",
    "print_computations",
    "read_task_input",
    "read_overlap_metadata",
    "report_task_progress",
    "run_analysis",
    "run_checkpointed_tasks",
    "resolve_computations",
    "selected_z_score_scale_variants",
    "run_fingerprint",
    "sha256_file",
    "task_input_path",
    "validate_checkpoint",
]
