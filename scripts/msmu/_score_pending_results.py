#!/usr/bin/env python3
"""Directory-driven serial orchestration for canonical MSMU scoring."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, TextIO

from spatial_vlm_eval.benchmarks.msmu.scorer import (
    MSMU_OFFICIAL_TEST_SIZE,
    OFFICIAL_QUAL_TYPES,
    OFFICIAL_QUANT_TYPES,
    SCORER_PROTOCOL,
)


STATE_NAMES = ("new", "resume", "retry", "complete", "excluded_protocol")
PENDING_STATES = frozenset({"new", "resume", "retry"})
EXPECTED_PUBLICATION_GATES = frozenset(
    {
        "prediction_validation_passed",
        "full_official_test_split",
        "all_official_types_present",
        "judge_failures_zero",
    }
)
CANONICAL_SCORE_FILES = (
    "prediction_validation.json",
    "judge_cache.jsonl",
    "judge_failures.jsonl",
    "scored_rows.jsonl",
    "summary.json",
    "score.log",
)


class ConfigurationError(RuntimeError):
    """The CLI, filesystem, or environment is not safe to use."""


class LockBusyError(RuntimeError):
    """Another serial scoring batch owns the output root."""


class JudgePreflightError(RuntimeError):
    """The configured judge endpoint is not ready for scoring."""


@dataclass(frozen=True)
class ResultState:
    predictions: Path
    score_dir: Path
    state: str
    included: bool
    reason: str
    result: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "predictions": str(self.predictions),
            "score_dir": str(self.score_dir),
            "state": self.state,
            "included": self.included,
            "reason": self.reason,
            "result": self.result,
            "scorer_protocol": SCORER_PROTOCOL,
        }


class BatchLock:
    """Non-blocking process lock backed by the persistent batch lock file."""

    def __init__(self, path: Path):
        self.path = path
        self.handle: TextIO | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise LockBusyError(f"batch lock is already held: {self.path}") from exc

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None

    def __enter__(self) -> BatchLock:
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.release()


class BatchLogger:
    """Mirror batch messages and scorer output to stdout and a run log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        rendered = message.rstrip("\n")
        print(rendered, flush=True)
        self.handle.write(rendered + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> BatchLogger:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and serially score pending MSMU stage-three results."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true", dest="list_results")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--status", action="store_true")
    parser.add_argument(
        "--results-root",
        help="Absolute stage-three result root; defaults to .env.server configuration.",
    )
    parser.add_argument(
        "--predictions",
        help=(
            "score only this absolute predictions.jsonl inside the results root; "
            "the candidate is still validated and executed under the canonical batch lock"
        ),
    )
    return parser.parse_args(argv)


def resolve_results_root(argument: str | None) -> Path:
    raw = argument or os.environ.get("MSMU_SCORE_RESULTS_ROOT", "")
    if not raw:
        raise ConfigurationError(
            "results root is not configured; set MANUAL_TEST_OUTPUT_ROOT in .env.server"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ConfigurationError(f"results root must be absolute: {raw}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ConfigurationError(f"results root is not a directory: {resolved}")
    return resolved


def resolve_dataset_root() -> Path:
    raw = os.environ.get("DATASET_ROOT", "")
    if not raw:
        raise ConfigurationError("DATASET_ROOT is missing from .env.server")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise ConfigurationError(f"DATASET_ROOT is not a directory: {path}")
    return path


def resolve_selected_predictions(argument: str | None, results_root: Path) -> Path | None:
    if argument is None:
        return None
    path = Path(argument).expanduser()
    if not path.is_absolute():
        raise ConfigurationError(f"predictions path must be absolute: {argument}")
    resolved = path.resolve()
    if resolved.name != "predictions.jsonl":
        raise ConfigurationError(
            f"selected predictions must be named predictions.jsonl: {resolved}"
        )
    if not resolved.is_file():
        raise ConfigurationError(f"selected predictions does not exist: {resolved}")
    try:
        resolved.relative_to(results_root)
    except ValueError as exc:
        raise ConfigurationError(
            f"selected predictions is outside results root {results_root}: {resolved}"
        ) from exc
    return resolved


def batch_directory(results_root: Path) -> Path:
    return results_root / "_serial_scoring" / SCORER_PROTOCOL


def read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def read_jsonl(path: Path) -> tuple[list[Any] | None, str | None]:
    rows: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    return None, f"line {line_number}: {exc}"
    except (OSError, UnicodeError) as exc:
        return None, str(exc)
    return rows, None


def indexed_rows_error(
    rows: list[Any],
    *,
    expected_size: int,
    allow_duplicates: bool,
) -> str | None:
    indices: list[int] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            return f"row {position} is not a JSON object"
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            return f"row {position} has invalid index"
        indices.append(index)
    if not allow_duplicates and len(indices) != len(set(indices)):
        return "duplicate indices are present"
    if set(indices) != set(range(expected_size)):
        return f"indices do not cover 0..{expected_size - 1}"
    if not allow_duplicates and len(rows) != expected_size:
        return f"expected {expected_size} rows, found {len(rows)}"
    return None


def complete_state_errors(
    predictions: Path,
    score_dir: Path,
    *,
    expected_protocol: str = SCORER_PROTOCOL,
) -> list[str]:
    errors: list[str] = []
    paths = {name: score_dir / name for name in CANONICAL_SCORE_FILES}
    summary_path = paths["summary.json"]
    if summary_path.is_file():
        summary, summary_error = read_json(summary_path)
        if summary_error is not None or not isinstance(summary, dict):
            return [
                f"summary.json is invalid: {summary_error or 'not a JSON object'}"
            ]
    else:
        summary = None

    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing canonical artifact {name}")
    if errors:
        return errors

    assert isinstance(summary, dict)

    expected_types = OFFICIAL_QUANT_TYPES | OFFICIAL_QUAL_TYPES
    gates = summary.get("publication_gates")
    official_types = summary.get("official_types")
    official_type_counts_are_valid = (
        isinstance(official_types, dict)
        and set(official_types) == expected_types
        and all(
            isinstance(item, dict)
            and isinstance(item.get("count"), int)
            and not isinstance(item.get("count"), bool)
            and item["count"] > 0
            for item in official_types.values()
        )
        and sum(item["count"] for item in official_types.values())
        == MSMU_OFFICIAL_TEST_SIZE
    )
    summary_checks = {
        "protocol": summary.get("protocol") == expected_protocol,
        "publishable": summary.get("publishable") is True,
        "status": summary.get("status") == "complete",
        "result_kind": summary.get("result_kind")
        == "official-compatible internal score",
        "num_samples": summary.get("num_samples") == MSMU_OFFICIAL_TEST_SIZE,
        "num_judge_failures": summary.get("num_judge_failures") == 0,
        "missing_official_types": summary.get("missing_official_types") == [],
        "publication_gate_failures": summary.get("publication_gate_failures") == [],
        "publication_gates": (
            isinstance(gates, dict)
            and set(gates) == EXPECTED_PUBLICATION_GATES
            and all(value is True for value in gates.values())
        ),
        "official_types": official_type_counts_are_valid,
        "official_macro8_accuracy": isinstance(
            summary.get("official_macro8_accuracy"), (int, float)
        )
        and not isinstance(summary.get("official_macro8_accuracy"), bool),
    }
    errors.extend(
        f"summary gate is invalid: {name}"
        for name, passed in summary_checks.items()
        if not passed
    )

    validation, validation_error = read_json(paths["prediction_validation.json"])
    if validation_error is not None or not isinstance(validation, dict):
        errors.append(
            "prediction_validation.json is invalid: "
            f"{validation_error or 'not a JSON object'}"
        )
    else:
        validation_checks = {
            "passed": validation.get("passed") is True,
            "allow_subset": validation.get("allow_subset") is False,
            "official_test_size": validation.get("official_test_size")
            == MSMU_OFFICIAL_TEST_SIZE,
            "num_prediction_rows": validation.get("num_prediction_rows")
            == MSMU_OFFICIAL_TEST_SIZE,
            "num_unique_indices": validation.get("num_unique_indices")
            == MSMU_OFFICIAL_TEST_SIZE,
            "errors": validation.get("errors") == [],
        }
        errors.extend(
            f"score validator is invalid: {name}"
            for name, passed in validation_checks.items()
            if not passed
        )

    scored_rows, scored_error = read_jsonl(paths["scored_rows.jsonl"])
    if scored_error is not None or scored_rows is None:
        errors.append(f"scored_rows.jsonl is invalid: {scored_error}")
    else:
        row_error = indexed_rows_error(
            scored_rows,
            expected_size=MSMU_OFFICIAL_TEST_SIZE,
            allow_duplicates=False,
        )
        if row_error:
            errors.append(f"scored_rows.jsonl is invalid: {row_error}")

    failures, failures_error = read_jsonl(paths["judge_failures.jsonl"])
    if failures_error is not None or failures is None:
        errors.append(f"judge_failures.jsonl is invalid: {failures_error}")
    elif failures:
        errors.append(f"judge_failures.jsonl contains {len(failures)} row(s)")

    cache_rows, cache_error = read_jsonl(paths["judge_cache.jsonl"])
    if cache_error is not None or cache_rows is None:
        errors.append(f"judge_cache.jsonl is invalid: {cache_error}")
    else:
        cache_index_error = indexed_rows_error(
            cache_rows,
            expected_size=MSMU_OFFICIAL_TEST_SIZE,
            allow_duplicates=True,
        )
        if cache_index_error:
            errors.append(f"judge_cache.jsonl is invalid: {cache_index_error}")

    if paths["score.log"].stat().st_size == 0:
        errors.append("score.log is empty")
    if predictions.parent.name != expected_protocol:
        errors.append("prediction parent no longer matches the scorer protocol")
    return errors


def classify_prediction(predictions: Path) -> ResultState:
    score_dir = predictions.parent / "scores" / SCORER_PROTOCOL
    if predictions.parent.name != SCORER_PROTOCOL:
        return ResultState(
            predictions=predictions,
            score_dir=score_dir,
            state="excluded_protocol",
            included=False,
            reason=(
                f"prediction parent is {predictions.parent.name!r}, "
                f"not current scorer protocol {SCORER_PROTOCOL!r}"
            ),
        )

    summary_path = score_dir / "summary.json"
    if summary_path.exists():
        errors = complete_state_errors(predictions, score_dir)
        if not errors:
            return ResultState(
                predictions=predictions,
                score_dir=score_dir,
                state="complete",
                included=False,
                reason="canonical summary and all publication artifacts are complete",
            )
        return ResultState(
            predictions=predictions,
            score_dir=score_dir,
            state="retry",
            included=True,
            reason="; ".join(errors),
        )

    cache_path = score_dir / "judge_cache.jsonl"
    if cache_path.exists():
        cache_rows, cache_error = read_jsonl(cache_path)
        if cache_error is not None or cache_rows is None:
            return ResultState(
                predictions=predictions,
                score_dir=score_dir,
                state="retry",
                included=True,
                reason=f"judge cache is invalid and no successful summary exists: {cache_error}",
            )
        if cache_rows:
            unique_indices = {
                row.get("index")
                for row in cache_rows
                if isinstance(row, dict)
                and isinstance(row.get("index"), int)
                and not isinstance(row.get("index"), bool)
            }
            return ResultState(
                predictions=predictions,
                score_dir=score_dir,
                state="resume",
                included=True,
                reason=(
                    f"judge cache has {len(unique_indices)} unique index entry/entries; "
                    "no successful summary exists"
                ),
            )

    scoring_footprint = score_dir.is_dir() and any(score_dir.iterdir())
    if scoring_footprint:
        return ResultState(
            predictions=predictions,
            score_dir=score_dir,
            state="retry",
            included=True,
            reason="incomplete scoring artifacts exist without a reusable judge cache",
        )
    return ResultState(
        predictions=predictions,
        score_dir=score_dir,
        state="new",
        included=True,
        reason="no scoring artifacts exist",
    )


def discover_results(
    results_root: Path,
    selected_predictions: Path | None = None,
) -> list[ResultState]:
    if selected_predictions is not None:
        predictions = [selected_predictions]
    else:
        predictions = sorted(
            (
                path.resolve()
                for path in results_root.rglob("predictions.jsonl")
                if path.is_file()
            ),
            key=lambda path: str(path),
        )
    return [classify_prediction(path) for path in predictions]


def print_listing(states: Iterable[ResultState]) -> None:
    print("state\tincluded\treason\tpredictions\tscore_dir")
    for state in states:
        print(
            "\t".join(
                (
                    state.state,
                    "yes" if state.included else "no",
                    tsv_value(state.reason),
                    tsv_value(str(state.predictions)),
                    tsv_value(str(state.score_dir)),
                )
            )
        )


def print_status(states: Iterable[ResultState]) -> None:
    counts = {name: 0 for name in STATE_NAMES}
    for state in states:
        counts[state.state] += 1
    for name in STATE_NAMES:
        print(f"{name}\t{counts[name]}")
    print(f"pending\t{sum(counts[name] for name in PENDING_STATES)}")
    print(f"discovered\t{sum(counts.values())}")


def tsv_value(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_status(path: Path, run_id: str, states: Iterable[ResultState]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            "run_id\tstate\tincluded\tresult\treason\tpredictions\tscore_dir\n"
        )
        for state in states:
            handle.write(
                "\t".join(
                    (
                        run_id,
                        state.state,
                        "yes" if state.included else "no",
                        tsv_value(state.result),
                        tsv_value(state.reason),
                        tsv_value(str(state.predictions)),
                        tsv_value(str(state.score_dir)),
                    )
                )
                + "\n"
            )
    temporary.replace(path)


def judge_configuration() -> tuple[str, str, str, float]:
    base_url = os.environ.get("JUDGE_BASE_URL", "").rstrip("/")
    model = os.environ.get("JUDGE_MODEL_NAME", "").strip()
    api_key = os.environ.get("API_KEY", "local")
    timeout_raw = os.environ.get("JUDGE_PREFLIGHT_TIMEOUT_SECONDS", "10")
    if not base_url:
        raise ConfigurationError("JUDGE_BASE_URL is missing from .env.server")
    if not model:
        raise ConfigurationError("JUDGE_MODEL_NAME is missing from .env.server")
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ConfigurationError(
            "JUDGE_PREFLIGHT_TIMEOUT_SECONDS must be a positive number"
        ) from exc
    if timeout <= 0:
        raise ConfigurationError(
            "JUDGE_PREFLIGHT_TIMEOUT_SECONDS must be a positive number"
        )
    return base_url, model, api_key, timeout


def check_judge_ready(
    *,
    base_url: str,
    expected_model: str,
    api_key: str,
    timeout: float,
) -> None:
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        UnicodeError,
        ValueError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as exc:
        raise JudgePreflightError(
            f"judge readiness failed at {base_url}/models: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise JudgePreflightError(
            f"judge readiness returned invalid JSON at {base_url}/models"
        )
    served_models = {
        item.get("id")
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if expected_model not in served_models:
        rendered = ", ".join(sorted(served_models)) or "<none>"
        raise JudgePreflightError(
            f"judge model {expected_model!r} is not served at {base_url}; "
            f"available models: {rendered}"
        )


def validate_positive_integer_environment(name: str) -> None:
    raw = os.environ.get(name)
    if raw is None:
        return
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")


def run_scorer(
    state: ResultState,
    *,
    score_script: Path,
    repository: Path,
    dataset_root: Path,
    base_url: str,
    model: str,
    logger: BatchLogger,
) -> int:
    environment = dict(os.environ)
    environment.update(
        {
            "PREDICTIONS": str(state.predictions),
            "OUTPUT_DIR": str(state.score_dir),
            "DATASET_ROOT": str(dataset_root),
            "BASE_URL": base_url,
            "JUDGE_MODEL_NAME": model,
        }
    )
    process = subprocess.Popen(
        ["bash", str(score_script)],
        cwd=repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            logger.write(line)
        return_code = process.wait()
        process.stdout.close()
        return return_code
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.stdout is not None:
            process.stdout.close()
        raise


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def dry_run(states: list[ResultState], results_root: Path) -> int:
    pending = [state for state in states if state.state in PENDING_STATES]
    print(f"[msmu-score-batch] protocol={SCORER_PROTOCOL}")
    print(f"[msmu-score-batch] results_root={results_root}")
    if not pending:
        print("[msmu-score-batch] no pending results; no judge/scorer action was taken")
        return 0
    for position, state in enumerate(pending, start=1):
        print(
            f"[msmu-score-batch] dry-run order={position} state={state.state} "
            f"predictions={state.predictions} output_dir={state.score_dir}"
        )
    print(
        f"[msmu-score-batch] dry-run pending={len(pending)}; "
        "no judge/scorer action was taken"
    )
    return 0


def execute_batch(
    *,
    results_root: Path,
    selected_predictions: Path | None,
    repository: Path,
    score_script: Path,
) -> int:
    control_dir = batch_directory(results_root)
    runs_dir = control_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    lock = BatchLock(control_dir / "lock")

    with lock:
        frozen = discover_results(results_root, selected_predictions)
        run_id = run_id_now()
        candidates_path = runs_dir / f"{run_id}.candidates.jsonl"
        log_path = runs_dir / f"{run_id}.log"
        write_jsonl_atomic(candidates_path, (state.as_json() for state in frozen))
        current = [
            replace(
                state,
                result="pending" if state.state in PENDING_STATES else "skipped",
            )
            for state in frozen
        ]
        write_status(control_dir / "status.tsv", run_id, current)
        pending_positions = [
            index for index, state in enumerate(current) if state.state in PENDING_STATES
        ]

        with BatchLogger(log_path) as logger:
            logger.write(f"[msmu-score-batch] run_id={run_id}")
            logger.write(f"[msmu-score-batch] protocol={SCORER_PROTOCOL}")
            logger.write(f"[msmu-score-batch] results_root={results_root}")
            logger.write(f"[msmu-score-batch] candidates={candidates_path}")
            if not pending_positions:
                logger.write(
                    "[msmu-score-batch] no pending results; "
                    "no judge/scorer action was taken"
                )
                return 0

            try:
                dataset_root = resolve_dataset_root()
                validate_positive_integer_environment("WORKERS")
                validate_positive_integer_environment("RETRIES")
                base_url, model, api_key, timeout = judge_configuration()
            except ConfigurationError as exc:
                logger.write(f"[msmu-score-batch] configuration error: {exc}")
                return 2

            try:
                check_judge_ready(
                    base_url=base_url,
                    expected_model=model,
                    api_key=api_key,
                    timeout=timeout,
                )
                logger.write(
                    f"[msmu-score-batch] judge ready endpoint={base_url} model={model}"
                )
            except JudgePreflightError as exc:
                first_position = pending_positions[0]
                current[first_position] = replace(
                    current[first_position],
                    result="judge_preflight_failed",
                    reason=str(exc),
                )
                write_status(control_dir / "status.tsv", run_id, current)
                logger.write(f"[msmu-score-batch] judge preflight failed: {exc}")
                return 4
            except KeyboardInterrupt:
                first_position = pending_positions[0]
                current[first_position] = replace(
                    current[first_position],
                    result="interrupted",
                )
                write_status(control_dir / "status.tsv", run_id, current)
                logger.write("[msmu-score-batch] interrupted by user")
                return 130

            for order, position in enumerate(pending_positions, start=1):
                state = current[position]
                logger.write(
                    f"[msmu-score-batch] START order={order}/{len(pending_positions)} "
                    f"state={state.state} predictions={state.predictions} "
                    f"output_dir={state.score_dir}"
                )
                try:
                    check_judge_ready(
                        base_url=base_url,
                        expected_model=model,
                        api_key=api_key,
                        timeout=timeout,
                    )
                except JudgePreflightError as exc:
                    current[position] = replace(
                        state,
                        result="judge_preflight_failed",
                        reason=str(exc),
                    )
                    write_status(control_dir / "status.tsv", run_id, current)
                    logger.write(f"[msmu-score-batch] judge preflight failed: {exc}")
                    return 4
                except KeyboardInterrupt:
                    current[position] = replace(state, result="interrupted")
                    write_status(control_dir / "status.tsv", run_id, current)
                    logger.write("[msmu-score-batch] interrupted by user")
                    return 130

                try:
                    return_code = run_scorer(
                        state,
                        score_script=score_script,
                        repository=repository,
                        dataset_root=dataset_root,
                        base_url=base_url,
                        model=model,
                        logger=logger,
                    )
                except KeyboardInterrupt:
                    current[position] = replace(state, result="interrupted")
                    write_status(control_dir / "status.tsv", run_id, current)
                    logger.write("[msmu-score-batch] interrupted by user")
                    return 130

                if return_code in (130, -signal.SIGINT):
                    current[position] = replace(state, result="interrupted")
                    write_status(control_dir / "status.tsv", run_id, current)
                    logger.write("[msmu-score-batch] scorer was interrupted")
                    return 130
                if return_code != 0:
                    current[position] = replace(
                        state,
                        result=f"scorer_failed_exit_{return_code}",
                    )
                    write_status(control_dir / "status.tsv", run_id, current)
                    logger.write(
                        f"[msmu-score-batch] FAIL scorer_exit={return_code}; "
                        "later results were not started"
                    )
                    return 1

                refreshed = classify_prediction(state.predictions)
                if refreshed.state != "complete":
                    current[position] = replace(
                        refreshed,
                        result="publication_gates_incomplete",
                    )
                    write_status(control_dir / "status.tsv", run_id, current)
                    logger.write(
                        "[msmu-score-batch] FAIL scorer exited 0 but canonical "
                        f"publication artifacts are incomplete: {refreshed.reason}"
                    )
                    return 1
                current[position] = replace(refreshed, result="complete")
                write_status(control_dir / "status.tsv", run_id, current)
                logger.write(
                    f"[msmu-score-batch] PASS order={order}/{len(pending_positions)} "
                    f"predictions={state.predictions}"
                )

            logger.write(
                f"[msmu-score-batch] COMPLETE scored={len(pending_positions)}"
            )
            return 0


def check_configuration(results_root: Path) -> int:
    dataset_root = resolve_dataset_root()
    validate_positive_integer_environment("WORKERS")
    validate_positive_integer_environment("RETRIES")
    base_url, model, api_key, timeout = judge_configuration()
    lock = BatchLock(batch_directory(results_root) / "lock")
    with lock:
        check_judge_ready(
            base_url=base_url,
            expected_model=model,
            api_key=api_key,
            timeout=timeout,
        )
    print(f"[msmu-score-batch] CHECK results_root={results_root}")
    print(f"[msmu-score-batch] CHECK dataset_root={dataset_root}")
    print(f"[msmu-score-batch] CHECK lock=available")
    print(f"[msmu-score-batch] CHECK judge_endpoint={base_url}")
    print(f"[msmu-score-batch] CHECK judge_model={model}")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    repository: Path | None = None,
    score_script: Path | None = None,
) -> int:
    try:
        args = parse_args(argv)
        results_root = resolve_results_root(args.results_root)
        selected_predictions = resolve_selected_predictions(
            args.predictions,
            results_root,
        )
        states = discover_results(results_root, selected_predictions)
        if args.list_results:
            print_listing(states)
            return 0
        if args.status:
            print_status(states)
            return 0
        if args.check:
            return check_configuration(results_root)
        if os.environ.get("MANUAL_DRY_RUN", "0") == "1":
            return dry_run(states, results_root)

        resolved_repository = (
            repository.resolve()
            if repository is not None
            else Path(__file__).resolve().parents[2]
        )
        resolved_score_script = (
            score_script.resolve()
            if score_script is not None
            else resolved_repository / "scripts" / "msmu" / "score_predictions.sh"
        )
        if not resolved_score_script.is_file():
            raise ConfigurationError(
                f"canonical scorer wrapper does not exist: {resolved_score_script}"
            )
        return execute_batch(
            results_root=results_root,
            selected_predictions=selected_predictions,
            repository=resolved_repository,
            score_script=resolved_score_script,
        )
    except ConfigurationError as exc:
        print(f"[msmu-score-batch] configuration error: {exc}", file=sys.stderr)
        return 2
    except LockBusyError as exc:
        print(f"[msmu-score-batch] lock unavailable: {exc}", file=sys.stderr)
        return 4
    except JudgePreflightError as exc:
        print(f"[msmu-score-batch] judge preflight failed: {exc}", file=sys.stderr)
        return 4
    except OSError as exc:
        print(f"[msmu-score-batch] filesystem error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[msmu-score-batch] interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
