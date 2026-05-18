from __future__ import annotations

import json
import re
from pathlib import Path

_FLOAT_PATTERN = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def _parse_last_float(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def parse_remote_exit_code(log_text: str) -> int | None:
    match = re.search(
        r"Remote command (?:completed successfully|failed) \(exit code: (\d+)\)",
        log_text,
    )
    if match:
        return int(match.group(1))
    return None


def parse_log_metrics(log_text: str) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "accuracy": None,
        "speedup": None,
        "latency_ms": None,
        "final_score": None,
        "cosine_similarity": None,
        "abs_error": None,
        "rel_error": None,
        "successful_runs": 0,
    }

    accuracy_match = re.search(r"Accuracy:\s*([0-9.]+|True|False)", log_text)
    if accuracy_match:
        value = accuracy_match.group(1)
        if value == "True":
            metrics["accuracy"] = 1.0
        elif value == "False":
            metrics["accuracy"] = 0.0
        else:
            metrics["accuracy"] = float(value)

    metrics["speedup"] = _parse_last_float(
        [rf"reduced_latency:\s*{_FLOAT_PATTERN}"], log_text
    )
    metrics["latency_ms"] = _parse_last_float(
        [
            rf"Latency:\s*{_FLOAT_PATTERN}",
            rf'"latency_ms_avg"\s*:\s*{_FLOAT_PATTERN}',
            rf'"latency_ms_p50"\s*:\s*{_FLOAT_PATTERN}',
        ],
        log_text,
    )
    metrics["final_score"] = _parse_last_float(
        [rf"Final Score:\s*{_FLOAT_PATTERN}"], log_text
    )
    metrics["cosine_similarity"] = _parse_last_float(
        [
            rf"cosine_sim\s*[=:]\s*{_FLOAT_PATTERN}",
            rf"cosine(?: similarity)?\s*[=:]\s*{_FLOAT_PATTERN}",
            rf"cos(?:ine)?\s*[=:]\s*{_FLOAT_PATTERN}",
        ],
        log_text,
    )
    metrics["abs_error"] = _parse_last_float(
        [
            rf"max_abs_err\s*[=:]\s*{_FLOAT_PATTERN}",
            rf"abs(?:olute)?(?:_error| error)?\s*[=:]\s*{_FLOAT_PATTERN}",
            rf"max_diff\s*[=:]\s*{_FLOAT_PATTERN}",
        ],
        log_text,
    )
    metrics["rel_error"] = _parse_last_float(
        [
            rf"max_rel_err\s*[=:]\s*{_FLOAT_PATTERN}",
            rf"rel(?:ative)?(?:_error| error)?\s*[=:]\s*{_FLOAT_PATTERN}",
        ],
        log_text,
    )

    # Hardware profiler latency (neuron-profile total_time, in microseconds).
    # When present, override the wall-clock latency_ms with the accurate
    # on-device kernel time so threshold checks use hardware-level data.
    profiler_us = _parse_last_float(
        [rf"profiler_total_time_us\s*[=:]\s*{_FLOAT_PATTERN}"],
        log_text,
    )
    if profiler_us is not None:
        metrics["latency_ms"] = profiler_us / 1000.0
        metrics["profiler_latency_us"] = profiler_us

    if "Passed logits validation" in log_text:
        metrics["successful_runs"] = 1
    elif re.search(r"\bPASS\s+thresholds\b", log_text):
        # ops/qkv/test_qkv_precision.py prints "... PASS  thresholds(cos>=..., abs<=...)"
        # when a precision case passes its cos/abs thresholds.
        metrics["successful_runs"] = 1

    return metrics


def parse_benchmark_report(path: str | Path) -> dict[str, float | int | None]:
    payload = json.loads(Path(path).read_text())
    report = payload.get("e2e_model") or {}
    metrics: dict[str, float | int | None] = {
        "latency_ms": None,
        "successful_runs": None,
    }

    if report.get("latency_ms_p99") is not None:
        metrics["latency_ms"] = float(report["latency_ms_p99"])
    elif report.get("latency_ms") is not None:
        metrics["latency_ms"] = float(report["latency_ms"])

    for key in ("n_runs", "num_runs"):
        if report.get(key) is not None:
            metrics["successful_runs"] = int(report[key])
            break

    if metrics["successful_runs"] is None and metrics["latency_ms"] is not None:
        metrics["successful_runs"] = 1

    return metrics


def find_existing_artifact(path: str | Path) -> str | None:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve().as_posix()
    return None


def classify_remote_failure(log_text: str) -> tuple[str, str]:
    """Classify why a remote validation run failed.

    Distinguishes between:
      - "compile_failed": NKI / HLO compiler rejected the kernel (no test
        actually ran, so missing metrics is a symptom, not a root cause).
      - "test_crashed": test harness (pytest / test_qkv_*.py) raised before
        any metric line was emitted (e.g. assertion, import error,
        RuntimeError inside wrapper).
      - "metrics_missing": test ran to completion but the parser did not
        find the expected metric lines — genuine parser drift or the test
        was skipped.
      - "remote_env_error": git sync / ssh / host-side setup failed before
        the test command even started executing.
      - "unknown_failure": fallback, exit code non-zero but none of the
        above patterns matched.

    Returns (category, human_readable_reason).
    """

    text = log_text or ""

    started_exec = "[Remote Debug] Starting command execution" in text

    # Remote environment / git sync layer: fail-fast before test execution.
    if "Not possible to fast-forward" in text:
        return (
            "remote_env_error",
            "Remote worktree git pull --ff-only failed; dev branch diverged",
        )
    if "Connection refused" in text or "Could not resolve hostname" in text:
        return ("remote_env_error", "Remote host unreachable")
    if not started_exec and ("exit code" in text or "Remote command failed" in text):
        return (
            "remote_env_error",
            "Remote command never started (environment setup failed)",
        )

    # NKI / HLO compiler-layer diagnostics (test crashed during compile).
    if "failed to specialize NKI kernel" in text:
        return ("compile_failed", "NKI frontend rejected kernel (specialize error)")
    if re.search(r"\berror:\s*unsupported expression\b", text):
        return (
            "compile_failed",
            "NKI frontend rejected an unsupported Python expression in kernel source",
        )
    if "Compiler status FAIL" in text or "Compiler status: FAIL" in text:
        return ("compile_failed", "Neuron compiler rejected HLO module")

    # Test harness crash after (or without) successful compile.
    if "Traceback (most recent call last)" in text or "AssertionError" in text:
        if "Compiler status PASS" in text:
            return (
                "test_crashed",
                "Kernel compiled but the test harness crashed before emitting metrics",
            )
        return (
            "test_crashed",
            "Test harness crashed (pre-compile Python error)",
        )

    # Exit non-zero but no traceback or compile error found.
    if "Remote command failed" in text:
        return (
            "unknown_failure",
            "Remote command returned non-zero exit but no compile/crash signature was found",
        )

    return ("unknown_failure", "Validation failed for an unknown reason")
