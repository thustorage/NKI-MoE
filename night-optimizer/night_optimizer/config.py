from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import FileScope, SessionConfig, ThresholdPolicy


def _require_keys(payload: dict[str, Any], keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"Missing {label} keys: {', '.join(missing)}")


def load_session_config(path: str | Path) -> SessionConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text())
    _require_keys(
        payload,
        ["session_name", "target_kernel", "objective", "scope"],
        "session config",
    )

    scope_payload = payload["scope"]
    _require_keys(scope_payload, ["allowed_paths"], "scope")

    thresholds_payload = payload.get("thresholds", {})
    thresholds = ThresholdPolicy(**thresholds_payload)
    scope = FileScope(
        allowed_paths=scope_payload["allowed_paths"],
        blocked_paths=scope_payload.get("blocked_paths", []),
        protected_roots=scope_payload.get("protected_roots", []),
    )

    return SessionConfig(
        session_name=payload["session_name"],
        target_kernel=payload["target_kernel"],
        objective=payload["objective"],
        scope=scope,
        remote_test_branch=payload.get("remote_test_branch", "dev"),
        proposer_command_template=payload.get("proposer_command_template"),
        apply_command_template=payload.get("apply_command_template"),
        validate_correctness_command=payload.get("validate_correctness_command"),
        validate_performance_command=payload.get("validate_performance_command"),
        thresholds=thresholds,
        artifact_root=payload.get("artifact_root", "night-optimizer/runtime"),
        max_fix_iterations=int(payload.get("max_fix_iterations", 3)),
        fixed_baseline_latency_ms=(
            float(payload["fixed_baseline_latency_ms"])
            if payload.get("fixed_baseline_latency_ms") is not None
            else None
        ),
    )


def dump_session_config(config: SessionConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_name": config.session_name,
        "target_kernel": config.target_kernel,
        "objective": config.objective,
        "artifact_root": config.artifact_root,
        "remote_test_branch": config.remote_test_branch,
        "proposer_command_template": config.proposer_command_template,
        "apply_command_template": config.apply_command_template,
        "validate_correctness_command": config.validate_correctness_command,
        "validate_performance_command": config.validate_performance_command,
        "max_fix_iterations": config.max_fix_iterations,
        "fixed_baseline_latency_ms": config.fixed_baseline_latency_ms,
        "scope": {
            "allowed_paths": config.scope.allowed_paths,
            "blocked_paths": config.scope.blocked_paths,
            "protected_roots": config.scope.protected_roots,
        },
        "thresholds": {
            "min_cosine_similarity": config.thresholds.min_cosine_similarity,
            "max_abs_error": config.thresholds.max_abs_error,
            "max_rel_error": config.thresholds.max_rel_error,
            "min_speedup": config.thresholds.min_speedup,
            "min_successful_runs": config.thresholds.min_successful_runs,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
