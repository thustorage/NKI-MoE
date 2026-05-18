from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class InsightStatus(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class AttemptStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    REVIEW_REQUIRED = "review_required"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


class ReviewVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_HUMAN = "needs_human"


@dataclass(frozen=True)
class ThresholdPolicy:
    min_cosine_similarity: float | None = 0.99
    max_abs_error: float | None = 1e-4
    max_rel_error: float | None = None
    min_speedup: float | None = None
    min_successful_runs: int = 1


@dataclass(frozen=True)
class FileScope:
    allowed_paths: list[str]
    blocked_paths: list[str] = field(default_factory=list)
    protected_roots: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionConfig:
    session_name: str
    target_kernel: str
    objective: str
    scope: FileScope
    remote_test_branch: str = "dev"
    proposer_command_template: str | None = None
    apply_command_template: str | None = None
    validate_correctness_command: str | None = None
    validate_performance_command: str | None = None
    thresholds: ThresholdPolicy = field(default_factory=ThresholdPolicy)
    artifact_root: str = "night-optimizer/runtime"
    max_fix_iterations: int = 3
    # Static baseline latency (ms) against which `speedup` is computed for
    # round 1 before any accepted attempt exists in this session's sqlite.
    # Supply this when starting a fresh session to avoid the "no baseline
    # -> speedup=None -> first candidate auto-accepts" trap. Runtime will
    # still prefer the smallest latency among historical accepted
    # attempts once such history accumulates.
    fixed_baseline_latency_ms: float | None = None


@dataclass
class CandidatePatch:
    changed_files: list[str]
    diff_summary: str
    intent: str
    repo_root: str | None = None
    base_ref: str | None = None
    base_commit: str | None = None
    head_commit: str | None = None
    head_branch: str | None = None
    patch_hash: str | None = None
    diff_text_path: str | None = None
    worklog: list[str] = field(default_factory=list)


@dataclass
class AttemptMetrics:
    speedup: float | None = None
    latency_ms: float | None = None
    cosine_similarity: float | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    successful_runs: int = 0


@dataclass
class ReviewEvidence:
    correctness_report: str | None = None
    performance_report: str | None = None
    diff_summary: str | None = None
    review_notes: str | None = None
    raw_artifacts: list[str] = field(default_factory=list)


@dataclass
class ExecutionRecord:
    execution_id: str
    session_name: str
    attempt_id: str
    command: str
    result_dir: str
    branch: str
    head_commit: str
    status: str
    remote_exit_code: int | None = None
    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    log_path: str | None = None
    summary_path: str | None = None
    fetched_artifacts: list[str] = field(default_factory=list)


@dataclass
class AgentTaskRecord:
    task_id: str
    session_name: str
    attempt_id: str | None
    title: str
    prompt_path: str
    output_path: str
    summary_path: str | None = None
    created_at: str = field(default_factory=utc_now)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProposerLaunch:
    task: AgentTaskRecord
    prompt_path: str
    output_path: str
    command: str
    summary_path: str


@dataclass
class AttemptRecord:
    attempt_id: str
    session_name: str
    parent_attempt_id: str | None
    status: AttemptStatus
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    summary: str = ""
    candidate: CandidatePatch | None = None
    metrics: AttemptMetrics = field(default_factory=AttemptMetrics)
    evidence: ReviewEvidence = field(default_factory=ReviewEvidence)
    rejection_reasons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class InsightRecord:
    insight_id: str
    session_name: str
    source_attempt_id: str | None
    title: str
    detail: str
    status: InsightStatus = InsightStatus.PROPOSED
    confidence: float = 0.0
    created_at: str = field(default_factory=utc_now)
    validated_at: str | None = None
    supersedes: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ReviewDecision:
    verdict: ReviewVerdict
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class OvernightRoundResult:
    round_index: int
    proposer_output_path: str
    apply_output_path: str
    apply_summary_path: str | None = None
    correctness_execution_id: str | None = None
    correctness_summary_path: str | None = None
    performance_execution_id: str | None = None
    performance_summary_path: str | None = None
    review_verdict: str | None = None
    review_reasons: list[str] = field(default_factory=list)
    status: str = "pending"
    fix_iterations: int = 0
    fix_output_paths: list[str] = field(default_factory=list)


def to_plain_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
