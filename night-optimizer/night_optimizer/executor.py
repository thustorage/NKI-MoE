from __future__ import annotations

import json
import subprocess
from pathlib import Path
from uuid import uuid4

from .models import (
    AttemptMetrics,
    AttemptRecord,
    ExecutionRecord,
    ReviewEvidence,
    SessionConfig,
    utc_now,
)
from .repository import GitInspector
from .results import parse_log_metrics, parse_remote_exit_code


class ExecutionError(RuntimeError):
    pass


class RemoteExecutor:
    def __init__(
        self, config: SessionConfig, repo_root: str | Path, state_store
    ) -> None:
        self.config = config
        self.repo_root = Path(repo_root)
        self.state = state_store
        self.git = GitInspector(self.repo_root)

    def run_attempt(
        self,
        attempt: AttemptRecord,
        result_root: str | Path,
        remote_test_command: str,
    ) -> ExecutionRecord:
        if not remote_test_command:
            raise ExecutionError("remote_test_command is required to run an attempt")
        if not attempt.candidate or not attempt.candidate.head_commit:
            raise ExecutionError("Attempt is missing recorded head_commit")

        worktree = self.git.get_worktree_state()
        if not worktree.is_clean:
            raise ExecutionError(
                "Working tree must be clean before remote execution: "
                + "; ".join(worktree.status_entries)
            )
        if worktree.head_commit != attempt.candidate.head_commit:
            raise ExecutionError(
                f"HEAD commit {worktree.head_commit} does not match attempt head_commit {attempt.candidate.head_commit}"
            )
        if worktree.branch != self.config.remote_test_branch:
            raise ExecutionError(
                f"Current branch {worktree.branch} does not match configured remote_test_branch {self.config.remote_test_branch}"
            )
        if (
            attempt.candidate.head_branch
            and worktree.branch != attempt.candidate.head_branch
        ):
            raise ExecutionError(
                f"Current branch {worktree.branch} does not match attempt head_branch {attempt.candidate.head_branch}"
            )

        result_dir = Path(result_root)
        result_dir.mkdir(parents=True, exist_ok=True)
        execution_id = str(uuid4())
        log_path = result_dir / "remote_test.log"
        summary_path = result_dir / "execution_summary.json"
        command = [
            "./remote_test.sh",
            "--push",
            "--output",
            log_path.as_posix(),
            "--commit-message",
            f"night-optimizer: validate attempt {attempt.attempt_id}",
            remote_test_command,
        ]

        execution = ExecutionRecord(
            execution_id=execution_id,
            session_name=attempt.session_name,
            attempt_id=attempt.attempt_id,
            command=" ".join(command),
            result_dir=result_dir.as_posix(),
            branch=worktree.branch,
            head_commit=worktree.head_commit,
            status="running",
            log_path=log_path.as_posix(),
            summary_path=summary_path.as_posix(),
        )
        self.state.upsert_execution(execution)

        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            check=False,
            text=True,
        )

        log_text = log_path.read_text() if log_path.exists() else ""
        metrics = parse_log_metrics(log_text)
        remote_exit_code = parse_remote_exit_code(log_text)

        summary_payload = {
            "execution_id": execution.execution_id,
            "attempt_id": attempt.attempt_id,
            "command": execution.command,
            "branch": execution.branch,
            "head_commit": execution.head_commit,
            "remote_exit_code": remote_exit_code,
            "process_exit_code": completed.returncode,
            "metrics": metrics,
            "completed_at": utc_now(),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

        execution.status = "passed" if completed.returncode == 0 else "failed"
        execution.remote_exit_code = (
            remote_exit_code if remote_exit_code is not None else completed.returncode
        )
        execution.completed_at = summary_payload["completed_at"]
        execution.fetched_artifacts = []
        self.state.upsert_execution(execution)
        return execution

    def build_attempt_metrics(self, execution: ExecutionRecord) -> AttemptMetrics:
        summary_payload = json.loads(Path(execution.summary_path).read_text())
        metrics_payload = summary_payload.get("metrics") or {}
        return AttemptMetrics(
            speedup=self._to_float(metrics_payload.get("speedup")),
            latency_ms=self._to_float(metrics_payload.get("latency_ms")),
            cosine_similarity=self._to_float(metrics_payload.get("cosine_similarity")),
            abs_error=self._to_float(metrics_payload.get("abs_error")),
            rel_error=self._to_float(metrics_payload.get("rel_error")),
            successful_runs=self._to_int(metrics_payload.get("successful_runs")),
        )

    def build_review_evidence(
        self, execution: ExecutionRecord, attempt: AttemptRecord
    ) -> ReviewEvidence:
        return ReviewEvidence(
            correctness_report=execution.summary_path,
            performance_report=execution.summary_path,
            diff_summary=attempt.candidate.diff_summary if attempt.candidate else None,
            review_notes=f"Execution status: {execution.status}",
            raw_artifacts=[execution.log_path, execution.summary_path],
        )

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _to_int(value: object) -> int:
        if value is None:
            return 0
        return int(value)
