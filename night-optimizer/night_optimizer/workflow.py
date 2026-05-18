from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .config import dump_session_config
from .executor import RemoteExecutor
from .results import classify_remote_failure
from .models import (
    AgentTaskRecord,
    AttemptMetrics,
    AttemptRecord,
    AttemptStatus,
    CandidatePatch,
    ExecutionRecord,
    InsightRecord,
    InsightStatus,
    OvernightRoundResult,
    ProposerLaunch,
    ReviewDecision,
    ReviewEvidence,
    ReviewVerdict,
    SessionConfig,
    utc_now,
)
from .policy import ReviewPolicy
from .repository import GitInspectionError, GitInspector, GitPatchSnapshot
from .scope import ScopeValidator
from .state import StateStore


class SessionController:
    def __init__(self, config: SessionConfig, root: str | Path | None = None) -> None:
        self.config = config
        self.scope = ScopeValidator(config.scope)
        self.policy = ReviewPolicy(config)
        self.git = GitInspector.discover()
        # Anchor artifact_root to the repository root when it is a relative
        # path so that running the CLI from any cwd places runtime/ in the
        # same canonical location.
        base = Path(root or config.artifact_root)
        if not base.is_absolute():
            base = Path(self.git.repo_root) / base
        self.root = base / config.session_name
        self.state = StateStore(self.root / "state.sqlite3")
        self.executor = RemoteExecutor(config, self.git.repo_root, self.state)

    def bootstrap(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "attempts").mkdir(exist_ok=True)
        (self.root / "agents").mkdir(exist_ok=True)
        (self.root / "exports").mkdir(exist_ok=True)
        (self.root / "patches").mkdir(exist_ok=True)
        (self.root / "executions").mkdir(exist_ok=True)
        dump_session_config(self.config, self.root / "session_config.json")
        return self.root

    _RATE_LIMIT_PATTERNS = (
        "frequency limit",
        "usage exceeds",
        "rate limit",
        "rate_limit",
        "429 usage",
        "too many requests",
    )

    @classmethod
    def detect_rate_limit_or_empty(cls, output_path: str | Path | None) -> str | None:
        # Why: codebuddy and other LLM providers return HTTP 429 as a normal
        # 248-byte text body that the applier/proposer subprocess writes to
        # the output file without nonzero exit code. The previous harness
        # then treated this as "agent produced no diff" and cycled the round
        # at 30s intervals, producing 443 junk rounds in one night.
        # Surface it as a distinct failure mode so the overnight loop can
        # back off instead of hot-spinning.
        if not output_path:
            return None
        try:
            path = Path(output_path)
            if not path.exists():
                return None
            size = path.stat().st_size
        except OSError:
            return None
        if size == 0:
            return "empty agent output (0 bytes)"
        if size > 8 * 1024:
            # Real LLM transcripts are routinely >10KB; skip the grep to
            # avoid false positives on code that discusses rate limiting.
            return None
        try:
            text = path.read_text(errors="replace").lower()
        except OSError:
            return None
        for pattern in cls._RATE_LIMIT_PATTERNS:
            if pattern in text:
                return f"rate-limit response detected ({pattern!r})"
        return None

    def write_progress(
        self,
        state: str,
        round_index: int | None = None,
        reason: str | None = None,
        extra: dict | None = None,
    ) -> None:
        # Progress file is a heartbeat for external monitors (e.g. the cron
        # health check). It's tiny, frequently-overwritten JSON; never let
        # a filesystem hiccup take down the round.
        payload: dict = {
            "session_name": self.config.session_name,
            "state": state,
            "round_index": round_index,
            "reason": reason,
            "updated_at": utc_now(),
            "head_commit": None,
            "seconds_since_last_commit": None,
        }
        try:
            payload["head_commit"] = self._current_head()
        except Exception:
            pass
        last_commit_at = self._last_commit_timestamp()
        if last_commit_at is not None:
            from time import time

            payload["seconds_since_last_commit"] = max(0, int(time() - last_commit_at))
        if extra:
            payload.update(extra)
        progress_path = self.root / "overnight_progress.json"
        try:
            progress_path.write_text(json.dumps(payload, indent=2) + "\n")
        except OSError:
            pass

    def _last_commit_timestamp(self) -> float | None:
        try:
            completed = subprocess.run(
                ["git", "log", "-1", "--format=%ct", "HEAD"],
                cwd=self.git.repo_root,
                check=True,
                text=True,
                capture_output=True,
            )
            return float(completed.stdout.strip())
        except Exception:
            return None

    # Backoff schedule for 429 / empty LLM output retries.
    # 5min, 10min, 20min, then capped at 30min. No hard retry limit — the
    # overnight loop should prefer waiting out a transient quota window
    # over burning a round on a 429 (which was the failure mode that
    # produced ~443 junk rounds in a single night). Operators can Ctrl+C
    # if the provider is truly dead.
    _RATE_LIMIT_BACKOFF_SECONDS = (300, 600, 1200, 1800)
    _RATE_LIMIT_BACKOFF_CAP = 1800

    def _run_agent_with_retry_on_rate_limit(
        self,
        runner,
        output_path: Path,
        *,
        label: str,
        round_index: int | None = None,
        reset_between_retries=None,
    ):
        # Runs the LLM subprocess via `runner()` (which must invoke the
        # agent and write to `output_path`). After each invocation we
        # inspect output_path with detect_rate_limit_or_empty; if it
        # signals a 429/empty body we sleep with exponential backoff and
        # re-invoke the *same* command without advancing the round
        # counter. Returns the last CompletedProcess once a non-rate-limit
        # response is produced.
        import time as _time

        attempt = 0
        total_waited = 0
        while True:
            attempt += 1
            if attempt > 1 and reset_between_retries is not None:
                try:
                    reset_between_retries()
                except Exception:
                    pass
            completed = runner()
            reason = self.detect_rate_limit_or_empty(output_path)
            if reason is None:
                if attempt > 1:
                    # Surface the recovery in the progress file so external
                    # monitors can see we got unstuck.
                    self.write_progress(
                        state="rate_limit_recovered",
                        round_index=round_index,
                        reason=f"{label}: recovered after {attempt} attempts",
                        extra={
                            "label": label,
                            "retry_attempt": attempt,
                            "total_waited_seconds": total_waited,
                        },
                    )
                return completed
            # 429 / empty response: sleep in place, then retry.
            if attempt - 1 < len(self._RATE_LIMIT_BACKOFF_SECONDS):
                sleep_s = self._RATE_LIMIT_BACKOFF_SECONDS[attempt - 1]
            else:
                sleep_s = self._RATE_LIMIT_BACKOFF_CAP
            self.write_progress(
                state="rate_limit_retry",
                round_index=round_index,
                reason=f"{label}: {reason}; sleeping {sleep_s}s before retry #{attempt + 1}",
                extra={
                    "label": label,
                    "retry_attempt": attempt,
                    "next_sleep_seconds": sleep_s,
                    "total_waited_seconds": total_waited,
                },
            )
            _time.sleep(sleep_s)
            total_waited += sleep_s

    def inspect_scope(
        self, base_ref: str, head_ref: str = "HEAD"
    ) -> tuple[GitPatchSnapshot, list[str]]:
        snapshot = self.git.inspect_patch(base_ref=base_ref, head_ref=head_ref)
        violations = list(self.scope.validate(snapshot.changed_files).violations)
        if not snapshot.changed_files:
            violations.append(
                f"No changed files found between {snapshot.base_commit} and {snapshot.head_commit}"
            )
        return snapshot, violations

    def create_attempt(
        self, summary: str, intent: str, base_ref: str = "HEAD~1"
    ) -> AttemptRecord:
        worktree = self.git.get_worktree_state()
        snapshot, violations = self.inspect_scope(base_ref=base_ref)

        if not worktree.is_clean:
            violations.append(
                "Working tree must be clean before creating a git-backed attempt: "
                + "; ".join(worktree.status_entries)
            )

        attempt_id = str(uuid4())
        patch_path = self._write_patch_snapshot(attempt_id, snapshot.diff_text)
        status = AttemptStatus.PROPOSED if not violations else AttemptStatus.REJECTED
        attempt = AttemptRecord(
            attempt_id=attempt_id,
            session_name=self.config.session_name,
            parent_attempt_id=None,
            status=status,
            summary=summary,
            candidate=CandidatePatch(
                changed_files=snapshot.changed_files,
                diff_summary=snapshot.diff_stat,
                intent=intent,
                repo_root=snapshot.repo_root,
                base_ref=base_ref,
                base_commit=snapshot.base_commit,
                head_commit=snapshot.head_commit,
                head_branch=snapshot.branch,
                patch_hash=snapshot.patch_hash,
                diff_text_path=str(patch_path),
                worklog=[] if not violations else violations,
            ),
            rejection_reasons=[] if not violations else violations,
        )
        self.state.upsert_attempt(attempt)
        self._write_attempt_snapshot(attempt)
        return attempt

    def load_attempt(self, attempt_file: str | Path) -> AttemptRecord:
        payload = json.loads(Path(attempt_file).read_text())
        candidate_payload = payload.get("candidate") or {}
        attempt = AttemptRecord(
            attempt_id=payload["attempt_id"],
            session_name=payload["session_name"],
            parent_attempt_id=payload.get("parent_attempt_id"),
            status=AttemptStatus(payload["status"]),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            summary=payload.get("summary", ""),
            candidate=(
                CandidatePatch(
                    changed_files=candidate_payload.get("changed_files", []),
                    diff_summary=candidate_payload.get("diff_summary", ""),
                    intent=candidate_payload.get("intent", ""),
                    repo_root=candidate_payload.get("repo_root"),
                    base_ref=candidate_payload.get("base_ref"),
                    base_commit=candidate_payload.get("base_commit"),
                    head_commit=candidate_payload.get("head_commit"),
                    head_branch=candidate_payload.get("head_branch"),
                    patch_hash=candidate_payload.get("patch_hash"),
                    diff_text_path=candidate_payload.get("diff_text_path"),
                    worklog=candidate_payload.get("worklog", []),
                )
                if candidate_payload
                else None
            ),
            metrics=AttemptMetrics(**(payload.get("metrics") or {})),
            evidence=ReviewEvidence(**(payload.get("evidence") or {})),
            rejection_reasons=payload.get("rejection_reasons", []),
            tags=payload.get("tags", []),
        )
        return attempt

    def review_attempt(self, attempt: AttemptRecord) -> ReviewDecision:
        attempt.updated_at = utc_now()
        patch_reasons = self._validate_attempt_patch(attempt)
        policy_decision = self.policy.evaluate(attempt)
        reasons = [*patch_reasons, *policy_decision.reasons]
        verdict = policy_decision.verdict if not reasons else ReviewVerdict.FAIL
        decision = ReviewDecision(
            verdict=verdict, reasons=reasons, warnings=policy_decision.warnings
        )
        attempt.status = (
            AttemptStatus.ACCEPTED
            if decision.verdict.value == "pass"
            else AttemptStatus.REJECTED
        )
        attempt.rejection_reasons = list(decision.reasons)
        self.state.upsert_attempt(attempt)
        self._write_attempt_snapshot(attempt)
        return decision

    def record_insight(
        self,
        title: str,
        detail: str,
        source_attempt_id: str | None,
        status: InsightStatus = InsightStatus.PROPOSED,
        confidence: float = 0.0,
    ) -> InsightRecord:
        insight = InsightRecord(
            insight_id=str(uuid4()),
            session_name=self.config.session_name,
            source_attempt_id=source_attempt_id,
            title=title,
            detail=detail,
            status=status,
            confidence=confidence,
            validated_at=utc_now() if status == InsightStatus.VALIDATED else None,
        )
        self.state.upsert_insight(insight)
        return insight

    def record_agent_task(
        self,
        title: str,
        prompt_path: str,
        output_path: str,
        attempt_id: str | None = None,
        summary_path: str | None = None,
        tags: list[str] | None = None,
    ) -> AgentTaskRecord:
        task = AgentTaskRecord(
            task_id=str(uuid4()),
            session_name=self.config.session_name,
            attempt_id=attempt_id,
            title=title,
            prompt_path=prompt_path,
            output_path=output_path,
            summary_path=summary_path,
            tags=tags or [],
        )
        self.state.upsert_agent_task(task)
        return task

    def build_prompt_context(self) -> dict:
        validated_insights = self.state.list_validated_insight_payloads(
            self.config.session_name
        )
        return {
            "session": self.config.session_name,
            "target_kernel": self.config.target_kernel,
            "objective": self.config.objective,
            "allowed_paths": self.config.scope.allowed_paths,
            "blocked_paths": self.config.scope.blocked_paths,
            "validated_insights": validated_insights,
            "attempt_history": self.state.list_attempt_payloads(
                self.config.session_name
            ),
        }

    def export_prompt_context(self) -> Path:
        payload = self.build_prompt_context()
        export_path = self.root / "exports" / "prompt_context.json"
        export_path.write_text(json.dumps(payload, indent=2) + "\n")
        return export_path

    def build_memory_index(self, attempt_id: str | None = None) -> dict:
        validated_insights = self.state.list_validated_insight_payloads(
            self.config.session_name
        )
        attempt_history = self.state.list_attempt_payloads(self.config.session_name)
        indexed_attempt = (
            self.state.get_attempt_payload(attempt_id) if attempt_id else None
        )

        return {
            "session_name": self.config.session_name,
            "target_kernel": self.config.target_kernel,
            "objective": self.config.objective,
            "scope": {
                "allowed_paths": self.config.scope.allowed_paths,
                "blocked_paths": self.config.scope.blocked_paths,
                "protected_roots": self.config.scope.protected_roots,
            },
            "thresholds": {
                "min_cosine_similarity": self.config.thresholds.min_cosine_similarity,
                "max_abs_error": self.config.thresholds.max_abs_error,
                "max_rel_error": self.config.thresholds.max_rel_error,
                "min_speedup": self.config.thresholds.min_speedup,
                "min_successful_runs": self.config.thresholds.min_successful_runs,
            },
            "memory_handles": {
                "validated_insights": {
                    "count": len(validated_insights),
                    "items": [
                        {
                            "insight_id": item.get("insight_id"),
                            "title": item.get("title"),
                            "status": item.get("status"),
                            "confidence": item.get("confidence"),
                        }
                        for item in validated_insights
                    ],
                },
                "attempt_history": {
                    "count": len(attempt_history),
                    "items": [
                        {
                            "attempt_id": item.get("attempt_id"),
                            "status": item.get("status"),
                            "summary": item.get("summary"),
                            "created_at": item.get("created_at"),
                            "rejection_reasons": item.get("rejection_reasons", []),
                        }
                        for item in attempt_history
                    ],
                },
                "current_attempt": indexed_attempt,
            },
            "memory_instructions": {
                "rule": "Read memory only when needed. Do not assume the full memory is in the prompt.",
                "available_files": {
                    "prompt_context": (
                        self.root / "exports" / "prompt_context.json"
                    ).as_posix(),
                    "session_config": (self.root / "session_config.json").as_posix(),
                    "attempt_dir": (self.root / "attempts").as_posix(),
                },
            },
        }

    def build_proposer_context(self, attempt_id: str | None = None) -> dict:
        return self.build_memory_index(attempt_id=attempt_id)

    def build_proposer_prompt(self, attempt_id: str | None = None) -> str:
        payload = self.build_proposer_context(attempt_id=attempt_id)
        context_json = json.dumps(payload, indent=2)
        return (
            "You are the proposer for a deadline-driven NKI kernel optimization session.\n"
            "Your job is to propose the next concrete optimization attempt for the target kernel.\n\n"
            "Hard requirements:\n"
            "- Optimize only within the declared scope.\n"
            "- Do not modify tests, benchmarks, or thresholds unless explicitly permitted by the scope.\n"
            "- Do not relax correctness requirements.\n"
            "- Read memory on demand when you need prior attempts or validated insights.\n"
            "- Prefer concrete code-level changes over vague advice.\n\n"
            "Return format:\n"
            "1. Hypothesis\n"
            "2. Planned file edits\n"
            "3. Validation plan\n"
            "4. Risks\n"
            "5. Expected effect on correctness and performance\n\n"
            "Memory usage:\n"
            "- Start from memory_handles only.\n"
            "- Read exported files only if needed for the current decision.\n"
            "- Do not restate full attempt history unless it is directly relevant.\n\n"
            "Structured context:\n"
            f"{context_json}\n"
        )

    def build_applier_context(
        self, suggestion_text: str, attempt_id: str | None = None
    ) -> dict:
        memory_index = self.build_memory_index(attempt_id=attempt_id)
        return {
            "session_name": self.config.session_name,
            "target_kernel": self.config.target_kernel,
            "scope": {
                "allowed_paths": self.config.scope.allowed_paths,
            },
            "thresholds": memory_index["thresholds"],
            "attempt_id": attempt_id,
            "suggestion": suggestion_text,
            "memory_index": memory_index,
        }

    def build_applier_prompt(
        self, suggestion_text: str, attempt_id: str | None = None
    ) -> str:
        payload = self.build_applier_context(
            suggestion_text=suggestion_text, attempt_id=attempt_id
        )
        context_json = json.dumps(payload, indent=2)
        return (
            "You are the applier for a deadline-driven NKI kernel optimization session.\n"
            "Apply the provided optimization suggestion by editing files in place.\n"
            "You may use any tool or intermediate process (read files, run commands,\n"
            "create scratch files under tmp/) as long as the FINAL repository state\n"
            "obeys the hard requirements below.\n\n"
            "Hard requirements on the final state (a git diff will be checked):\n"
            "- Only files listed under scope.allowed_paths may end up modified,\n"
            "  added, or deleted.\n"
            "- Do NOT modify tests, benchmarks, thresholds, docs, skills, nkilib,\n"
            "  remote_test.sh, or .gitignore.\n"
            "- Do NOT modify the orchestrator itself (anything under\n"
            "  `night-optimizer/`). You are a kernel-optimization agent, not a\n"
            "  harness-maintenance agent. If something looks wrong in the\n"
            "  orchestrator (metrics parser, scope check, config), stop and\n"
            "  report it in your output instead of editing it.\n"
            "- Do not weaken correctness requirements.\n"
            "- Do NOT run `git commit`, `git add`, `git push`, or any command\n"
            "  that advances HEAD or rewrites history. If HEAD moves during\n"
            "  your run, the orchestrator will soft-reset your commits and\n"
            "  treat their content as your uncommitted output.\n"
            "- Do NOT run tests, benchmarks, or remote_test.sh. Validation is\n"
            "  performed independently by the orchestrator on a remote machine;\n"
            "  your local test runs (if any) are ignored.\n"
            '- Do NOT self-report metrics ("PASS", cosine_similarity,\n'
            "  max_abs_err, speedup, etc.) in your output. Such claims are\n"
            "  stripped and never forwarded to future agents.\n"
            "- Scratch work must live under tmp/ (gitignored). Anything you write\n"
            "  under ops/, nkilib/, docs/, skills/, night-optimizer/ will be\n"
            "  flagged as out-of-scope, even if .gitignore would otherwise hide it.\n"
            "- If no safe edit is possible within scope.allowed_paths, do nothing\n"
            "  and leave the worktree unchanged.\n\n"
            "Structured context:\n"
            f"{context_json}\n"
        )

    def build_fixer_context(
        self,
        intent_text: str,
        metadata: dict,
        log_tail: str,
        diff_text: str,
        attempt_id: str | None = None,
    ) -> dict:
        memory_index = self.build_memory_index(attempt_id=attempt_id)
        return {
            "session_name": self.config.session_name,
            "target_kernel": self.config.target_kernel,
            "scope": {
                "allowed_paths": self.config.scope.allowed_paths,
            },
            "thresholds": memory_index["thresholds"],
            "attempt_id": attempt_id,
            "intent": intent_text,
            "metadata_pointers": metadata,
            "remote_test_log_tail": log_tail,
            "diff_since_round_start": diff_text,
            "memory_index": memory_index,
        }

    def build_fixer_prompt(
        self,
        intent_text: str,
        metadata: dict,
        log_tail: str,
        diff_text: str,
        fix_iteration: int,
        attempt_id: str | None = None,
    ) -> str:
        payload = self.build_fixer_context(
            intent_text=intent_text,
            metadata=metadata,
            log_tail=log_tail,
            diff_text=diff_text,
            attempt_id=attempt_id,
        )
        context_json = json.dumps(payload, indent=2)
        return (
            "You are the fixer for a deadline-driven NKI kernel optimization session.\n"
            f"This is fix iteration {fix_iteration}. A previous draft implementing\n"
            "the intent below failed our remote validation. Edit files under\n"
            "scope.allowed_paths so the intent's implementation becomes correct\n"
            "(and fast). Keep the original optimization idea — do not rewrite\n"
            "from scratch unless the intent is unreachable.\n\n"
            "Hard requirements on the final state (a git diff will be checked):\n"
            "- Only files listed under scope.allowed_paths may end up modified,\n"
            "  added, or deleted.\n"
            "- Do NOT modify tests, benchmarks, thresholds, docs, skills, nkilib,\n"
            "  remote_test.sh, or .gitignore.\n"
            "- Do NOT modify the orchestrator itself (anything under\n"
            "  `night-optimizer/`). If the failure log looks like a parser,\n"
            "  scope, config, or harness bug — i.e. the kernel seems correct\n"
            "  but the pipeline reports it as failed — STOP editing, leave the\n"
            "  worktree unchanged, and write a short diagnosis to stdout\n"
            "  starting with `HARNESS_BUG:` so the human operator can fix it.\n"
            "  Fixing orchestrator code is never your job.\n"
            "- Do not weaken correctness requirements.\n"
            "- Do NOT run `git commit`, `git add`, `git push`, or any command\n"
            "  that advances HEAD or rewrites history. If HEAD moves during\n"
            "  your run, the orchestrator will soft-reset your commits.\n"
            "- Do NOT run tests, benchmarks, or remote_test.sh. Validation is\n"
            "  performed independently by the orchestrator on the next turn.\n"
            '- Do NOT self-report metrics ("PASS", cosine_similarity, etc.).\n'
            "- Scratch work must live under tmp/ (gitignored). Anything you write\n"
            "  under ops/, nkilib/, docs/, skills/, night-optimizer/ will be\n"
            "  flagged as out-of-scope.\n"
            "- If you truly cannot make progress, leave the worktree unchanged.\n\n"
            "Inputs provided below:\n"
            "- `intent`: the original proposer text. Do NOT redesign away from it.\n"
            "- `metadata_pointers`: absolute paths to round_start_commit,\n"
            "  pre_apply_commit, current_head, full remote_test.log, last\n"
            "  execution summary, and the original suggestion text. Read them\n"
            "  with your file tools when the inline excerpts are not enough.\n"
            "- `remote_test_log_tail`: last 200 lines of the failing run.\n"
            "- `diff_since_round_start`: git diff of scope.allowed_paths\n"
            "  relative to the round-start commit. This is your current\n"
            "  in-progress implementation.\n\n"
            "Structured context:\n"
            f"{context_json}\n"
        )

    def prepare_proposer_launch(
        self,
        output_path: str,
        title: str = "proposer",
        attempt_id: str | None = None,
        tags: list[str] | None = None,
    ) -> ProposerLaunch:
        prompt_text = self.build_proposer_prompt(attempt_id=attempt_id)
        task_id = str(uuid4())
        prompt_path = self.root / "agents" / f"{task_id}.prompt.txt"
        summary_path = self.root / "agents" / f"{task_id}.launch.json"
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_text)

        command = self._build_proposer_command(
            prompt_text=prompt_text,
            prompt_path=prompt_path,
            output_path=output_file,
        )
        task = AgentTaskRecord(
            task_id=task_id,
            session_name=self.config.session_name,
            attempt_id=attempt_id,
            title=title,
            prompt_path=prompt_path.as_posix(),
            output_path=output_file.as_posix(),
            summary_path=summary_path.as_posix(),
            tags=["proposer", *(tags or [])],
        )
        self.state.upsert_agent_task(task)

        summary_payload = {
            "task_id": task.task_id,
            "session_name": task.session_name,
            "attempt_id": task.attempt_id,
            "title": task.title,
            "prompt_path": task.prompt_path,
            "output_path": task.output_path,
            "summary_path": task.summary_path,
            "command": command,
            "tags": task.tags,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

        return ProposerLaunch(
            task=task,
            prompt_path=prompt_path.as_posix(),
            output_path=output_file.as_posix(),
            command=command,
            summary_path=summary_path.as_posix(),
        )

    def run_prepared_proposer(self, launch: ProposerLaunch) -> dict:
        summary_path = Path(launch.summary_path)
        summary_payload = (
            json.loads(summary_path.read_text()) if summary_path.exists() else {}
        )
        summary_payload["started_at"] = utc_now()
        summary_payload["status"] = "running"
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

        completed = subprocess.run(
            ["zsh", "-lc", launch.command],
            cwd=self.git.repo_root,
            check=False,
            text=True,
        )

        summary_payload["completed_at"] = utc_now()
        summary_payload["exit_code"] = completed.returncode
        summary_payload["status"] = "passed" if completed.returncode == 0 else "failed"
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
        return summary_payload

    def run_proposer_once(
        self,
        output_dir: str,
        title: str = "proposer",
        attempt_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid4())[:8]
        timestamp = utc_now()[:19].replace("-", "").replace("T", "-").replace(":", "")
        output_path = output_root / f"{timestamp}-{title}-{run_id}.txt"
        launch = self.prepare_proposer_launch(
            output_path=output_path.as_posix(),
            title=title,
            attempt_id=attempt_id,
            tags=tags,
        )
        result = self.run_prepared_proposer(launch)
        result["task_id"] = launch.task.task_id
        result["prompt_path"] = launch.prompt_path
        result["output_path"] = launch.output_path
        result["summary_path"] = launch.summary_path
        result["command"] = launch.command
        return result

    def run_configured_proposer_once(
        self,
        output_dir: str,
        title: str = "proposer",
        attempt_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        if not self.config.proposer_command_template:
            return self.run_proposer_once(
                output_dir=output_dir,
                title=title,
                attempt_id=attempt_id,
                tags=tags,
            )

        prompt_text = self.build_proposer_prompt(attempt_id=attempt_id)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid4())[:8]
        timestamp = utc_now()[:19].replace("-", "").replace("T", "-").replace(":", "")
        output_path = output_root / f"{timestamp}-{title}-{run_id}.txt"
        prompt_path = self.root / "agents" / f"{run_id}.proposer.prompt.txt"
        summary_path = self.root / "agents" / f"{run_id}.proposer.json"
        prompt_path.write_text(prompt_text)

        command = self._build_proposer_command(
            prompt_text=prompt_text,
            prompt_path=prompt_path,
            output_path=Path(output_path),
        )
        task = AgentTaskRecord(
            task_id=str(uuid4()),
            session_name=self.config.session_name,
            attempt_id=attempt_id,
            title=title,
            prompt_path=prompt_path.as_posix(),
            output_path=output_path.as_posix(),
            summary_path=summary_path.as_posix(),
            tags=["proposer", *(tags or [])],
        )
        self.state.upsert_agent_task(task)

        summary_payload = {
            "task_id": task.task_id,
            "command": command,
            "prompt_path": task.prompt_path,
            "output_path": task.output_path,
            "started_at": utc_now(),
            "status": "running",
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

        # Retry 429 / empty LLM responses in place — the caller must not
        # see a "rate_limited" status; from its point of view the agent
        # simply took longer than usual to respond.
        def _invoke():
            return subprocess.run(
                ["zsh", "-lc", command],
                cwd=self.git.repo_root,
                check=False,
                text=True,
            )

        completed = self._run_agent_with_retry_on_rate_limit(
            runner=_invoke,
            output_path=Path(output_path),
            label="proposer",
        )
        summary_payload["completed_at"] = utc_now()
        summary_payload["exit_code"] = completed.returncode
        summary_payload["status"] = "passed" if completed.returncode == 0 else "failed"
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
        summary_payload["summary_path"] = summary_path.as_posix()
        return summary_payload

    def _build_proposer_command(
        self, prompt_text: str, prompt_path: Path, output_path: Path
    ) -> str:
        if self.config.proposer_command_template:
            return self.config.proposer_command_template.format(
                prompt=shlex.quote(prompt_text),
                prompt_path=shlex.quote(prompt_path.as_posix()),
                output_path=shlex.quote(output_path.as_posix()),
            )
        return (
            f"codebuddy --model 'glm-5.1-ioa' --effort high "
            f"-p {shlex.quote(prompt_text)} "
            f"--dangerously-skip-permissions "
            f"| tee {shlex.quote(output_path.as_posix())}"
        )

    def _build_apply_command(
        self,
        prompt_text: str,
        prompt_path: Path,
        output_path: Path,
        suggestion_path: Path,
    ) -> str:
        if self.config.apply_command_template:
            return self.config.apply_command_template.format(
                prompt=shlex.quote(prompt_text),
                prompt_path=shlex.quote(prompt_path.as_posix()),
                output_path=shlex.quote(output_path.as_posix()),
                suggestion_path=shlex.quote(suggestion_path.as_posix()),
            )
        return (
            f"codebuddy --model 'glm-5.1-ioa' --effort high "
            f"-p {shlex.quote(prompt_text)} "
            f"--dangerously-skip-permissions"
        )

    def run_dual_validation_round(
        self, summary: str, intent: str, base_ref: str = "HEAD~1"
    ) -> dict:
        auto_committed = self._commit_current_worktree_if_needed(
            commit_message="night-optimizer: candidate for dual validation"
        )
        attempt = self.create_attempt(summary=summary, intent=intent, base_ref=base_ref)

        correctness_execution = None
        performance_execution = None
        if self.config.validate_correctness_command:
            correctness_execution = self._run_named_validation_execution(
                attempt=attempt,
                remote_command=self.config.validate_correctness_command,
                suffix="correctness",
            )
        if self.config.validate_performance_command:
            performance_execution = self._run_named_validation_execution(
                attempt=attempt,
                remote_command=self.config.validate_performance_command,
                suffix="performance",
            )

        merged_metrics = AttemptMetrics()
        review_notes: list[str] = []
        raw_artifacts: list[str] = []

        if correctness_execution is not None:
            merged_metrics = self._merge_attempt_metrics(
                merged_metrics,
                self.executor.build_attempt_metrics(correctness_execution),
            )
            review_notes.append(
                f"correctness execution status: {correctness_execution.status}"
            )
            raw_artifacts.extend(
                [
                    correctness_execution.log_path,
                    correctness_execution.summary_path,
                    *correctness_execution.fetched_artifacts,
                ]
            )

        if performance_execution is not None:
            merged_metrics = self._merge_attempt_metrics(
                merged_metrics,
                self.executor.build_attempt_metrics(performance_execution),
            )
            review_notes.append(
                f"performance execution status: {performance_execution.status}"
            )
            raw_artifacts.extend(
                [
                    performance_execution.log_path,
                    performance_execution.summary_path,
                    *performance_execution.fetched_artifacts,
                ]
            )

        attempt.metrics = merged_metrics

        # Compute speedup = baseline_latency / current_latency. Baseline =
        # smallest latency_ms among previously ACCEPTED attempts in this
        # session (previous "best-known-good"). If there is no such history
        # yet, speedup stays None and policy.evaluate() will treat a
        # configured min_speedup as a non-blocking warning for round 1.
        baseline_latency = self._baseline_latency_ms(
            exclude_attempt_id=attempt.attempt_id
        )
        if (
            baseline_latency is not None
            and attempt.metrics.latency_ms is not None
            and attempt.metrics.latency_ms > 0
        ):
            attempt.metrics.speedup = baseline_latency / attempt.metrics.latency_ms

        attempt.evidence = ReviewEvidence(
            correctness_report=(
                correctness_execution.summary_path if correctness_execution else None
            ),
            performance_report=(
                performance_execution.summary_path if performance_execution else None
            ),
            diff_summary=attempt.candidate.diff_summary if attempt.candidate else None,
            review_notes="; ".join(review_notes),
            raw_artifacts=[artifact for artifact in raw_artifacts if artifact],
        )
        attempt.updated_at = utc_now()

        # When a remote execution fails we classify the failure from the
        # remote_test.log so downstream consumers (future proposers, human
        # reviewers) see the ROOT CAUSE — "compile_failed: unsupported
        # expression" — instead of the SYMPTOM "Missing cosine similarity
        # metric". The generic policy evaluate() only looks at metric
        # values; it cannot tell a crashed run apart from a run whose
        # parser is stale.
        execution_failure_reasons: list[str] = []
        execution_failure_categories: list[str] = []
        for label, execution in (
            ("correctness", correctness_execution),
            ("performance", performance_execution),
        ):
            if execution is None or execution.status == "passed":
                continue
            log_text = ""
            try:
                if execution.log_path and Path(execution.log_path).exists():
                    log_text = Path(execution.log_path).read_text(errors="replace")
            except OSError:
                log_text = ""
            category, reason = classify_remote_failure(log_text)
            execution_failure_categories.append(category)
            execution_failure_reasons.append(
                f"{label} {category}: {reason} "
                f"(execution_id={execution.execution_id})"
            )

        if (
            correctness_execution is not None
            and correctness_execution.status != "passed"
        ):
            attempt.status = AttemptStatus.ERROR
            attempt.rejection_reasons = list(execution_failure_reasons) or [
                f"Correctness execution failed for execution_id={correctness_execution.execution_id}"
            ]
        elif (
            performance_execution is not None
            and performance_execution.status != "passed"
        ):
            attempt.status = AttemptStatus.ERROR
            attempt.rejection_reasons = list(execution_failure_reasons) or [
                f"Performance execution failed for execution_id={performance_execution.execution_id}"
            ]
        else:
            attempt.status = AttemptStatus.REVIEW_REQUIRED
            attempt.rejection_reasons = []
        self.state.upsert_attempt(attempt)
        self._write_attempt_snapshot(attempt)

        decision = self.review_attempt(attempt)
        # If the attempt errored out at execution layer, surface the
        # execution-level classification as the PRIMARY rejection reason
        # for the round result, so _auto_record_insight /
        # run_overnight_two_stage_round / fixer prompts see the root cause
        # rather than the policy-layer symptoms.
        if execution_failure_reasons:
            decision = ReviewDecision(
                verdict=ReviewVerdict.FAIL,
                reasons=[*execution_failure_reasons, *decision.reasons],
                warnings=decision.warnings,
            )
            attempt.rejection_reasons = list(decision.reasons)
            self.state.upsert_attempt(attempt)
            self._write_attempt_snapshot(attempt)
        return {
            "attempt_id": attempt.attempt_id,
            "attempt_path": (
                self.root / "attempts" / f"{attempt.attempt_id}.json"
            ).as_posix(),
            "auto_committed": auto_committed,
            "correctness_execution_id": (
                correctness_execution.execution_id if correctness_execution else None
            ),
            "correctness_summary_path": (
                correctness_execution.summary_path if correctness_execution else None
            ),
            "performance_execution_id": (
                performance_execution.execution_id if performance_execution else None
            ),
            "performance_summary_path": (
                performance_execution.summary_path if performance_execution else None
            ),
            "review_verdict": decision.verdict.value,
            "review_reasons": decision.reasons,
            "review_warnings": decision.warnings,
        }

    def run_overnight_two_stage_round(
        self,
        round_index: int,
        output_dir: str,
        summary: str,
        intent: str,
        base_ref: str = "HEAD~1",
    ) -> OvernightRoundResult:
        # Record the worktree starting point so we can fully roll back if the
        # round exhausts its fix budget without passing validation.
        round_start_head = self._current_head()
        fix_outputs: list[str] = []
        validation: dict = {}
        proposer_output_path = ""
        apply_output_path = Path(output_dir) / f"round-{round_index:02d}-apply.txt"
        apply_result: dict = {}
        passed = False
        try:
            # Refresh the on-disk memory handle so proposer/apply prompts can
            # dereference exports/prompt_context.json with up-to-date content.
            self.export_prompt_context()

            proposer_result = self.run_configured_proposer_once(
                output_dir=output_dir,
                title=f"proposer-round-{round_index}",
                tags=[f"round-{round_index}"],
            )
            proposer_output_path = proposer_result["output_path"]
            # NOTE: proposer_result never carries status="rate_limited"
            # because run_configured_proposer_once retries 429/empty
            # responses in place via _run_agent_with_retry_on_rate_limit.
            # The same is true for apply_suggestion below.
            intent_text = Path(proposer_output_path).read_text()

            apply_result = self.apply_suggestion(
                suggestion_path=proposer_output_path,
                output_path=apply_output_path.as_posix(),
                title=f"applier-round-{round_index}",
                tags=[f"round-{round_index}"],
            )

            # If the applier produced no diff (empty proposer output, 401
            # auth error, or the agent genuinely declined), skip validation
            # for this round and let the outer overnight loop move on.
            if apply_result.get("status") in ("no_change", "scope_violation"):
                validation = {
                    "review_verdict": "fail",
                    "review_reasons": [
                        f"apply_suggestion returned status={apply_result.get('status')}: "
                        + str(apply_result.get("reason") or "no diff produced")
                    ],
                }
                # Roll back any agent scratch state back to the round start.
                try:
                    self._reset_worktree_to(round_start_head)
                except Exception:
                    pass
                # Skip the fix loop — without a draft diff there is nothing
                # to fix; the fixer has no signal to act on.
                return OvernightRoundResult(
                    round_index=round_index,
                    proposer_output_path=proposer_output_path,
                    apply_output_path=apply_output_path.as_posix(),
                    apply_summary_path=apply_result.get("summary_path"),
                    correctness_execution_id=None,
                    correctness_summary_path=None,
                    performance_execution_id=None,
                    performance_summary_path=None,
                    review_verdict="fail",
                    review_reasons=validation["review_reasons"],
                    status="failed",
                    fix_iterations=0,
                    fix_output_paths=[],
                )

            max_fix = max(0, int(self.config.max_fix_iterations))
            # Total validations per round = 1 initial + up to max_fix retries.
            for fix_i in range(max_fix + 1):
                validation = self.run_dual_validation_round(
                    summary=summary, intent=intent, base_ref=base_ref
                )
                if validation.get("review_verdict") == "pass":
                    break
                if fix_i == max_fix:
                    break  # out of fix budget

                fix_output_path = (
                    Path(output_dir) / f"round-{round_index:02d}-fix-{fix_i + 1}.txt"
                )
                remote_log_path = self._derive_remote_log_path(validation)
                last_summary_path = validation.get(
                    "correctness_summary_path"
                ) or validation.get("performance_summary_path")
                fix_summary = self.apply_fix(
                    intent_text=intent_text,
                    suggestion_path=proposer_output_path,
                    output_path=fix_output_path.as_posix(),
                    fix_iteration=fix_i + 1,
                    round_start_commit=round_start_head,
                    remote_log_path=remote_log_path,
                    last_execution_summary_path=last_summary_path,
                    title=f"fixer-round-{round_index}-fix-{fix_i + 1}",
                    tags=[f"round-{round_index}", f"fix-{fix_i + 1}"],
                )
                fix_outputs.append(fix_output_path.as_posix())
                if fix_summary.get("status") in ("no_change", "scope_violation"):
                    # Fix agent declined to edit, or its edits violated scope
                    # (e.g. tried to modify the orchestrator). Either way there
                    # is no new draft to re-validate, so abort the fix loop.
                    break

            passed = validation.get("review_verdict") == "pass"
            if passed:
                # Enrich the validation payload with round-level context so
                # the insight can record how many fix iterations were
                # needed to reach pass.
                validation_for_insight = dict(validation)
                validation_for_insight["fix_iterations"] = len(fix_outputs)

                # Run the reflector agent (best-effort) on all materials
                # available for this round. Its output is a structured
                # Markdown summary that gets appended to the insight
                # detail so future proposers can inherit lessons.
                attempt_id_for_reflector = validation.get("attempt_id")
                attempt_payload_for_reflector = (
                    self.state.get_attempt_payload(attempt_id_for_reflector)
                    if attempt_id_for_reflector
                    else None
                )
                current_metrics = (
                    attempt_payload_for_reflector.get("metrics")
                    if attempt_payload_for_reflector
                    else {}
                ) or {}
                baseline_latency = self._baseline_latency_ms(
                    exclude_attempt_id=attempt_id_for_reflector
                )
                baseline_metrics = (
                    {"latency_ms": baseline_latency}
                    if baseline_latency is not None
                    else None
                )
                correctness_log_path = None
                if validation.get("correctness_summary_path"):
                    correctness_log_path = (
                        Path(validation["correctness_summary_path"]).parent
                        / "remote_test.log"
                    ).as_posix()
                performance_log_path = None
                if validation.get("performance_summary_path"):
                    performance_log_path = (
                        Path(validation["performance_summary_path"]).parent
                        / "remote_test.log"
                    ).as_posix()

                reflection_text = self._run_reflector(
                    intent=intent,
                    proposer_output_path=proposer_output_path,
                    apply_output_path=apply_output_path.as_posix(),
                    fix_output_paths=fix_outputs,
                    round_start_commit=round_start_head,
                    correctness_log_path=correctness_log_path,
                    performance_log_path=performance_log_path,
                    metrics=current_metrics,
                    baseline_metrics=baseline_metrics,
                    attempt_id=attempt_id_for_reflector,
                )

                self._auto_record_insight_from_validation(
                    round_index=round_index,
                    validation=validation_for_insight,
                    intent=intent,
                    reflection_text=reflection_text,
                )
            else:
                # Exhausted budget or fixer declined. Drop all commits this
                # round added so the next round starts from a clean slate.
                self._reset_worktree_to(round_start_head)
        except Exception:
            # Best-effort rollback before propagating so subsequent rounds do
            # not inherit a dirty worktree or stray commits.
            try:
                self._reset_worktree_to(round_start_head)
            except Exception:
                pass
            raise
        finally:
            # Refresh prompt_context so the NEXT round's proposer sees the
            # insight / attempt_history this round just produced (or failed
            # to produce). Without this, multi-round overnight runs would
            # replay the same stale context from before round 1.
            try:
                self.export_prompt_context()
            except Exception:
                # Export is best-effort; never mask the round outcome with
                # a serialization error.
                pass

        return OvernightRoundResult(
            round_index=round_index,
            proposer_output_path=proposer_output_path,
            apply_output_path=apply_output_path.as_posix(),
            apply_summary_path=apply_result.get("summary_path"),
            correctness_execution_id=validation.get("correctness_execution_id"),
            correctness_summary_path=validation.get("correctness_summary_path"),
            performance_execution_id=validation.get("performance_execution_id"),
            performance_summary_path=validation.get("performance_summary_path"),
            review_verdict=validation.get("review_verdict"),
            review_reasons=validation.get("review_reasons", []),
            status="passed" if passed else "failed",
            fix_iterations=len(fix_outputs),
            fix_output_paths=fix_outputs,
        )

    @staticmethod
    def _derive_remote_log_path(validation: dict) -> str | None:
        # executor.py places remote_test.log alongside execution_summary.json
        # in the per-execution result directory.
        for key in ("correctness_summary_path", "performance_summary_path"):
            summary_path = validation.get(key)
            if summary_path:
                return (Path(summary_path).parent / "remote_test.log").as_posix()
        return None

    def build_reflector_prompt(
        self,
        intent: str,
        proposer_text: str,
        apply_text: str,
        fix_texts: list[str],
        remote_log_tails: list[tuple[str, str]],  # list of (label, tail)
        final_diff_text: str,
        metrics: dict,
        baseline_metrics: dict | None,
    ) -> str:
        # Truncate each material so the total prompt stays bounded. We
        # deliberately keep proposer_text (the actual optimization idea)
        # and fix_texts (how it was repaired) in full whenever we can —
        # they're the most information-dense inputs for reflection.
        def _clamp(text: str, max_chars: int) -> str:
            if text is None:
                return ""
            if len(text) <= max_chars:
                return text
            head = text[: max_chars // 2]
            tail = text[-max_chars // 2 :]
            return f"{head}\n...[truncated {len(text) - max_chars} chars]...\n{tail}"

        sections = []
        sections.append(f"## Intent\n{intent}\n")
        sections.append(
            "## Proposer Output (full text)\n" f"{_clamp(proposer_text, 12000)}\n"
        )
        sections.append("## Applier Output (short)\n" f"{_clamp(apply_text, 4000)}\n")
        for i, ft in enumerate(fix_texts):
            sections.append(
                f"## Fix Iteration {i + 1} Output\n" f"{_clamp(ft, 4000)}\n"
            )
        for label, tail in remote_log_tails:
            sections.append(f"## Remote Log Tail: {label}\n" f"{_clamp(tail, 6000)}\n")
        sections.append(
            "## Final Diff (round_start_commit..HEAD on allowed_paths)\n"
            f"{_clamp(final_diff_text, 8000)}\n"
        )
        sections.append(
            "## Final Metrics (this round)\n"
            f"{json.dumps(metrics, indent=2, sort_keys=True)}\n"
        )
        if baseline_metrics:
            sections.append(
                "## Baseline Metrics (previous best-known-good accepted attempt)\n"
                f"{json.dumps(baseline_metrics, indent=2, sort_keys=True)}\n"
            )
        else:
            sections.append(
                "## Baseline Metrics\n(none — this is the first accepted "
                "candidate in the session)\n"
            )

        materials = "\n".join(sections)

        return (
            "You are the reflector for a deadline-driven NKI kernel "
            "optimization session. A fresh candidate just passed the "
            "remote correctness + performance validation. Your ONLY job "
            "is to distill a reusable lesson from the full round "
            "(proposal, edits, fix iterations, failing logs, final "
            "diff, metrics) so the NEXT proposer round starts smarter.\n\n"
            "Do NOT propose new optimizations in this step. Do NOT "
            "repeat the proposer text verbatim. Do NOT speculate beyond "
            "what the materials support.\n\n"
            "Output EXACTLY these five Markdown `## ` headings (use the "
            "English headings below verbatim as the section labels — the "
            "orchestrator parses them by literal match). BODY TEXT inside "
            "each section may be in Chinese or English, whatever reads "
            "clearest. Each section: 1-5 concrete, NKI-specific bullets.\n\n"
            "## What Worked\n"
            "  - <core optimization mechanism that produced the passing "
            "diff, stated in one line>\n\n"
            "## Why It Worked\n"
            "  - <NKI / compiler reasoning: which parallelism primitive "
            "was unlocked, which hazard was avoided, etc.>\n\n"
            "## What Failed Along The Way\n"
            "  - <for each failed validation before pass, the root "
            "cause in one line — cite the compiler error or the crash>\n\n"
            "## Caveats / Limits\n"
            "  - <shape / dtype / SBUF pressure / numerical stability "
            "pre-conditions this optimization depends on>\n\n"
            "## Suggested Next Directions\n"
            "  - <3 progressive follow-up ideas a future proposer could "
            "try; cite specific lines / concepts from the final diff>\n\n"
            "Your whole response must be PLAIN TEXT with those exact "
            "five headers, no preamble, no conclusion, no code blocks "
            "outside bullet items.\n\n"
            "---BEGIN ROUND MATERIALS---\n"
            f"{materials}\n"
            "---END ROUND MATERIALS---\n"
        )

    def _run_reflector(
        self,
        intent: str,
        proposer_output_path: str,
        apply_output_path: str,
        fix_output_paths: list[str],
        round_start_commit: str,
        correctness_log_path: str | None,
        performance_log_path: str | None,
        metrics: dict,
        baseline_metrics: dict | None,
        attempt_id: str | None,
    ) -> str | None:
        """Launch the reflector agent. Returns the reflection text, or
        None if anything went wrong. Caller must treat this as
        best-effort — failure here must not block insight recording.
        """

        try:
            proposer_text = (
                Path(proposer_output_path).read_text(errors="replace")
                if proposer_output_path
                else ""
            )
            apply_text = (
                Path(apply_output_path).read_text(errors="replace")
                if apply_output_path
                else ""
            )
            fix_texts = [
                Path(p).read_text(errors="replace") if Path(p).exists() else ""
                for p in fix_output_paths
            ]

            remote_log_tails: list[tuple[str, str]] = []
            for label, log_path in (
                ("correctness", correctness_log_path),
                ("performance", performance_log_path),
            ):
                if not log_path:
                    continue
                tail = self._read_log_tail(log_path, n_lines=200)
                if tail:
                    remote_log_tails.append((label, tail))

            final_diff_text = (
                self._collect_diff_since(
                    commit=round_start_commit,
                    paths=list(self.config.scope.allowed_paths),
                )
                if round_start_commit
                else ""
            )

            prompt_text = self.build_reflector_prompt(
                intent=intent,
                proposer_text=proposer_text,
                apply_text=apply_text,
                fix_texts=fix_texts,
                remote_log_tails=remote_log_tails,
                final_diff_text=final_diff_text,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
            )

            task_id = str(uuid4())
            prompt_path = self.root / "agents" / f"{task_id}.reflector.prompt.txt"
            output_path = self.root / "agents" / f"{task_id}.reflector.output.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt_text)

            task = AgentTaskRecord(
                task_id=task_id,
                session_name=self.config.session_name,
                attempt_id=attempt_id,
                title="reflector",
                prompt_path=prompt_path.as_posix(),
                output_path=output_path.as_posix(),
                summary_path=None,
                tags=["reflector"],
            )
            self.state.upsert_agent_task(task)

            command = self._build_proposer_command(
                prompt_text=prompt_text,
                prompt_path=prompt_path,
                output_path=output_path,
            )
            with output_path.open("w") as handle:
                completed = subprocess.run(
                    ["zsh", "-lc", command],
                    cwd=self.git.repo_root,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode != 0:
                return None
            text = output_path.read_text(errors="replace").strip()
            # Sanity: reject obvious garbage. We intentionally do NOT
            # require the exact English heading strings because the LLM
            # (per project CLAUDE.md) may localize them to Chinese. Accept
            # any output that is reasonably long AND has at least three
            # Markdown "## " section headings, which is the minimum for
            # a useful reflection.
            heading_count = sum(
                1 for line in text.splitlines() if line.lstrip().startswith("## ")
            )
            if len(text) < 200 or heading_count < 3:
                return None
            return text
        except Exception:
            # Reflector is best-effort. Never let its failure surface as
            # a round-level exception.
            return None

    def _auto_record_insight_from_validation(
        self,
        round_index: int,
        validation: dict,
        intent: str,
        reflection_text: str | None = None,
    ) -> InsightRecord | None:
        # Promote a passing overnight round to a VALIDATED insight so that
        # future proposer rounds can see a growing knowledge base instead of
        # an empty validated_insights list.
        if validation.get("review_verdict") != "pass":
            return None

        attempt_id = validation.get("attempt_id")
        attempt_payload = (
            self.state.get_attempt_payload(attempt_id) if attempt_id else None
        )
        title = f"round-{round_index} validated attempt"

        detail_parts = [
            f"intent: {intent}",
            f"attempt_id: {attempt_id}",
            f"correctness_summary: {validation.get('correctness_summary_path')}",
            f"performance_summary: {validation.get('performance_summary_path')}",
        ]

        # Pack hard, machine-readable metrics + diff stat into the detail
        # so downstream automation (next proposer, ranking, plots) can read
        # them without reparsing result logs or attempt JSON files.
        metrics_block: dict[str, float | int | str | None] = {}
        diff_summary: str | None = None
        if attempt_payload is not None:
            m = attempt_payload.get("metrics") or {}
            metrics_block = {
                "cosine_similarity": m.get("cosine_similarity"),
                "abs_error": m.get("abs_error"),
                "rel_error": m.get("rel_error"),
                "latency_ms": m.get("latency_ms"),
                "speedup": m.get("speedup"),
                "successful_runs": m.get("successful_runs"),
            }
            candidate = attempt_payload.get("candidate") or {}
            diff_summary = candidate.get("diff_summary")
        metrics_block["fix_iterations"] = validation.get("fix_iterations")
        metrics_block["round_index"] = round_index

        detail_parts.append("metrics: " + json.dumps(metrics_block, sort_keys=True))
        if diff_summary:
            detail_parts.append(f"diff_stat:\n{diff_summary}")

        warnings = validation.get("review_warnings") or []
        if warnings:
            detail_parts.append("warnings: " + "; ".join(warnings))

        if reflection_text:
            # Append the reflector agent's structured Markdown summary
            # (What Worked / Why / Failed / Caveats / Next Directions)
            # as a clearly delimited section, so future proposers can
            # parse it or display it verbatim.
            detail_parts.append(
                "\n---\n## Reflection Summary\n" + reflection_text.strip()
            )

        return self.record_insight(
            title=title,
            detail="\n".join(detail_parts),
            source_attempt_id=attempt_id,
            status=InsightStatus.VALIDATED,
            confidence=1.0,
        )

    def apply_suggestion(
        self,
        suggestion_path: str,
        output_path: str,
        title: str = "applier",
        attempt_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        self._ensure_clean_worktree()
        pre_agent_head = self._current_head()

        suggestion_text = Path(suggestion_path).read_text()
        prompt_text = self.build_applier_prompt(
            suggestion_text=suggestion_text, attempt_id=attempt_id
        )
        task_id = str(uuid4())
        prompt_path = self.root / "agents" / f"{task_id}.apply.prompt.txt"
        summary_path = self.root / "agents" / f"{task_id}.apply.json"
        raw_output_path = Path(output_path)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_text)

        task = AgentTaskRecord(
            task_id=task_id,
            session_name=self.config.session_name,
            attempt_id=attempt_id,
            title=title,
            prompt_path=prompt_path.as_posix(),
            output_path=raw_output_path.as_posix(),
            summary_path=summary_path.as_posix(),
            tags=["applier", *(tags or [])],
        )
        self.state.upsert_agent_task(task)

        summary_payload = {
            "task_id": task.task_id,
            "session_name": task.session_name,
            "attempt_id": task.attempt_id,
            "title": task.title,
            "prompt_path": task.prompt_path,
            "output_path": task.output_path,
            "summary_path": task.summary_path,
            "started_at": utc_now(),
            "status": "running",
            "tags": task.tags,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

        command = self._build_apply_command(
            prompt_text=prompt_text,
            prompt_path=prompt_path,
            output_path=raw_output_path,
            suggestion_path=Path(suggestion_path),
        )
        summary_payload["command"] = command

        # Let the agent edit files in place. We do NOT parse a unified diff
        # from stdout: the agent is free to use any tool chain as long as the
        # final worktree only touches scope.allowed_paths.
        # Retry 429 / empty LLM responses in place so the overnight loop
        # does not advance its round counter for what is really a transient
        # quota miss. Between retries we reset any partial worktree edits
        # from the previous attempt — otherwise a second invocation would
        # start from a dirty tree and the final scope check would conflate
        # two agent runs.
        def _invoke_applier():
            with raw_output_path.open("w") as handle:
                return subprocess.run(
                    ["zsh", "-lc", command],
                    cwd=self.git.repo_root,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

        def _reset_before_retry():
            # If a 429 came back the agent may still have left partial
            # edits (e.g. it started editing before the provider blocked
            # the next tool call). Snap back to pre_agent_head so the
            # retry operates on the same clean base as the first try.
            try:
                self._soft_reset_agent_commits(pre_agent_head)
            except Exception:
                pass
            try:
                self._reset_worktree()
            except Exception:
                pass

        completed = self._run_agent_with_retry_on_rate_limit(
            runner=_invoke_applier,
            output_path=raw_output_path,
            label="applier",
            reset_between_retries=_reset_before_retry,
        )

        # The applier prompt forbids `git commit`, but we mechanically enforce
        # it: if the agent created commits we soft-reset back to pre_agent_head
        # so their edits become ordinary uncommitted changes and the scope
        # check below applies uniformly.
        agent_committed = self._current_head() != pre_agent_head
        if agent_committed:
            self._soft_reset_agent_commits(pre_agent_head)
        summary_payload["agent_committed"] = agent_committed
        summary_payload["pre_agent_head"] = pre_agent_head

        changed_files = self._list_worktree_changed_files()
        scope_decision = self.scope.validate(changed_files)

        if scope_decision.violations:
            summary_payload["completed_at"] = utc_now()
            summary_payload["exit_code"] = completed.returncode
            summary_payload["status"] = "failed"
            summary_payload["reason"] = "Applied worktree violates scope"
            summary_payload["changed_files"] = changed_files
            summary_payload["violations"] = scope_decision.violations
            summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
            # Roll back so the next round can start from a clean worktree.
            try:
                self._reset_worktree()
            except Exception as reset_error:
                summary_payload["reset_error"] = str(reset_error)
                summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
            raise RuntimeError(
                "Applied worktree violates scope: "
                + "; ".join(scope_decision.violations)
            )

        if not changed_files:
            # Soft-fail: mirror apply_fix's "no_change" path. This can happen
            # when the proposer produced an empty suggestion (e.g. auth
            # failure, 401, transient network error) or when the applier
            # correctly concluded there was no safe edit. Either way we do
            # NOT crash the overnight loop — the caller treats no_change as
            # a round-level rejection and moves on to the next round.
            summary_payload["completed_at"] = utc_now()
            summary_payload["exit_code"] = completed.returncode
            summary_payload["status"] = "no_change"
            summary_payload["reason"] = "Agent produced no file changes"
            summary_payload["changed_files"] = []
            summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
            return summary_payload

        summary_payload["completed_at"] = utc_now()
        summary_payload["exit_code"] = completed.returncode
        summary_payload["status"] = "applied"
        summary_payload["changed_files"] = changed_files
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
        return summary_payload

    def apply_fix(
        self,
        intent_text: str,
        suggestion_path: str,
        output_path: str,
        fix_iteration: int,
        round_start_commit: str,
        remote_log_path: str | None,
        last_execution_summary_path: str | None,
        title: str | None = None,
        attempt_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        # Fix agents run *after* a validation commit. Worktree is clean again
        # because run_dual_validation_round committed the previous draft. We
        # soft-reset any agent commits (same policy as apply_suggestion) and
        # enforce scope on the resulting uncommitted diff.
        self._ensure_clean_worktree()
        pre_agent_head = self._current_head()

        log_tail = (
            self._read_log_tail(remote_log_path, n_lines=200) if remote_log_path else ""
        )
        diff_text = self._collect_diff_since(
            commit=round_start_commit,
            paths=list(self.config.scope.allowed_paths),
        )
        metadata = {
            "round_start_commit": round_start_commit,
            "pre_apply_commit": pre_agent_head,
            "current_head": pre_agent_head,
            "remote_test_log_path": str(remote_log_path) if remote_log_path else None,
            "last_execution_summary_path": last_execution_summary_path,
            "suggestion_path": str(suggestion_path) if suggestion_path else None,
            "allowed_paths": list(self.config.scope.allowed_paths),
        }
        prompt_text = self.build_fixer_prompt(
            intent_text=intent_text,
            metadata=metadata,
            log_tail=log_tail,
            diff_text=diff_text,
            fix_iteration=fix_iteration,
            attempt_id=attempt_id,
        )

        task_id = str(uuid4())
        prompt_path = self.root / "agents" / f"{task_id}.fix.prompt.txt"
        summary_path = self.root / "agents" / f"{task_id}.fix.json"
        raw_output_path = Path(output_path)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_text)

        task_title = title or f"fixer-iter-{fix_iteration}"
        task = AgentTaskRecord(
            task_id=task_id,
            session_name=self.config.session_name,
            attempt_id=attempt_id,
            title=task_title,
            prompt_path=prompt_path.as_posix(),
            output_path=raw_output_path.as_posix(),
            summary_path=summary_path.as_posix(),
            tags=["fixer", f"fix-{fix_iteration}", *(tags or [])],
        )
        self.state.upsert_agent_task(task)

        summary_payload = {
            "task_id": task.task_id,
            "session_name": task.session_name,
            "attempt_id": task.attempt_id,
            "title": task.title,
            "prompt_path": task.prompt_path,
            "output_path": task.output_path,
            "summary_path": task.summary_path,
            "started_at": utc_now(),
            "status": "running",
            "fix_iteration": fix_iteration,
            "round_start_commit": round_start_commit,
            "pre_apply_commit": pre_agent_head,
            "remote_test_log_path": metadata["remote_test_log_path"],
            "tags": task.tags,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

        command = self._build_apply_command(
            prompt_text=prompt_text,
            prompt_path=prompt_path,
            output_path=raw_output_path,
            suggestion_path=(
                Path(suggestion_path) if suggestion_path else Path(prompt_path)
            ),
        )
        summary_payload["command"] = command

        with raw_output_path.open("w") as handle:
            completed = subprocess.run(
                ["zsh", "-lc", command],
                cwd=self.git.repo_root,
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )

        agent_committed = self._current_head() != pre_agent_head
        if agent_committed:
            self._soft_reset_agent_commits(pre_agent_head)
        summary_payload["agent_committed"] = agent_committed

        changed_files = self._list_worktree_changed_files()
        scope_decision = self.scope.validate(changed_files)

        if scope_decision.violations:
            # Treat scope violations as a soft "no_change" so the outer fix
            # loop can exit gracefully instead of crashing the overnight run.
            # The offending edits are discarded; the round will be rolled back
            # to round_start_commit by the caller if no pass is reached.
            try:
                self._reset_worktree_to(pre_agent_head)
            except Exception as reset_error:
                summary_payload["reset_error"] = str(reset_error)
            summary_payload["completed_at"] = utc_now()
            summary_payload["exit_code"] = completed.returncode
            summary_payload["status"] = "scope_violation"
            summary_payload["reason"] = "Fix worktree violates scope"
            summary_payload["changed_files"] = changed_files
            summary_payload["violations"] = scope_decision.violations
            summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
            return summary_payload

        summary_payload["completed_at"] = utc_now()
        summary_payload["exit_code"] = completed.returncode
        summary_payload["status"] = "applied" if changed_files else "no_change"
        summary_payload["changed_files"] = changed_files
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
        return summary_payload

    def validate_current_worktree(
        self,
        summary: str,
        intent: str,
        base_ref: str = "HEAD~1",
        commit_message: str = "night-optimizer: validate candidate",
    ) -> dict:
        changed_files = self._list_worktree_changed_files()
        auto_committed = False

        if changed_files:
            scope_decision = self.scope.validate(changed_files)
            if scope_decision.violations:
                raise RuntimeError(
                    "Current worktree violates scope: "
                    + "; ".join(scope_decision.violations)
                )
            self._run_git(["add", "--all", "--", *changed_files])
            self._run_git(["commit", "-m", commit_message])
            auto_committed = True

        attempt = self.create_attempt(summary=summary, intent=intent, base_ref=base_ref)
        execution = self.run_attempt(attempt)
        decision = self.review_attempt(attempt)
        attempt_path = self.root / "attempts" / f"{attempt.attempt_id}.json"
        return {
            "attempt_id": attempt.attempt_id,
            "attempt_path": attempt_path.as_posix(),
            "auto_committed": auto_committed,
            "head_commit": attempt.candidate.head_commit if attempt.candidate else None,
            "execution_id": execution.execution_id,
            "execution_status": execution.status,
            "execution_summary_path": execution.summary_path,
            "review_verdict": decision.verdict.value,
            "review_reasons": decision.reasons,
            "review_warnings": decision.warnings,
        }

    def run_attempt(
        self, attempt: AttemptRecord, remote_test_command: str | None = None
    ) -> ExecutionRecord:
        attempt.status = AttemptStatus.RUNNING
        attempt.updated_at = utc_now()
        self.state.upsert_attempt(attempt)
        self._write_attempt_snapshot(attempt)

        command = remote_test_command or self.config.validate_correctness_command
        if not command:
            raise RuntimeError(
                "run_attempt requires a remote command; pass remote_test_command "
                "or set validate_correctness_command in session config"
            )

        try:
            result_dir = self._make_execution_result_dir(attempt.attempt_id)
            execution = self.executor.run_attempt(
                attempt, result_dir, remote_test_command=command
            )
        except Exception as error:
            attempt.status = AttemptStatus.ERROR
            attempt.updated_at = utc_now()
            attempt.rejection_reasons = [f"Remote execution raised error: {error}"]
            self.state.upsert_attempt(attempt)
            self._write_attempt_snapshot(attempt)
            raise

        attempt.metrics = self.executor.build_attempt_metrics(execution)
        attempt.evidence = self.executor.build_review_evidence(execution, attempt)
        attempt.updated_at = utc_now()
        if execution.status == "passed":
            attempt.status = AttemptStatus.REVIEW_REQUIRED
            attempt.rejection_reasons = []
        else:
            attempt.status = AttemptStatus.ERROR
            attempt.rejection_reasons = [
                f"Remote execution failed for execution_id={execution.execution_id}"
            ]
        self.state.upsert_attempt(attempt)
        self._write_attempt_snapshot(attempt)
        return execution

    def _validate_attempt_patch(self, attempt: AttemptRecord) -> list[str]:
        if not attempt.candidate:
            return ["Attempt is missing candidate patch metadata"]
        if not attempt.candidate.base_commit or not attempt.candidate.head_commit:
            return ["Attempt is missing base_commit/head_commit patch identity"]

        try:
            snapshot = self.git.inspect_patch(
                base_ref=attempt.candidate.base_commit,
                head_ref=attempt.candidate.head_commit,
            )
        except GitInspectionError as error:
            return [f"Failed to re-inspect stored patch identity: {error}"]

        reasons: list[str] = []
        if (
            attempt.candidate.patch_hash
            and snapshot.patch_hash != attempt.candidate.patch_hash
        ):
            reasons.append(
                "Stored patch hash does not match git diff for the recorded commit pair"
            )
        if snapshot.changed_files != attempt.candidate.changed_files:
            reasons.append(
                "Stored changed_files do not match git diff for the recorded commit pair"
            )
        reasons.extend(self.scope.validate(snapshot.changed_files).violations)
        return reasons

    def _ensure_clean_worktree(self) -> None:
        worktree = self.git.get_worktree_state()
        if not worktree.is_clean:
            raise RuntimeError(
                "Working tree must be clean before apply: "
                + "; ".join(worktree.status_entries)
            )

    def _list_worktree_changed_files(self) -> list[str]:
        # Tracked changes (modified/deleted/renamed relative to HEAD).
        tracked = self._run_git(["diff", "--name-only", "HEAD"]).splitlines()
        # Untracked files not matched by .gitignore.
        untracked = self._run_git(
            ["ls-files", "--others", "--exclude-standard"]
        ).splitlines()

        # Ignored files only when they land under protected_roots. Scope-free
        # areas like tmp/, results/, runtime/ remain invisible.
        ignored: list[str] = []
        protected_roots = [root for root in self.config.scope.protected_roots if root]
        if protected_roots:
            git_args = [
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                *protected_roots,
            ]
            ignored = self._run_git(git_args).splitlines()

        merged: list[str] = []
        for raw in (*tracked, *untracked, *ignored):
            stripped = raw.strip()
            if not stripped:
                continue
            normalized = Path(stripped).as_posix()
            if normalized.startswith("./"):
                normalized = normalized[2:]
            if not normalized or self._is_benign_runtime_artifact(normalized):
                continue
            if normalized not in merged:
                merged.append(normalized)
        return merged

    @staticmethod
    def _is_benign_runtime_artifact(path: str) -> bool:
        # Python / editor byproducts that can appear under protected_roots but
        # are never produced by an optimization agent. Skipping them avoids
        # false-positive scope violations.
        if "/__pycache__/" in path or path.startswith("__pycache__/"):
            return True
        if path.endswith(".pyc") or path.endswith(".pyo"):
            return True
        if path.endswith(".ipynb_checkpoints") or "/.ipynb_checkpoints/" in path:
            return True
        if path.endswith(".DS_Store"):
            return True
        return False

    def _reset_worktree(self) -> None:
        # Drop any tracked-file modifications and remove untracked files
        # (including those we only see via protected_roots). This keeps the
        # worktree usable for the next round after a failed apply.
        self._run_git(["reset", "--hard", "HEAD"])
        protected_roots = [root for root in self.config.scope.protected_roots if root]
        clean_args = ["clean", "-fd"]
        if protected_roots:
            clean_args.append("-x")
            clean_args.append("--")
            clean_args.extend(protected_roots)
        self._run_git(clean_args)

    def _current_head(self) -> str:
        return self._run_git(["rev-parse", "HEAD"])

    def _baseline_latency_ms(
        self, exclude_attempt_id: str | None = None
    ) -> float | None:
        """Minimum latency_ms among historical ACCEPTED attempts, i.e. the
        best-known-good candidate for this session. Falls back to
        `config.fixed_baseline_latency_ms` (the committed latency of the
        starting kernel) when no accepted history exists yet. Returns None
        only when both sources are empty, in which case speedup stays
        None and policy.evaluate() will reject under min_speedup>1.0.
        """

        best: float | None = None
        for payload in self.state.list_attempt_payloads(self.config.session_name):
            if exclude_attempt_id and payload.get("attempt_id") == exclude_attempt_id:
                continue
            if payload.get("status") != "accepted":
                continue
            latency = (payload.get("metrics") or {}).get("latency_ms")
            if latency is None:
                continue
            try:
                latency_f = float(latency)
            except (TypeError, ValueError):
                continue
            if latency_f <= 0:
                continue
            if best is None or latency_f < best:
                best = latency_f
        if best is not None:
            return best
        # No accepted history yet -> use the config-provided baseline so
        # round 1 has a real speedup number to evaluate against.
        fixed = self.config.fixed_baseline_latency_ms
        if fixed is not None and fixed > 0:
            return float(fixed)
        return None

    def _reset_worktree_to(self, commit: str) -> None:
        # Full rollback to a specific commit. Used to abort a round when the
        # fix-loop budget is exhausted, so the next round starts clean.
        self._run_git(["reset", "--hard", commit])
        protected_roots = [root for root in self.config.scope.protected_roots if root]
        clean_args = ["clean", "-fd"]
        if protected_roots:
            clean_args.append("-x")
            clean_args.append("--")
            clean_args.extend(protected_roots)
        self._run_git(clean_args)

    def _soft_reset_agent_commits(self, pre_agent_head: str) -> None:
        # If the agent authored commits during its subprocess (forbidden by
        # the applier/fixer prompts but we enforce it mechanically), soft-reset
        # so their edits become uncommitted changes. Our standard
        # _list_worktree_changed_files + scope.validate path then applies
        # uniformly, regardless of whether the agent obeyed.
        current = self._current_head()
        if current == pre_agent_head:
            return
        self._run_git(["reset", "--soft", pre_agent_head])

    def _collect_diff_since(self, commit: str, paths: list[str] | None = None) -> str:
        args = ["diff", commit]
        if paths:
            args.append("--")
            args.extend(paths)
        # `git diff <commit>` with a single ref compares the worktree against
        # <commit>, so it captures both committed and uncommitted changes.
        return self._run_git(args)

    @staticmethod
    def _read_log_tail(log_path: str | Path, n_lines: int = 200) -> str:
        path = Path(log_path)
        if not path.exists():
            return ""
        # Read then slice. Typical remote_test logs are small enough that
        # streaming is overkill.
        lines = path.read_text(errors="replace").splitlines()
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        return "\n".join(tail)

    def _commit_current_worktree_if_needed(self, commit_message: str) -> bool:
        changed_files = self._list_worktree_changed_files()
        if not changed_files:
            return False

        scope_decision = self.scope.validate(changed_files)
        if scope_decision.violations:
            raise RuntimeError(
                "Current worktree violates scope: "
                + "; ".join(scope_decision.violations)
            )

        self._run_git(["add", "--all", "--", *changed_files])
        self._run_git(["commit", "-m", commit_message])
        return True

    @staticmethod
    def _merge_attempt_metrics(
        base: AttemptMetrics, override: AttemptMetrics
    ) -> AttemptMetrics:
        merged = AttemptMetrics(**asdict(base))
        for field_name in (
            "speedup",
            "latency_ms",
            "cosine_similarity",
            "abs_error",
            "rel_error",
        ):
            value = getattr(override, field_name)
            if value is not None:
                setattr(merged, field_name, value)
        if override.successful_runs:
            merged.successful_runs = override.successful_runs
        return merged

    def _run_git(self, args: list[str]) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=self.git.repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            command = " ".join(["git", *args])
            raise RuntimeError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"Failed to run {command}"
            )
        return completed.stdout.strip()

    def _run_named_validation_execution(
        self, attempt: AttemptRecord, remote_command: str, suffix: str
    ) -> ExecutionRecord:
        result_dir = self._make_execution_result_dir(f"{attempt.attempt_id}-{suffix}")
        return self.executor.run_attempt(
            attempt, result_dir, remote_test_command=remote_command
        )

    def _write_attempt_snapshot(self, attempt: AttemptRecord) -> None:
        attempt_dir = self.root / "attempts"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = attempt_dir / f"{attempt.attempt_id}.json"
        snapshot_path.write_text(json.dumps(asdict(attempt), indent=2) + "\n")

    def _write_patch_snapshot(self, attempt_id: str, diff_text: str) -> Path:
        patch_dir = self.root / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{attempt_id}.patch"
        patch_path.write_text(diff_text)
        return patch_path

    def _make_execution_result_dir(self, attempt_id: str) -> Path:
        raw_timestamp = utc_now()
        safe_timestamp = (
            raw_timestamp[:19].replace("-", "").replace("T", "-").replace(":", "")
        )
        result_dir = (
            self.git.repo_root
            / "results"
            / f"{safe_timestamp}-night-optimizer-{attempt_id[:8]}"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir
