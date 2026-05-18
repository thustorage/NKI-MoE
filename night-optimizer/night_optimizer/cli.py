from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import load_session_config
from .models import AttemptMetrics, InsightStatus, ReviewEvidence
from .workflow import SessionController


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="night-optimizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--config", required=True)

    scope_parser = subparsers.add_parser("check-scope")
    scope_parser.add_argument("--config", required=True)
    scope_parser.add_argument("--base-ref")
    scope_parser.add_argument("files", nargs="*")

    create_parser = subparsers.add_parser("create-attempt")
    create_parser.add_argument("--config", required=True)
    create_parser.add_argument("--summary", required=True)
    create_parser.add_argument("--intent", required=True)
    create_parser.add_argument("--base-ref", default="HEAD~1")
    create_parser.add_argument("--diff-summary")
    create_parser.add_argument("--files", nargs="*")

    review_parser = subparsers.add_parser("review-attempt")
    review_parser.add_argument("--config", required=True)
    review_parser.add_argument("--attempt-file", required=True)
    review_parser.add_argument("--metrics-json")
    review_parser.add_argument("--evidence-json")

    run_parser = subparsers.add_parser("run-attempt")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--attempt-file", required=True)

    insight_parser = subparsers.add_parser("record-insight")
    insight_parser.add_argument("--config", required=True)
    insight_parser.add_argument("--title", required=True)
    insight_parser.add_argument("--detail", required=True)
    insight_parser.add_argument("--source-attempt-id")
    insight_parser.add_argument(
        "--status",
        default=InsightStatus.PROPOSED.value,
        choices=[status.value for status in InsightStatus],
    )
    insight_parser.add_argument("--confidence", type=float, default=0.0)

    agent_parser = subparsers.add_parser("record-agent-task")
    agent_parser.add_argument("--config", required=True)
    agent_parser.add_argument("--title", required=True)
    agent_parser.add_argument("--prompt-path", required=True)
    agent_parser.add_argument("--output-path", required=True)
    agent_parser.add_argument("--attempt-id")
    agent_parser.add_argument("--summary-path")
    agent_parser.add_argument("--tags", nargs="*")

    context_parser = subparsers.add_parser("export-context")
    context_parser.add_argument("--config", required=True)

    proposer_parser = subparsers.add_parser("prepare-proposer")
    proposer_parser.add_argument("--config", required=True)
    proposer_parser.add_argument("--output-path", required=True)
    proposer_parser.add_argument("--title", default="proposer")
    proposer_parser.add_argument("--attempt-id")
    proposer_parser.add_argument("--tags", nargs="*")

    overnight_parser = subparsers.add_parser("overnight")
    overnight_parser.add_argument("--config", required=True)
    overnight_parser.add_argument("--output-dir", required=True)
    overnight_parser.add_argument("--title", default="proposer")
    overnight_parser.add_argument("--attempt-id")
    overnight_parser.add_argument("--tags", nargs="*")
    overnight_parser.add_argument("--interval-seconds", type=int, default=30)
    overnight_parser.add_argument("--max-runs", type=int, default=0)
    overnight_parser.add_argument(
        "--summary", default="night-optimizer overnight round"
    )
    overnight_parser.add_argument(
        "--intent", default="propose apply and validate the next optimization attempt"
    )
    overnight_parser.add_argument("--base-ref", default="HEAD~1")

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--config", required=True)
    apply_parser.add_argument("--suggestion-path", required=True)
    apply_parser.add_argument("--output-path", required=True)
    apply_parser.add_argument("--title", default="applier")
    apply_parser.add_argument("--attempt-id")
    apply_parser.add_argument("--tags", nargs="*")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--summary", required=True)
    validate_parser.add_argument("--intent", required=True)
    validate_parser.add_argument("--base-ref", default="HEAD~1")
    validate_parser.add_argument(
        "--commit-message", default="night-optimizer: validate candidate"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_session_config(args.config)
    controller = SessionController(config)

    if args.command == "init":
        root = controller.bootstrap()
        print(root)
        return 0

    if args.command == "check-scope":
        if args.base_ref:
            snapshot, violations = controller.inspect_scope(base_ref=args.base_ref)
            payload = {
                "allowed": not violations,
                "violations": violations,
                "base_commit": snapshot.base_commit,
                "head_commit": snapshot.head_commit,
                "patch_hash": snapshot.patch_hash,
                "changed_files": snapshot.changed_files,
                "diff_summary": snapshot.diff_stat,
            }
            print(json.dumps(payload, indent=2))
            return 0 if not violations else 2

        if not args.files:
            parser.error("check-scope requires --base-ref or at least one file path")

        decision = controller.scope.validate(args.files)
        print(
            json.dumps(
                {"allowed": decision.allowed, "violations": decision.violations},
                indent=2,
            )
        )
        return 0 if decision.allowed else 2

    if args.command == "create-attempt":
        controller.bootstrap()
        attempt = controller.create_attempt(
            summary=args.summary,
            intent=args.intent,
            base_ref=args.base_ref,
        )
        print(controller.root / "attempts" / f"{attempt.attempt_id}.json")
        return 0 if not attempt.rejection_reasons else 2

    if args.command == "review-attempt":
        controller.bootstrap()
        attempt = controller.load_attempt(args.attempt_file)
        if args.metrics_json:
            attempt.metrics = AttemptMetrics(
                **json.loads(Path(args.metrics_json).read_text())
            )
        if args.evidence_json:
            attempt.evidence = ReviewEvidence(
                **json.loads(Path(args.evidence_json).read_text())
            )
        decision = controller.review_attempt(attempt)
        print(
            json.dumps(
                {
                    "verdict": decision.verdict.value,
                    "reasons": decision.reasons,
                    "warnings": decision.warnings,
                },
                indent=2,
            )
        )
        return 0 if decision.verdict.value == "pass" else 2

    if args.command == "run-attempt":
        controller.bootstrap()
        attempt = controller.load_attempt(args.attempt_file)
        execution = controller.run_attempt(attempt)
        print(
            json.dumps(
                {
                    "execution_id": execution.execution_id,
                    "status": execution.status,
                    "result_dir": execution.result_dir,
                    "log_path": execution.log_path,
                    "summary_path": execution.summary_path,
                    "remote_exit_code": execution.remote_exit_code,
                    "fetched_artifacts": execution.fetched_artifacts,
                },
                indent=2,
            )
        )
        return 0 if execution.status == "passed" else 2

    if args.command == "record-insight":
        controller.bootstrap()
        insight = controller.record_insight(
            title=args.title,
            detail=args.detail,
            source_attempt_id=args.source_attempt_id,
            status=InsightStatus(args.status),
            confidence=args.confidence,
        )
        print(insight.insight_id)
        return 0

    if args.command == "record-agent-task":
        controller.bootstrap()
        task = controller.record_agent_task(
            title=args.title,
            prompt_path=args.prompt_path,
            output_path=args.output_path,
            attempt_id=args.attempt_id,
            summary_path=args.summary_path,
            tags=args.tags,
        )
        print(task.task_id)
        return 0

    if args.command == "export-context":
        controller.bootstrap()
        path = controller.export_prompt_context()
        print(path)
        return 0

    if args.command == "prepare-proposer":
        controller.bootstrap()
        launch = controller.prepare_proposer_launch(
            output_path=args.output_path,
            title=args.title,
            attempt_id=args.attempt_id,
            tags=args.tags,
        )
        print(
            json.dumps(
                {
                    "task_id": launch.task.task_id,
                    "prompt_path": launch.prompt_path,
                    "output_path": launch.output_path,
                    "summary_path": launch.summary_path,
                    "command": launch.command,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "overnight":
        controller.bootstrap()
        run_count = 0
        # Rate-limit retries happen at the subprocess layer inside
        # run_configured_proposer_once / apply_suggestion — they sleep and
        # re-invoke the same codebuddy command without advancing the round
        # counter. So this loop only sees "passed" / "failed" / "crashed"
        # statuses, and the inter-round interval is just the configured
        # pacing delay.
        controller.write_progress(state="starting", round_index=0)
        while True:
            if args.max_runs and run_count >= args.max_runs:
                break

            round_index = run_count + 1
            controller.write_progress(
                state="round_running",
                round_index=round_index,
            )
            round_status: str | None = None
            try:
                result = controller.run_overnight_two_stage_round(
                    round_index=round_index,
                    output_dir=args.output_dir,
                    summary=args.summary,
                    intent=args.intent,
                    base_ref=args.base_ref,
                )
                round_status = result.status
                print(json.dumps(result.__dict__, indent=2))
            except Exception as exc:  # noqa: BLE001
                # Never let a single round kill the overnight loop. Any
                # unhandled exception (LLM auth failure, remote_test.sh
                # crash, parser bug) is logged and the loop proceeds to
                # the next round. Fatal errors still surface via the log.
                import traceback

                round_status = "crashed"
                print(
                    json.dumps(
                        {
                            "round_index": round_index,
                            "status": "crashed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                        indent=2,
                    )
                )
            run_count += 1

            controller.write_progress(
                state="round_done",
                round_index=round_index,
                reason=round_status,
            )

            if args.max_runs and run_count >= args.max_runs:
                break

            sleep_seconds = max(args.interval_seconds, 0)
            controller.write_progress(
                state="sleeping",
                round_index=round_index,
                reason=f"interval sleep={sleep_seconds}s",
                extra={"sleep_seconds": sleep_seconds},
            )
            time.sleep(sleep_seconds)
        controller.write_progress(state="stopped", round_index=run_count)
        return 0

    if args.command == "apply":
        controller.bootstrap()
        result = controller.apply_suggestion(
            suggestion_path=args.suggestion_path,
            output_path=args.output_path,
            title=args.title,
            attempt_id=args.attempt_id,
            tags=args.tags,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "validate":
        controller.bootstrap()
        result = controller.validate_current_worktree(
            summary=args.summary,
            intent=args.intent,
            base_ref=args.base_ref,
            commit_message=args.commit_message,
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
