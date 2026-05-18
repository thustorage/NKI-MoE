# Night Optimizer Progress

## Goal

This project is being reduced to a trustworthy execution harness for deadline-driven kernel tuning.
Only three capabilities matter right now:

- real git-backed patch identity
- real remote execution through `remote_test.sh`
- real result collection and metric extraction into `results/`

Everything else is either legacy scaffolding or optional later work.

## Audit Summary

### Implemented And Useful

- `repository.py` adds git inspection for repo root, current branch, `HEAD`, diff file list, diff stat, diff text, and patch hash.
- `workflow.py` now creates attempts from actual git diffs and stores patch snapshots under `night-optimizer/runtime/<session>/patches/`.
- `review_attempt` re-inspects the stored commit pair and rejects hash and file mismatches; it also re-runs scope validation on actual changed files.
- `executor.py` adds `run-attempt`, checks clean worktree and matching commit and branch, invokes `remote_test.sh --push`, writes `results/<timestamp>-night-optimizer-*/remote_test.log`, and stores execution metadata plus a summary JSON.
- `results.py` parses minimal metrics from logs and `benchmark_report.json` when fetched successfully.
- `state.py` now persists execution records in sqlite.
- `cli.py` exposes `check-scope --base-ref`, git-backed `create-attempt`, `run-attempt`, and `review-attempt` can reuse metrics and evidence already written into the attempt file.
- Local verification exists:
  - `python -m py_compile ...`
  - `tmp/night_optimizer_smoke.py`
  - archived at `results/20260418-203449-night-optimizer-local-smoke/`

### Implemented But Still Misleading Or Weak

- `review_attempt` still executes the old keyword-based `ReviewPolicy`. The project has not actually escaped legacy policy scaffolding yet.
- `cli.py` still accepts stale `create-attempt --files` and `--diff-summary` arguments, but they are now ignored. This is misleading and should be removed.
- `check-scope` still has two modes: real git diff mode and legacy declared-file mode. The legacy mode is now lower trust and should probably be retired.
- Correctness parsing is still mostly regex over unstructured logs. `cosine_similarity`, `abs_error`, and `rel_error` do not yet come from a guaranteed machine-readable artifact.
- Artifact fetching assumes `scp` from a fixed remote host, fixed remote project path, and fixed artifact filename list. This is workable, but fragile.
- `run-attempt` is synchronous. For long `main.py` or large remote runs this does not yet satisfy the repo guidance to use background task workflow.
- `execution.status` becomes `failed` when artifact fetch fails even if the remote command itself passed. That is conservative, but the summary should distinguish execution failure from post-run collection failure more clearly.
- The example session config still hardcodes branch and remote path choices that must be aligned with the actual repo workflow before the first real run.
- `night-optimizer/runtime/` is local state, not committed source. It is now ignored, which is correct, but this should remain deliberate.

### Still Present But Not On The Critical Path

- `policy.py`
- insight and agent task memory
- prompt-context export
- historian-like persistence structure

These do not block the three core goals, but they still add mental overhead. They should not receive new investment until the execution path is fully trusted.

## Biggest Current Risks

1. Configuration drift between repo truth, `remote_test.sh`, and session config.
   Current repo instructions mention remote `dev` and `/home/ubuntu/code/nki-moe`, while the current `remote_test.sh` and example config are wired for `dev-gsw` and `/home/ubuntu/code/nki-moe-gsw`. This must be unified before trusting any run.
2. Correctness evidence is not machine-readable enough yet.
   Performance can come from `benchmark_report.json`, but correctness still depends on log scraping unless the command writes a structured file.
3. No real remote validation has been executed for this implementation yet.
   The current state has local smoke coverage only. There is no proof yet that `run-attempt` survives the actual remote environment, `scp`, or artifact paths.
4. Review still mixes trustable checks with legacy text heuristics.
   Patch identity and scope are concrete; refusal and suspicious keyword checks are not.
5. The commit model is strict but still incomplete.
   `create-attempt` and `run-attempt` require a clean worktree, which protects provenance, but there is no helper yet for selecting the correct baseline commit or merge-base automatically.

## Immediate Next Work

### Priority 1: Make The First Real Remote Run Succeed

- Align `remote_test.sh`, repo instructions, and session config on one branch and one remote project path.
- Commit the current implementation cleanly.
- Create a safe dry-run attempt against a tiny remote command.
- Confirm that `remote_test.log`, `execution_summary.json`, and fetched artifacts land in `results/`.

### Priority 2: Remove Misleading Legacy Interfaces

- Delete unused `create-attempt --files` and `--diff-summary` inputs.
- Consider deleting the legacy `check-scope` declared-file mode.
- Mark the keyword-based policy layer as secondary, or bypass it when concrete execution evidence already exists.

### Priority 3: Make Correctness Evidence Real

- Require the validation command to write a structured correctness artifact.
- Parse cosine similarity, absolute error, and relative error from that artifact, not from stdout regex.
- Report threshold values explicitly whenever they are changed or enforced:
  - cosine similarity threshold: usually `>= 0.99`
  - absolute error threshold: usually `<= 1e-4`
  - relative error threshold: currently optional or unset in the example config

### Priority 4: Improve Execution Robustness

- Distinguish remote command failure from artifact collection failure in stored status.
- Add better artifact path mapping so results can be fetched from remote `results/` or other explicit output directories.
- Add background execution support for long `main.py`-based runs, consistent with repo expectations.
- Add an execution listing and reload path in sqlite so failed runs can be inspected without manually opening JSON files.

## Concrete Definition Of Done For The Next Checkpoint

The next checkpoint should not add new optimizer ideas. It should prove one boring but trustworthy loop:

1. commit candidate patch
2. create attempt from real git diff
3. run remote validation through `remote_test.sh --push`
4. fetch machine-readable artifacts
5. parse metrics into the attempt
6. review using real patch identity plus real metrics
7. archive everything under `results/`

If this loop works once on the real remote environment, the project becomes useful. If it does not, more policy or memory work is wasted effort.

## Verification Status

- Local syntax check passed for the `night-optimizer` package.
- Local smoke check passed via `tmp/night_optimizer_smoke.py`.
- No real remote test has been executed yet.
