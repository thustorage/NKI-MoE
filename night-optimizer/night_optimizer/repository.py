from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitInspectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorktreeState:
    branch: str
    head_commit: str
    status_entries: list[str]

    @property
    def is_clean(self) -> bool:
        return not self.status_entries


@dataclass(frozen=True)
class GitPatchSnapshot:
    repo_root: str
    branch: str
    base_ref: str
    base_commit: str
    head_commit: str
    changed_files: list[str]
    diff_stat: str
    diff_text: str
    patch_hash: str


class GitInspector:
    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else self._discover_root()

    @classmethod
    def discover(cls) -> "GitInspector":
        return cls()

    def current_branch(self) -> str:
        return self._run_git(["branch", "--show-current"])

    def head_commit(self) -> str:
        return self.rev_parse("HEAD")

    def rev_parse(self, ref: str) -> str:
        return self._run_git(["rev-parse", ref])

    def get_worktree_state(self) -> WorktreeState:
        status_output = self._run_git(["status", "--short", "--untracked-files=all"])
        entries = [line for line in status_output.splitlines() if line.strip()]
        return WorktreeState(
            branch=self.current_branch(),
            head_commit=self.head_commit(),
            status_entries=entries,
        )

    def inspect_patch(self, base_ref: str, head_ref: str = "HEAD") -> GitPatchSnapshot:
        base_commit = self.rev_parse(base_ref)
        head_commit = self.rev_parse(head_ref)
        diff_range = f"{base_commit}..{head_commit}"
        changed_output = self._run_git(
            ["diff", "--name-only", "--find-renames", diff_range]
        )
        changed_files = [line for line in changed_output.splitlines() if line.strip()]
        diff_stat = self._run_git(["diff", "--stat", "--find-renames", diff_range])
        diff_text = self._run_git(
            ["diff", "--find-renames", diff_range], strip_output=False
        )
        patch_hash = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
        return GitPatchSnapshot(
            repo_root=self.repo_root.as_posix(),
            branch=self.current_branch(),
            base_ref=base_ref,
            base_commit=base_commit,
            head_commit=head_commit,
            changed_files=changed_files,
            diff_stat=diff_stat,
            diff_text=diff_text,
            patch_hash=patch_hash,
        )

    def _discover_root(self) -> Path:
        return Path(self._run_git(["rev-parse", "--show-toplevel"]))

    def _run_git(self, args: list[str], strip_output: bool = True) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root if hasattr(self, "repo_root") else None,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            command = " ".join(["git", *args])
            raise GitInspectionError(
                result.stderr.strip() or f"Failed to run: {command}"
            )
        if strip_output:
            return result.stdout.strip()
        return result.stdout
