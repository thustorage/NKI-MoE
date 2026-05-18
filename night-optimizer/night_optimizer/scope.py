from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .models import FileScope


@dataclass
class ScopeDecision:
    allowed: bool
    violations: list[str] = field(default_factory=list)


class ScopeValidator:
    def __init__(self, scope: FileScope) -> None:
        self.scope = scope

    def validate(self, changed_files: list[str]) -> ScopeDecision:
        violations: list[str] = []
        for raw_path in changed_files:
            path = self._normalize(raw_path)
            if self._matches_any(path, self.scope.blocked_paths):
                violations.append(f"Blocked path modified: {path}")
                continue
            if not self._matches_any(path, self.scope.allowed_paths):
                violations.append(f"Out-of-scope path modified: {path}")
        return ScopeDecision(allowed=not violations, violations=violations)

    def is_under_protected_root(self, path: str) -> bool:
        normalized = self._normalize(path)
        return self._matches_any(normalized, self.scope.protected_roots)

    @staticmethod
    def _normalize(path: str) -> str:
        normalized = Path(path).as_posix()
        # Strip a leading "./" but preserve dotfiles like ".gitignore".
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    @staticmethod
    def _matches_any(path: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            normalized_pattern = Path(pattern).as_posix()
            if normalized_pattern.startswith("./"):
                normalized_pattern = normalized_pattern[2:]
            if not normalized_pattern:
                continue
            if path == normalized_pattern or path.startswith(
                normalized_pattern.rstrip("/") + "/"
            ):
                return True
        return False
