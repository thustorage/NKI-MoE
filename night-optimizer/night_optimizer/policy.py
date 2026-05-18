from __future__ import annotations

from .models import AttemptRecord, ReviewDecision, ReviewVerdict, SessionConfig


class ReviewPolicy:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config

    def evaluate(self, attempt: AttemptRecord) -> ReviewDecision:
        reasons: list[str] = []
        warnings: list[str] = []

        metrics = attempt.metrics
        thresholds = self.config.thresholds

        if (
            thresholds.min_successful_runs
            and metrics.successful_runs < thresholds.min_successful_runs
        ):
            reasons.append(
                f"Insufficient successful runs: {metrics.successful_runs} < {thresholds.min_successful_runs}"
            )

        if thresholds.min_cosine_similarity is not None:
            if metrics.cosine_similarity is None:
                reasons.append("Missing cosine similarity metric")
            elif metrics.cosine_similarity < thresholds.min_cosine_similarity:
                reasons.append(
                    "Cosine similarity below threshold: "
                    f"{metrics.cosine_similarity} < {thresholds.min_cosine_similarity}"
                )

        if thresholds.max_abs_error is not None:
            if metrics.abs_error is None:
                reasons.append("Missing absolute error metric")
            elif metrics.abs_error > thresholds.max_abs_error:
                reasons.append(
                    f"Absolute error above threshold: {metrics.abs_error} > {thresholds.max_abs_error}"
                )

        if thresholds.max_rel_error is not None:
            if metrics.rel_error is None:
                reasons.append("Missing relative error metric")
            elif metrics.rel_error > thresholds.max_rel_error:
                reasons.append(
                    f"Relative error above threshold: {metrics.rel_error} > {thresholds.max_rel_error}"
                )

        if thresholds.min_speedup is not None:
            if metrics.speedup is None:
                # Why: a missing speedup was previously treated as a warning,
                # which let a candidate like 517a27c get accepted without any
                # proof of improvement. When the threshold demands a real
                # speedup (> 1.0) we reject; when it only forbids regression
                # (<= 1.0) we still accept so that the very first baseline
                # can land.
                if thresholds.min_speedup > 1.0:
                    reasons.append(
                        "Missing speedup metric while min_speedup>"
                        f"{thresholds.min_speedup:g} requires a real improvement"
                    )
                else:
                    warnings.append(
                        "Missing speedup metric (no baseline or perf log not parsed); "
                        "min_speedup threshold skipped"
                    )
            elif metrics.speedup < thresholds.min_speedup:
                reasons.append(
                    f"Speedup below threshold: {metrics.speedup} < {thresholds.min_speedup}"
                )

        if reasons:
            return ReviewDecision(
                verdict=ReviewVerdict.FAIL, reasons=reasons, warnings=warnings
            )
        return ReviewDecision(verdict=ReviewVerdict.PASS, reasons=[], warnings=warnings)
