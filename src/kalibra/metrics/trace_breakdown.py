"""Trace breakdown metric — per-task regression/improvement detection.

Computation:
    Group traces by task ID (from metadata field or trace_id heuristic).
    For each task, count successes and total traces with outcome.
    Compare per-task success rates between baseline and current.
    A task "regressed" if its success rate dropped, "improved" if it rose.

Statistical approach:
    Per-task comparison — no global statistical test.
    Direction: INCONCLUSIVE if both regressions and improvements exist,
    DEGRADATION if only regressions, UPGRADE if only improvements.

Threshold fields:
    regressions: number of tasks that regressed
    improvements: number of tasks that improved
"""

from __future__ import annotations

from kalibra.metrics import ComparisonMetric, Direction, Observation
from kalibra.model import OUTCOME_SUCCESS, Trace


class TraceBreakdownMetric(ComparisonMetric):
    name = "trace_breakdown"
    description = "Per-task regression and improvement detection"
    noise_threshold = 0.0
    higher_is_better = True
    _fields = {
        "regressions": "Number of tasks that regressed",
        "improvements": "Number of tasks that improved",
    }

    # Set by the engine from config before compare() is called.
    task_id_field: str | None = None

    def compare(
        self,
        baseline: list[Trace],
        current: list[Trace],
    ) -> Observation:
        b_tasks = self._group_by_task(baseline)
        c_tasks = self._group_by_task(current)

        all_task_ids = set(b_tasks) | set(c_tasks)
        if not all_task_ids:
            return self._no_data(
                "no task data",
                "No task data found",
            )

        regressions: list[dict] = []
        improvements: list[dict] = []
        unchanged: list[str] = []

        for tid in sorted(all_task_ids):
            b = b_tasks.get(tid, {"success": 0, "total": 0})
            c = c_tasks.get(tid, {"success": 0, "total": 0})

            b_rate = b["success"] / b["total"] if b["total"] else None
            c_rate = c["success"] / c["total"] if c["total"] else None

            if b_rate is None or c_rate is None:
                unchanged.append(tid)
                continue

            if c_rate < b_rate:
                regressions.append({
                    "task_id": tid,
                    "baseline": b,
                    "current": c,
                })
            elif c_rate > b_rate:
                improvements.append({
                    "task_id": tid,
                    "baseline": b,
                    "current": c,
                })
            else:
                unchanged.append(tid)

        n_reg = len(regressions)
        n_imp = len(improvements)

        if n_reg > 0 and n_imp > 0:
            direction = Direction.INCONCLUSIVE
        elif n_reg > 0:
            direction = Direction.DEGRADATION
        elif n_imp > 0:
            direction = Direction.UPGRADE
        else:
            direction = Direction.SAME

        return Observation(
            name=self.name,
            description=self.description,
            direction=direction,
            delta=None,
            baseline={"tasks": len(b_tasks)},
            current={"tasks": len(c_tasks)},
            metadata={
                "regressions": regressions,
                "improvements": improvements,
                "n_regressions": n_reg,
                "n_improvements": n_imp,
                "n_unchanged": len(unchanged),
            },
        )

    def threshold_fields(self, result: Observation) -> dict[str, float]:
        return {
            "regressions": float(result.metadata.get("n_regressions", 0)),
            "improvements": float(result.metadata.get("n_improvements", 0)),
        }

    def _group_by_task(self, traces: list[Trace]) -> dict[str, dict]:
        """Group traces into {task_id: {success: N, total: N}}."""
        groups: dict[str, dict] = {}
        for t in traces:
            if t.outcome is None:
                continue
            tid = _extract_task_id(t, self.task_id_field)
            if tid not in groups:
                groups[tid] = {"success": 0, "total": 0}
            groups[tid]["total"] += 1
            if t.outcome == OUTCOME_SUCCESS:
                groups[tid]["success"] += 1
        return groups


def _extract_task_id(trace: Trace, task_id_field: str | None) -> str:
    if task_id_field:
        val = trace.metadata.get(task_id_field)
        if val:
            return str(val)
    # Fallback: strip __model__index suffix
    parts = trace.trace_id.split("__")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "__".join(parts[:-2])
    return trace.trace_id
