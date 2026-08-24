"""Result returned from an ephemeral forked branch — never the child state."""

from __future__ import annotations

from dataclasses import dataclass, field

from deerflow.subagents.status_contract import SubagentStatusValue, SubagentStopReasonValue

ForkStatus = SubagentStatusValue


@dataclass
class ForkResult:
    """Parent-visible outcome of one forked branch.

    The branch's internal AIMessage/ToolMessage loop is discarded. Only this
    compact payload is merged back into the lead turn.
    """

    task_id: str
    status: ForkStatus
    result: str | None = None
    error: str | None = None
    stop_reason: SubagentStopReasonValue | None = None
    artifacts: list[str] = field(default_factory=list)
    tool_receipts: list[str] = field(default_factory=list)
    token_usage: dict[str, int] | None = None
    token_usage_records: list[dict] = field(default_factory=list)
    model_name: str | None = None
    # A fork borrows (or lazily acquires) the parent thread's sandbox. The
    # parent Command must retain that id so the normal lead lifecycle remains
    # the sole owner that eventually releases it.
    sandbox: dict[str, object] | None = None

    def as_display_text(self) -> str:
        if self.status == "completed":
            body = self.result or ""
            suffix = f" (capped: {self.stop_reason})" if self.stop_reason else ""
            return f"Fork Succeeded{suffix}. Result: {body}".rstrip()
        detail = self.error or self.result or self.status
        if self.stop_reason:
            return f"Fork failed (capped: {self.stop_reason}): {detail}"
        return f"Fork {self.status}: {detail}"
