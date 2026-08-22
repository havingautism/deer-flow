"""Runtime helpers for detecting an active forked branch."""

from typing import Any


def is_fork_runtime(runtime: Any | None) -> bool:
    """Return whether this tool/middleware invocation is inside a forked branch."""
    if runtime is None:
        return False
    context = getattr(runtime, "context", None)
    return isinstance(context, dict) and context.get("is_fork") is True
