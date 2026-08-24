"""Fork the lead agent's current state into an ephemeral parallel branch."""

from __future__ import annotations

import logging
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command

from deerflow.forks.executor import DEFAULT_FORK_MAX_TURNS, ForkExecutor
from deerflow.subagents.status_contract import make_subagent_additional_kwargs
from deerflow.tools.types import Runtime
from deerflow.utils.custom_events import aemit_custom_event

logger = logging.getLogger(__name__)


def _parent_context(runtime: Runtime | None) -> dict:
    context = getattr(runtime, "context", None) if runtime is not None else None
    return dict(context) if isinstance(context, dict) else {}


def _parent_config(runtime: Runtime | None) -> dict:
    config = getattr(runtime, "config", None) if runtime is not None else None
    return dict(config) if isinstance(config, dict) else {}


def _parent_state(runtime: Runtime | None):
    if runtime is None:
        return {}
    return getattr(runtime, "state", None) or {}


def _fork_result_command(*, tool_call_id: str, fork_result) -> Command:
    update: dict[str, object] = {
        "messages": [
            ToolMessage(
                content=fork_result.as_display_text(),
                tool_call_id=tool_call_id,
                name="fork_task",
                additional_kwargs=make_subagent_additional_kwargs(
                    fork_result.status,
                    result=fork_result.result,
                    error=fork_result.error,
                    stop_reason=fork_result.stop_reason,
                    model_name=fork_result.model_name,
                    token_usage=fork_result.token_usage,
                ),
            )
        ]
    }
    if fork_result.sandbox is not None:
        update["sandbox"] = fork_result.sandbox
    return Command(update=update)


@tool("fork_task", parse_docstring=True)
async def fork_task_tool(
    runtime: Runtime,
    prompt: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Fork the current lead-agent state into a parallel related branch.

    This is NOT the `task` subagent. `task` starts a fresh, context-isolated
    agent for independent work. `fork_task` clones THIS conversation
    (messages, tools, files, sandbox) and appends only your prompt as a
    suffix so related work stays continuous and prefix/KV cache can be reused.
    Sibling `fork_task` calls in the same response run concurrently.

    Use fork_task when the work is related to the current state AND you can
    launch two or more sibling forks in this same response:
    - Compare approaches, inspect two files, or try A/B against the same context
    - Each branch can finish without the others' results

    Do NOT use fork_task when:
    - The job is independent of this conversation — use `task`
    - One sequential step on the lead agent is enough
    - You would only launch a single fork (stay on the lead)
    - Nested agents or sibling-dependent steps are required

    Forks share the parent workspace and may read and write files. Prefer
    different output paths when sibling forks write in parallel.

    Args:
        prompt: Branch-specific instruction, appended AFTER the inherited
            parent context so sibling forks share the same prefix.
            ALWAYS PROVIDE THIS PARAMETER FIRST.
    """
    writer = get_stream_writer()
    await aemit_custom_event(
        {
            "type": "task_started",
            "task_id": tool_call_id,
            "description": prompt.strip()[:200] or "fork",
        },
        writer=writer,
    )

    parent_context = _parent_context(runtime)
    if parent_context.get("is_fork") is True:
        from deerflow.forks.result import ForkResult

        result = ForkResult(
            task_id=tool_call_id,
            status="failed",
            error="Nested fork_task is not allowed. Complete the current branch directly.",
        )
        await aemit_custom_event({"type": "task_failed", "task_id": tool_call_id, "error": result.error}, writer=writer)
        return _fork_result_command(tool_call_id=tool_call_id, fork_result=result)

    executor = ForkExecutor(
        max_turns=DEFAULT_FORK_MAX_TURNS,
        parent_context=parent_context,
        parent_config=_parent_config(runtime),
    )
    fork_result = await executor.aexecute(
        prompt=prompt,
        parent_state=_parent_state(runtime),
        task_id=tool_call_id,
    )

    event_type = "task_completed" if fork_result.status == "completed" else "task_failed"
    await aemit_custom_event(
        {
            "type": event_type,
            "task_id": tool_call_id,
            "result": fork_result.result,
            "error": fork_result.error,
        },
        writer=writer,
    )
    return _fork_result_command(tool_call_id=tool_call_id, fork_result=fork_result)
