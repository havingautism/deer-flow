"""Snapshot a lead ThreadState into an ephemeral forked branch.

Clone is classified, not a blind deepcopy:

- **Inherited** (shared identity, prefix-cache sensitive): ``messages`` objects,
  ``sandbox``, ``thread_data``, ``summary_text``, ``skill_context`` entries.
- **Branch-local snapshot**: ``todos``, ``artifacts``, ``viewed_images``,
  ``promoted``, ``delegations``, ``goal``, ``background_tasks``. Mutations stay
  on the branch and are discarded after the result is extracted.
- **Shared resources**: sandbox/workspace paths are reused. File reads and
  writes run on that shared workspace; same-path mutations serialize through
  the existing sandbox / read-before-write locks.

Branch-specific instructions MUST be a trailing HumanMessage suffix. Never
prepend or inject a SystemMessage — ``SystemMessageCoalescingMiddleware``
would rewrite the leading prefix and destroy KV/prefix cache reuse.
"""

from __future__ import annotations

import copy
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

FORK_INSTRUCTION_KEY = "deerflow_fork_instruction"

# Keys whose values are reused by reference (sandbox/workspace) or as a shallow
# list of the same message/skill objects so the model prefix stays identical.
_SHARED_KEYS = ("sandbox", "thread_data", "title", "summary_text", "uploaded_files")
_SNAPSHOT_KEYS = (
    "todos",
    "artifacts",
    "viewed_images",
    "promoted",
    "delegations",
    "skill_context",
    "goal",
    "background_tasks",
)


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if isinstance(name, str):
            return name
        function = tool_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
    return ""


def strip_in_flight_fork_message(messages: list[Any] | None) -> list[Any]:
    """Drop the trailing AIMessage that dispatched the current ``fork_task`` batch.

    That in-flight turn still has dangling tool calls (sibling forks have not
    returned). Forks must not see it, or they inherit an incomplete tool loop.
    The shared prefix therefore ends at the last completed parent turn.
    """
    if not messages:
        return []
    cloned = list(messages)
    last = cloned[-1]
    if not isinstance(last, AIMessage):
        return cloned
    tool_calls = getattr(last, "tool_calls", None) or []
    if any(_tool_call_name(tool_call) == "fork_task" for tool_call in tool_calls):
        return cloned[:-1]
    return cloned


def build_fork_instruction_message(prompt: str) -> HumanMessage:
    """Return the cache-boundary suffix for one forked branch.

    This is a HumanMessage on purpose. A SystemMessage here would be coalesced
    to the front of the request and diverge the A/B/C prefixes.
    """
    text = prompt.strip()
    return HumanMessage(
        content=(
            "<fork-task>\n"
            f"{text}\n"
            "</fork-task>\n"
            "You are an ephemeral branch of the current agent. You inherit this conversation; "
            "stay continuous with that context and complete only this forked exploration using "
            "the inherited tools. Prefer a short final answer; do not start a long tool loop. "
            "You may read and write workspace files. Do not call fork_task or task. "
            "Return a concise result the parent can merge."
        ),
        additional_kwargs={FORK_INSTRUCTION_KEY: True},
    )


def fork_state(parent_state: Any, prompt: str) -> dict[str, Any]:
    """Build an ephemeral branch state from the lead agent's current ThreadState."""
    source = parent_state if isinstance(parent_state, dict) else {}
    branch: dict[str, Any] = {}

    for key in _SHARED_KEYS:
        if key in source:
            branch[key] = source[key]

    for key in _SNAPSHOT_KEYS:
        if key in source:
            branch[key] = copy.deepcopy(source[key])

    branch["messages"] = [
        *strip_in_flight_fork_message(source.get("messages")),
        build_fork_instruction_message(prompt),
    ]
    return branch
