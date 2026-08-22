"""Block nested delegation and user interrupts inside a forked branch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.tool_result_meta import normalize_tool_result
from deerflow.forks.runtime import is_fork_runtime

# Tool schemas stay on the model (prefix cache). Nested agents and user
# interrupts are denied at execution; file reads and writes are allowed.
FORK_BLOCKED_TOOLS = frozenset(
    {
        "fork_task",
        "task",
        "ask_clarification",
        "setup_agent",
        "update_agent",
        "present_files",
    }
)

_BLOCK_MESSAGE = "Error: {tool_name} is blocked on a forked branch. This branch inherits the parent workspace and must not spawn nested agents or interrupt the user. Complete the work with the inherited tools and return a concise result."


class ForkExecutionGuardMiddleware(AgentMiddleware[AgentState]):
    """No-op on the lead path; denies unsafe tools when ``context.is_fork`` is set."""

    def _blocked_tool_message(self, request: ToolCallRequest) -> ToolMessage | None:
        runtime = getattr(request, "runtime", None)
        if not is_fork_runtime(runtime):
            return None
        name = str(request.tool_call.get("name") or "")
        if name not in FORK_BLOCKED_TOOLS:
            return None
        tool_call_id = str(request.tool_call.get("id") or "missing_tool_call_id")
        return normalize_tool_result(
            ToolMessage(
                content=_BLOCK_MESSAGE.format(tool_name=name),
                tool_call_id=tool_call_id,
                name=name,
                status="error",
            )
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        blocked = self._blocked_tool_message(request)
        if blocked is not None:
            return blocked
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        blocked = self._blocked_tool_message(request)
        if blocked is not None:
            return blocked
        return await handler(request)
