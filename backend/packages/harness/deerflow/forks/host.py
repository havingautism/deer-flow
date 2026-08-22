"""Capture the live lead graph so fork_task can invoke an ephemeral copy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

_fork_host_graph: ContextVar[Any | None] = ContextVar("deerflow_fork_host_graph", default=None)


def get_fork_host_graph() -> Any | None:
    """Return the lead compiled graph for the current run, if captured."""
    return _fork_host_graph.get()


class ForkHostMiddleware(AgentMiddleware[AgentState]):
    """Publish the lead compiled graph on a ContextVar for the duration of a run.

    The graph reference is set by ``make_lead_agent`` after ``create_agent``
    returns. Storing it on a ContextVar (not runtime.context) keeps it out of
    checkpoints.

    ``before_agent`` is a separate LangGraph node from tool execution. Node
    boundaries do not preserve ContextVars, so the graph is also published in
    ``wrap_tool_call`` / ``awrap_tool_call`` (same task as ``fork_task``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.graph: Any | None = None

    def _publish(self) -> None:
        if self.graph is not None:
            _fork_host_graph.set(self.graph)

    @override
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        self._publish()
        return None

    @override
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        self._publish()
        return None

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._publish()
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._publish()
        return await handler(request)
