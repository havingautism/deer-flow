"""Tests for ForkHostMiddleware graph publishing across LangGraph nodes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from deerflow.forks import host as host_module
from deerflow.forks.host import ForkHostMiddleware, get_fork_host_graph


def _reset_host_graph():
    host_module._fork_host_graph.set(None)


def test_before_agent_publishes_attached_graph():
    _reset_host_graph()
    mw = ForkHostMiddleware()
    graph = object()
    mw.graph = graph
    mw.before_agent({}, None)
    assert get_fork_host_graph() is graph
    _reset_host_graph()


def test_wrap_tool_call_republishes_after_node_boundary():
    """before_agent is a different LangGraph node than tools.

    ContextVars set there are not visible when fork_task runs. wrap_tool_call
    must republish so ForkExecutor can resolve the lead graph.
    """
    _reset_host_graph()
    mw = ForkHostMiddleware()
    graph = object()
    mw.graph = graph
    mw.before_agent({}, None)
    assert get_fork_host_graph() is graph

    _reset_host_graph()
    assert get_fork_host_graph() is None

    seen = {}

    def handler(request):
        seen["graph"] = get_fork_host_graph()
        return "ok"

    assert mw.wrap_tool_call(SimpleNamespace(), handler) == "ok"
    assert seen["graph"] is graph
    _reset_host_graph()


def test_awrap_tool_call_republishes_in_async_tool_task():
    _reset_host_graph()
    mw = ForkHostMiddleware()
    graph = object()
    mw.graph = graph

    async def handler(request):
        return get_fork_host_graph()

    result = asyncio.run(mw.awrap_tool_call(SimpleNamespace(), handler))
    assert result is graph
    _reset_host_graph()


def test_publish_is_noop_until_graph_is_attached():
    _reset_host_graph()
    mw = ForkHostMiddleware()
    mw.wrap_tool_call(SimpleNamespace(), lambda _request: "ok")
    assert get_fork_host_graph() is None
    _reset_host_graph()


def test_awrap_tool_call_forwards_to_handler():
    _reset_host_graph()
    mw = ForkHostMiddleware()
    mw.graph = object()
    handler = AsyncMock(return_value="forked")
    request = SimpleNamespace()
    result = asyncio.run(mw.awrap_tool_call(request, handler))
    assert result == "forked"
    handler.assert_awaited_once_with(request)
    _reset_host_graph()
