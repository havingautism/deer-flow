"""Tests for ForkExecutionGuardMiddleware."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage

from deerflow.forks.guard import FORK_BLOCKED_TOOLS, ForkExecutionGuardMiddleware


def _request(name: str, *, is_fork: bool) -> MagicMock:
    request = MagicMock()
    request.tool_call = {"name": name, "id": "tc-1"}
    request.runtime = SimpleNamespace(context={"is_fork": is_fork} if is_fork else {})
    return request


def test_guard_is_noop_on_lead_path():
    mw = ForkExecutionGuardMiddleware()
    handler = MagicMock(return_value="ok")
    assert mw.wrap_tool_call(_request("write_file", is_fork=False), handler) == "ok"
    handler.assert_called_once()


def test_guard_blocks_writes_and_nested_delegation_on_fork():
    mw = ForkExecutionGuardMiddleware()
    handler = MagicMock(return_value="ok")
    for name in ("write_file", "str_replace", "fork_task", "task", "ask_clarification"):
        result = mw.wrap_tool_call(_request(name, is_fork=True), handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert name in result.content
    handler.assert_not_called()
    assert "write_file" in FORK_BLOCKED_TOOLS


def test_guard_allows_read_tools_on_fork():
    mw = ForkExecutionGuardMiddleware()
    handler = MagicMock(return_value="ok")
    assert mw.wrap_tool_call(_request("read_file", is_fork=True), handler) == "ok"
    handler.assert_called_once()
