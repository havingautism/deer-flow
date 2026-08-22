"""Tests for forked-branch execution and the fork_task tool."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from deerflow.forks.executor import ForkExecutor, extract_branch_result, fork_recursion_limit
from deerflow.forks.result import ForkResult
from deerflow.subagents.status_contract import SUBAGENT_MODEL_NAME_KEY, SUBAGENT_TOKEN_USAGE_KEY

fork_task_module = importlib.import_module("deerflow.tools.builtins.fork_task_tool")


def _astream(chunks, *, error=None, captured=None):
    async def fake_astream(*_args, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        for chunk in chunks:
            yield chunk
        if error is not None:
            raise error

    return fake_astream


def test_extract_branch_result_returns_final_assistant_text_only():
    text, receipts = extract_branch_result(
        {
            "messages": [
                HumanMessage(content="inherited"),
                AIMessage(content="", tool_calls=[{"name": "read_file", "id": "1", "args": {}}]),
                ToolMessage(content="file body", tool_call_id="1", name="read_file"),
                AIMessage(content="Option A is safer."),
            ]
        }
    )
    assert text == "Option A is safer."
    assert receipts == ["read_file"]


def test_extract_branch_result_falls_back_to_tool_call_text():
    text, receipts = extract_branch_result(
        {
            "messages": [
                AIMessage(
                    content="12345 × 6789 = 83810205",
                    tool_calls=[{"name": "bash", "id": "1", "args": {}}],
                ),
                ToolMessage(content="ok", tool_call_id="1", name="bash"),
            ]
        }
    )
    assert text == "12345 × 6789 = 83810205"
    assert receipts == ["bash"]


def test_fork_recursion_limit_includes_middleware_overhead():
    assert fork_recursion_limit(25) > 25
    assert fork_recursion_limit(1) > 1


def test_executor_does_not_merge_child_state(monkeypatch):
    captured: dict = {}
    graph = MagicMock()
    graph.astream = _astream(
        [
            {
                "messages": [
                    HumanMessage(content="parent"),
                    AIMessage(content="branch answer"),
                ],
                "todos": [{"id": "x"}],
            }
        ],
        captured=captured,
    )
    monkeypatch.setattr("deerflow.forks.executor._ephemeral_graph", lambda g: g)

    result = asyncio.run(
        ForkExecutor(graph=graph, parent_context={"thread_id": "t1"}).aexecute(
            prompt="check A",
            parent_state={"messages": [HumanMessage(content="parent")], "todos": []},
            task_id="fork-1",
        )
    )
    assert result.status == "completed"
    assert result.result == "branch answer"
    assert captured["context"]["is_fork"] is True
    assert captured["config"]["recursion_limit"] == fork_recursion_limit(25)
    assert "thread_id" not in (captured["config"].get("configurable") or {})


class _FakeCollector:
    def __init__(self, caller):
        self.caller = caller

    def snapshot_records(self):
        return [
            {
                "source_run_id": "fork-run",
                "caller": self.caller,
                "model_name": "test-model",
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "cache_read_tokens": 80,
            }
        ]


def test_executor_stamps_usage_without_parent_journal(monkeypatch):
    captured: dict = {}
    graph = MagicMock()
    graph.astream = _astream(
        [{"messages": [AIMessage(content="branch answer")]}],
        captured=captured,
    )
    monkeypatch.setattr("deerflow.forks.executor._ephemeral_graph", lambda g: g)
    monkeypatch.setattr("deerflow.forks.executor.SubagentTokenCollector", _FakeCollector)

    journal = SimpleNamespace(deerflow_loop_bound=True, record_external_llm_usage_records=lambda *_: None)
    tracer = SimpleNamespace(name="trace")
    result = asyncio.run(
        ForkExecutor(
            graph=graph,
            parent_config={"callbacks": [journal, tracer]},
        ).aexecute(
            prompt="check A",
            parent_state={"messages": [HumanMessage(content="parent")]},
            task_id="fork-1",
        )
    )
    assert result.token_usage == {
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
        "cache_read_tokens": 80,
    }
    assert result.model_name == "test-model"
    callbacks = captured["config"]["callbacks"]
    assert journal not in callbacks
    assert tracer in callbacks
    assert any(getattr(cb, "caller", "") == "fork:fork-1" for cb in callbacks)
    assert captured["config"]["tags"] == ["fork:fork-1"]


def test_executor_keeps_partial_answer_when_turn_capped(monkeypatch):
    graph = MagicMock()
    graph.astream = _astream(
        [{"messages": [AIMessage(content="12345 × 6789 = 83810205")]}],
        error=GraphRecursionError("Recursion limit of 25 reached"),
    )
    monkeypatch.setattr("deerflow.forks.executor._ephemeral_graph", lambda g: g)

    result = asyncio.run(
        ForkExecutor(graph=graph).aexecute(
            prompt="multiply",
            parent_state={"messages": [HumanMessage(content="parent")]},
            task_id="fork-cap",
        )
    )
    assert result.status == "completed"
    assert result.result == "12345 × 6789 = 83810205"
    assert result.stop_reason == "turn_capped"


def test_executor_fails_turn_cap_without_answer(monkeypatch):
    graph = MagicMock()
    graph.astream = _astream(
        [{"messages": [AIMessage(content="", tool_calls=[{"name": "ls", "id": "1", "args": {}}])]}],
        error=GraphRecursionError("Recursion limit of 25 reached"),
    )
    monkeypatch.setattr("deerflow.forks.executor._ephemeral_graph", lambda g: g)

    result = asyncio.run(
        ForkExecutor(graph=graph, max_turns=25).aexecute(
            prompt="explore",
            parent_state={"messages": [HumanMessage(content="parent")]},
            task_id="fork-empty",
        )
    )
    assert result.status == "failed"
    assert result.stop_reason == "turn_capped"
    assert "25-turn cap" in (result.error or "")


def _run_fork_tool(**kwargs):
    coroutine = getattr(fork_task_module.fork_task_tool, "coroutine", None)
    assert coroutine is not None
    return asyncio.run(coroutine(**kwargs))


def test_fork_task_tool_returns_parent_command(monkeypatch):
    async def fake_aexecute(self, **kwargs):
        return ForkResult(
            task_id=kwargs["task_id"],
            status="completed",
            result="done",
            token_usage={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "cache_read_tokens": 80,
            },
            model_name="test-model",
        )

    monkeypatch.setattr(fork_task_module.ForkExecutor, "aexecute", fake_aexecute)
    monkeypatch.setattr(fork_task_module, "get_stream_writer", lambda: lambda _event: None)

    async def _no_event(*_args, **_kwargs):
        return None

    monkeypatch.setattr(fork_task_module, "aemit_custom_event", _no_event)

    runtime = SimpleNamespace(state={"messages": []}, context={"thread_id": "t"}, config={})
    output = _run_fork_tool(runtime=runtime, prompt="check A", tool_call_id="tc-fork")
    assert isinstance(output, Command)
    message = output.update["messages"][0]
    assert message.name == "fork_task"
    assert "done" in message.content
    assert message.tool_call_id == "tc-fork"
    assert message.additional_kwargs[SUBAGENT_MODEL_NAME_KEY] == "test-model"
    assert message.additional_kwargs[SUBAGENT_TOKEN_USAGE_KEY]["cache_read_tokens"] == 80


def test_nested_fork_task_is_rejected(monkeypatch):
    monkeypatch.setattr(fork_task_module, "get_stream_writer", lambda: lambda _event: None)

    async def _no_event(*_args, **_kwargs):
        return None

    monkeypatch.setattr(fork_task_module, "aemit_custom_event", _no_event)
    runtime = SimpleNamespace(state={"messages": []}, context={"is_fork": True}, config={})
    output = _run_fork_tool(runtime=runtime, prompt="nested", tool_call_id="tc-nested")
    message = output.update["messages"][0]
    assert "Nested fork_task" in message.content
