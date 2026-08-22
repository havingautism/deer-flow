"""Tests for cache-friendly ThreadState forks."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deerflow.forks.state import FORK_INSTRUCTION_KEY, fork_state, strip_in_flight_fork_message


def test_fork_state_reuses_message_objects_and_appends_suffix():
    first = HumanMessage(content="hello", id="m1")
    parent = {
        "messages": [first],
        "sandbox": {"sandbox_id": "sbx-1"},
        "thread_data": {"workspace_path": "/ws"},
        "todos": [{"id": "1", "content": "do", "status": "pending"}],
        "skill_context": [{"name": "x", "path": "/x", "description": "d", "loaded_at": 1}],
        "summary_text": "summary",
    }

    branch = fork_state(parent, "analyze option A")

    assert branch["messages"][0] is first
    assert branch["sandbox"] is parent["sandbox"]
    assert branch["thread_data"] is parent["thread_data"]
    assert branch["summary_text"] == "summary"
    assert branch["todos"] == parent["todos"]
    assert branch["todos"] is not parent["todos"]
    suffix = branch["messages"][-1]
    assert isinstance(suffix, HumanMessage)
    assert suffix.additional_kwargs.get(FORK_INSTRUCTION_KEY) is True
    assert "analyze option A" in suffix.content
    assert isinstance(suffix, HumanMessage)
    assert not isinstance(suffix, SystemMessage)


def test_fork_state_strips_in_flight_fork_tool_calls():
    history = HumanMessage(content="context", id="h1")
    dispatch = AIMessage(
        content="",
        id="ai1",
        tool_calls=[
            {"name": "fork_task", "args": {"prompt": "A"}, "id": "c1", "type": "tool_call"},
            {"name": "fork_task", "args": {"prompt": "B"}, "id": "c2", "type": "tool_call"},
        ],
    )
    stripped = strip_in_flight_fork_message([history, dispatch])
    assert stripped == [history]

    branch = fork_state({"messages": [history, dispatch]}, "A")
    assert branch["messages"][0] is history
    assert branch["messages"][-1].content.startswith("<fork-task>")


def test_fork_instruction_is_never_a_system_message():
    branch = fork_state({"messages": [HumanMessage(content="p")]}, "task A")
    assert all(not isinstance(message, SystemMessage) for message in branch["messages"])
