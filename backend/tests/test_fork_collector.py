"""Fork-only token harvest — isolated from the `task` subagent collector."""

from langchain_core.messages import AIMessage, HumanMessage

from deerflow.forks.collector import prefer_usage_records, records_from_ai_messages


def test_records_from_ai_messages_skip_inherited_parent_objects():
    parent = AIMessage(
        id="parent-ai",
        content="parent",
        usage_metadata={"input_tokens": 20000, "output_tokens": 10, "total_tokens": 20010},
    )
    branch = AIMessage(
        id="fork-ai",
        content="branch",
        usage_metadata={
            "input_tokens": 21000,
            "output_tokens": 80,
            "total_tokens": 21080,
            "input_token_details": {"cache_read": 14000},
        },
    )
    records = records_from_ai_messages(
        [parent, HumanMessage(content="suffix"), branch],
        caller="fork:1",
        skip_objects=[parent],
    )
    assert len(records) == 1
    assert records[0]["source_run_id"] == "fork-ai"
    assert records[0]["total_tokens"] == 21080
    assert records[0]["cache_read_tokens"] == 14000


def test_prefer_usage_records_uses_harvest_when_it_covers_more():
    collected = [{"total_tokens": 100, "input_tokens": 90, "output_tokens": 10}]
    harvested = [
        {"total_tokens": 100, "input_tokens": 90, "output_tokens": 10},
        {"total_tokens": 40, "input_tokens": 30, "output_tokens": 10},
    ]
    assert prefer_usage_records(collected, harvested) == harvested
    assert prefer_usage_records(collected, []) == collected
