"""Fork-only token accounting — do not reuse from the `task` subagent path.

Sibling `fork_task` calls share the lead callback manager. A dedicated inline
collector plus message harvest keeps each branch's card on its own LLM turns
without changing ``SubagentTokenCollector``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage

from deerflow.subagents.token_collector import SubagentTokenCollector


class ForkTokenCollector(SubagentTokenCollector):
    """Same record shape as the subagent collector, dispatched inline.

    LangChain's async manager otherwise runs sync handlers via
    ``run_in_executor``, so ``astream`` can return before later turns land.
    """

    run_inline = True


def _usage_mapping(usage: Any) -> dict[str, Any]:
    if isinstance(usage, Mapping):
        return dict(usage)
    return {}


def _cache_read_tokens(usage: Mapping[str, Any]) -> int:
    details = usage.get("input_token_details") or {}
    if not isinstance(details, Mapping):
        return 0
    try:
        return max(int(details.get("cache_read") or 0), 0)
    except (TypeError, ValueError):
        return 0


def _record_from_usage(
    usage: Any,
    *,
    caller: str,
    source_run_id: str,
    model_name: str | None = None,
) -> dict[str, int | str | None] | None:
    usage_dict = _usage_mapping(usage)
    input_tk = usage_dict.get("input_tokens", 0) or 0
    output_tk = usage_dict.get("output_tokens", 0) or 0
    total_tk = usage_dict.get("total_tokens", 0) or 0
    if total_tk <= 0:
        total_tk = input_tk + output_tk
    if total_tk <= 0:
        return None
    cache_read_tk = _cache_read_tokens(usage_dict)
    if cache_read_tk <= 0:
        try:
            cache_read_tk = max(int(usage_dict.get("cache_read_tokens") or 0), 0)
        except (TypeError, ValueError):
            cache_read_tk = 0
    record: dict[str, int | str | None] = {
        "source_run_id": source_run_id,
        "caller": caller,
        "model_name": model_name,
        "input_tokens": input_tk,
        "output_tokens": output_tk,
        "total_tokens": total_tk,
    }
    if cache_read_tk > 0:
        record["cache_read_tokens"] = cache_read_tk
    return record


def records_from_ai_messages(
    messages: Sequence[Any] | None,
    *,
    caller: str,
    skip_objects: Sequence[Any] | None = None,
) -> list[dict[str, int | str | None]]:
    """Usage from this branch's assistant messages, skipping inherited parent turns."""
    skip_ids = {id(obj) for obj in skip_objects or ()}
    skip_msg_ids = {
        message_id
        for obj in skip_objects or ()
        if isinstance((message_id := getattr(obj, "id", None)), str) and message_id
    }
    records: list[dict[str, int | str | None]] = []
    for index, message in enumerate(messages or ()):
        if not isinstance(message, AIMessage):
            continue
        if id(message) in skip_ids:
            continue
        message_id = getattr(message, "id", None)
        if isinstance(message_id, str) and message_id in skip_msg_ids:
            continue
        response_metadata = getattr(message, "response_metadata", None) or {}
        model_name: str | None = None
        if isinstance(response_metadata, Mapping):
            model_name = response_metadata.get("model_name") or response_metadata.get("model")
        source_id = message_id if isinstance(message_id, str) and message_id else f"{caller}:msg:{index}"
        record = _record_from_usage(
            getattr(message, "usage_metadata", None),
            caller=caller,
            source_run_id=source_id,
            model_name=model_name if isinstance(model_name, str) else None,
        )
        if record is not None:
            records.append(record)
    return records


def _total_tokens(records: Sequence[Mapping[str, Any]] | None) -> int:
    return sum(int(record.get("total_tokens") or 0) for record in records or ())


def prefer_usage_records(
    collected: Sequence[dict[str, int | str | None]] | None,
    harvested: Sequence[dict[str, int | str | None]] | None,
) -> list[dict[str, int | str | None]]:
    """Prefer harvested branch messages when they cover more of the run."""
    collected_records = list(collected or ())
    harvested_records = list(harvested or ())
    if _total_tokens(harvested_records) > _total_tokens(collected_records):
        return harvested_records
    return collected_records
