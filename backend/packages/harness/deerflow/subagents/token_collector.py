"""Callback handler that collects LLM token usage within a subagent.

Each subagent execution creates its own collector. After the subagent
finishes, the collected records are transferred to the parent RunJournal
via :meth:`RunJournal.record_external_llm_usage_records`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class SubagentTokenCollector(BaseCallbackHandler):
    """Lightweight callback handler that collects LLM token usage within a subagent."""

    def __init__(self, caller: str):
        super().__init__()
        self.caller = caller
        self._records: list[dict[str, int | str | None]] = []
        self._counted_run_ids: set[str] = set()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid in self._counted_run_ids:
            return

        for generation in response.generations:
            for gen in generation:
                if not hasattr(gen, "message"):
                    continue
                usage = getattr(gen.message, "usage_metadata", None)
                usage_dict = dict(usage) if usage else {}
                input_tk = usage_dict.get("input_tokens", 0) or 0
                output_tk = usage_dict.get("output_tokens", 0) or 0
                total_tk = usage_dict.get("total_tokens", 0) or 0
                if total_tk <= 0:
                    total_tk = input_tk + output_tk
                if total_tk <= 0:
                    continue
                # Prompt-cache hits (needed for cache-aware cost accounting)
                details = usage_dict.get("input_token_details") or {}
                cache_read_tk = 0
                if isinstance(details, Mapping):
                    try:
                        cache_read_tk = max(int(details.get("cache_read") or 0), 0)
                    except (TypeError, ValueError):
                        cache_read_tk = 0
                # Capture the model that actually produced this response so the
                # parent journal can bucket tokens by real model rather than the
                # lead agent's resolved model
                response_metadata = getattr(gen.message, "response_metadata", None) or {}
                model_name: str | None = None
                if isinstance(response_metadata, Mapping):
                    model_name = response_metadata.get("model_name") or response_metadata.get("model")
                self._counted_run_ids.add(rid)
                record: dict[str, int | str | None] = {
                    "source_run_id": rid,
                    "caller": self.caller,
                    "model_name": model_name,
                    "input_tokens": input_tk,
                    "output_tokens": output_tk,
                    "total_tokens": total_tk,
                }
                # Sparse, matching the journal's per-model buckets: the key is
                # only present when the provider actually reported cache hits.
                if cache_read_tk > 0:
                    record["cache_read_tokens"] = cache_read_tk
                self._records.append(record)
                return

    def snapshot_records(self) -> list[dict[str, int | str | None]]:
        """Return a copy of the accumulated usage records."""
        return list(self._records)


def summarize_token_usage_records(records: list[dict] | None) -> dict[str, int] | None:
    """Collapse collector records into the card/ToolMessage usage snapshot.

    ``cache_read_tokens`` is sparse: included only when the summed prompt-cache
    hits are greater than zero.
    """
    if not records:
        return None
    input_tokens = sum(int(r.get("input_tokens") or 0) for r in records)
    output_tokens = sum(int(r.get("output_tokens") or 0) for r in records)
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in records)
    cache_read_tokens = sum(int(r.get("cache_read_tokens") or 0) for r in records)
    if input_tokens <= 0 and output_tokens <= 0 and total_tokens <= 0:
        return None
    usage: dict[str, int] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if cache_read_tokens > 0:
        usage["cache_read_tokens"] = cache_read_tokens
    return usage


def model_name_from_usage_records(records: list[dict] | None) -> str | None:
    """Return the last non-blank model name reported by a collector."""
    if not records:
        return None
    for record in reversed(records):
        name = record.get("model_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None
