"""Execute an ephemeral forked branch against a copy of the lead graph."""

from __future__ import annotations

import copy
import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from deerflow.forks.host import get_fork_host_graph
from deerflow.forks.result import ForkResult
from deerflow.forks.state import fork_state
from deerflow.subagents.token_collector import (
    SubagentTokenCollector,
    model_name_from_usage_records,
    summarize_token_usage_records,
)
from deerflow.utils.messages import message_content_to_text

logger = logging.getLogger(__name__)

DEFAULT_FORK_MAX_TURNS = 25
# LangGraph ``recursion_limit`` counts super-steps (middleware nodes + model +
# tools), not conversational turns. The lead graph has a long one-shot
# ``before_agent`` chain, so a 25-turn product cap must not be used raw.
_FORK_STEPS_PER_TURN = 8
_FORK_GRAPH_OVERHEAD_STEPS = 48


def fork_recursion_limit(max_turns: int) -> int:
    """Translate a conversational turn cap into a LangGraph super-step budget."""
    return max_turns * _FORK_STEPS_PER_TURN + _FORK_GRAPH_OVERHEAD_STEPS


def _ephemeral_graph(graph: Any) -> Any:
    """Shallow-copy the lead graph and disable checkpoint persistence."""
    fork_graph = copy.copy(graph)
    try:
        fork_graph.checkpointer = False
    except Exception:
        logger.debug("Could not clear checkpointer on forked graph copy", exc_info=True)
    return fork_graph


def extract_branch_result(result_state: Any) -> tuple[str, list[str]]:
    """Return (final assistant text, tool names used) from a branch state."""
    if not isinstance(result_state, dict):
        return "", []
    messages = result_state.get("messages") or []
    receipts: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name:
            receipts.append(str(message.name))
    fallback = ""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = message_content_to_text(message.content).strip()
        if not text:
            continue
        if not getattr(message, "tool_calls", None):
            return text, receipts
        if not fallback:
            fallback = text
    return fallback, receipts


def _is_parent_usage_handler(handler: Any) -> bool:
    """Parent RunJournal must not see fork LLM events.

    Those tokens belong on the fork card. If the journal counted them, they
    would land in ``lead_agent_tokens`` and mix into the original header total.
    """
    return getattr(handler, "deerflow_loop_bound", False) is True or hasattr(handler, "record_external_llm_usage_records")


def _child_callbacks(parent_callbacks: Any, collector: SubagentTokenCollector) -> list[Any]:
    """Copy non-journal parent handlers and append the per-fork collector."""
    raw: list[Any]
    if isinstance(parent_callbacks, BaseCallbackManager):
        raw = list(parent_callbacks.handlers)
    elif isinstance(parent_callbacks, list):
        raw = list(parent_callbacks)
    elif parent_callbacks is not None:
        raw = [parent_callbacks]
    else:
        raw = []
    return [handler for handler in raw if not _is_parent_usage_handler(handler)] + [collector]


def _stamp_usage(result: ForkResult, collector: SubagentTokenCollector) -> ForkResult:
    records = collector.snapshot_records()
    result.token_usage_records = records
    result.token_usage = summarize_token_usage_records(records)
    result.model_name = model_name_from_usage_records(records)
    return result


def _artifacts(result_state: Any) -> list[str]:
    if isinstance(result_state, dict) and isinstance(result_state.get("artifacts"), list):
        return [str(item) for item in result_state["artifacts"]]
    return []


def _result_from_state(
    result_state: Any,
    *,
    task_id: str,
    max_turns: int,
    turn_capped: bool = False,
) -> ForkResult:
    text, receipts = extract_branch_result(result_state)
    artifacts = _artifacts(result_state)
    if text:
        return ForkResult(
            task_id=task_id,
            status="completed",
            result=text,
            artifacts=artifacts,
            tool_receipts=receipts,
            stop_reason="turn_capped" if turn_capped else None,
        )
    if turn_capped:
        return ForkResult(
            task_id=task_id,
            status="failed",
            error=f"Fork reached the {max_turns}-turn cap without a final answer.",
            artifacts=artifacts,
            tool_receipts=receipts,
            stop_reason="turn_capped",
        )
    return ForkResult(
        task_id=task_id,
        status="failed",
        error="Fork completed without a final assistant answer.",
        artifacts=artifacts,
        tool_receipts=receipts,
    )


class ForkExecutor:
    """Run one cache-friendly branch from the current lead state."""

    def __init__(
        self,
        *,
        graph: Any | None = None,
        max_turns: int = DEFAULT_FORK_MAX_TURNS,
        parent_context: dict[str, Any] | None = None,
        parent_config: dict[str, Any] | None = None,
    ) -> None:
        self.graph = graph
        self.max_turns = max_turns
        self.parent_context = parent_context if isinstance(parent_context, dict) else {}
        self.parent_config = parent_config if isinstance(parent_config, dict) else {}

    async def aexecute(self, *, prompt: str, parent_state: Any, task_id: str) -> ForkResult:
        graph = self.graph if self.graph is not None else get_fork_host_graph()
        if graph is None:
            return ForkResult(
                task_id=task_id,
                status="failed",
                error="Fork host graph is unavailable; cannot branch from the lead agent.",
            )

        branch_state = fork_state(parent_state, prompt)
        collector = SubagentTokenCollector(caller=f"fork:{task_id}")
        # Do not put checkpoint coordinates in the child config. Passing
        # thread_id/checkpoint_ns starts a new root lineage and can leak child
        # messages into the parent stream (same contract as SubagentExecutor).
        run_config: RunnableConfig = {
            "recursion_limit": fork_recursion_limit(self.max_turns),
            "callbacks": _child_callbacks(self.parent_config.get("callbacks"), collector),
            "tags": [f"fork:{task_id}"],
        }
        parent_metadata = self.parent_config.get("metadata")
        if isinstance(parent_metadata, dict):
            run_config["metadata"] = dict(parent_metadata)

        context = {**self.parent_context, "is_fork": True}
        final_state: Any = None

        try:
            async for chunk in _ephemeral_graph(graph).astream(
                branch_state,
                config=run_config,
                context=context,
                stream_mode="values",
            ):
                final_state = chunk
        except GraphRecursionError:
            logger.warning("Forked branch %s reached max_turns=%s", task_id, self.max_turns)
            return _stamp_usage(
                _result_from_state(
                    final_state,
                    task_id=task_id,
                    max_turns=self.max_turns,
                    turn_capped=True,
                ),
                collector,
            )
        except Exception as exc:
            logger.exception("Forked branch %s failed", task_id)
            return _stamp_usage(ForkResult(task_id=task_id, status="failed", error=str(exc)), collector)

        return _stamp_usage(
            _result_from_state(final_state, task_id=task_id, max_turns=self.max_turns),
            collector,
        )
