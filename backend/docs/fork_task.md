# fork_task: cache-friendly lead-agent state forks

`fork_task` is **not** a subagent. It branches from the lead agent's current `ThreadState` so several short/medium explorations can share one prefix (system prompt, tools, conversation) and only differ by a trailing instruction.

It coexists with `task()`:

| | `task` | `fork_task` |
|---|---|---|
| Context | scoped / fresh `ThreadState` | inherit lead state |
| System prompt | subagent prompt | same as lead |
| Tool schemas | subagent tools | same as lead (writes denied at execution) |
| Messages | newly constructed | parent messages + suffix |
| Prefix / KV cache | rebuilt | shared prefix |
| Fit | independent long-horizon work | parallel exploration of the current problem |
| Merge | subagent result | `ForkResult` only — never the child loop |

## Prefix-cache contract

What the model actually sees must stay identical across sibling forks **until the last suffix**:

- system prompt
- tool definitions and ordering
- inherited messages
- model params

Branch-specific text **must** be a trailing `HumanMessage`. Do **not** inject a `SystemMessage("you are Task A")`. `SystemMessageCoalescingMiddleware` would merge that to the front and make A/B/C prefixes diverge, destroying KV cache.

```
[identical parent prefix]
────────────────────────── cache boundary
<fork-task> instruction for this branch
```

`fork_state()` strips the in-flight lead `AIMessage` that contains the `fork_task` batch (dangling sibling tool calls) before appending that suffix.

## State classification

`fork_state()` does not `deepcopy` the whole parent state.

- **Shared**: `sandbox`, `thread_data`, `title`, `summary_text`, `uploaded_files`, and the parent message objects themselves.
- **Branch-local snapshot**: `todos`, `artifacts`, `viewed_images`, `promoted`, `delegations`, `skill_context`, `goal`, `background_tasks`.
- **Not merged back**: the branch's internal `AIMessage` / `ToolMessage` loop is discarded. Only `ForkResult` returns to the parent.

## Shared workspace (MVP)

Forks reuse the parent sandbox/workspace. Parallel writes would race even with `str_replace` path locks. MVP therefore **keeps write tools in the schema** (so the prefix stays identical) and **denies execution** via `ForkExecutionGuardMiddleware` when `context.is_fork` is set:

- blocked: `write_file`, `str_replace`, `fork_task`, `task`, `ask_clarification`, `setup_agent`, `update_agent`, `present_files`
- allowed: read/search/web, and bash with a prompt warning not to mutate files

A later iteration can give each branch a worktree.

## Runtime

- Tool: `fork_task(prompt)`
- Enable: `configurable.fork_enabled` (defaults to `subagent_enabled`)
- Host graph: `ForkHostMiddleware` attaches the compiled lead graph and republishes it in `wrap_tool_call` (tool-node ContextVar), not only `before_agent`
- Concurrency: counted together with `task` by `SubagentLimitMiddleware`
- Recursion: conversational cap `DEFAULT_FORK_MAX_TURNS = 25`, mapped to a higher LangGraph `recursion_limit` because that counter includes middleware nodes. A usable partial answer is kept if the cap still trips.
- Persistence: ephemeral copy of the lead graph with `checkpointer=False`
- Isolation: child `RunnableConfig` omits `thread_id` / checkpoint coordinates (same stream-leak contract as subagents)
- Side effects: memory and title middlewares no-op when `is_fork` is true

## Token usage in the UI

Fork usage stays on the `fork_task` card. It is **not** merged into the lead turn's original token usage (header / per-turn / debug).

- Lead `usage_metadata` and the header total keep the original lead (and `task()`) accounting
- Each fork card shows that branch's tokens in an expanded footer badge (same shape as the header token chip). Prompt-cache hits are included when the provider reports `cache_read`.
- `TokenUsageMiddleware` skips `fork_task` ToolMessages so branch tokens cannot fold into the dispatching AIMessage
