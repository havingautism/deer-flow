# fork_task: cache-friendly lead-agent state forks

`fork_task` is **not** a subagent. It forks the lead agent's current `ThreadState` so related work stays continuous with this conversation, can run in parallel, and can reuse prefix/KV cache. Only a trailing instruction differs across sibling branches.

It coexists with `task()` — do not change `task`; that tool is the isolated subagent for independent, context-free work:

| | `task` | `fork_task` |
|---|---|---|
| Role | isolated subagent | lead-state fork |
| Context | scoped / fresh `ThreadState` | inherit lead state |
| System prompt | subagent prompt | same as lead |
| Tool schemas | subagent tools | same as lead |
| Messages | newly constructed | parent messages + suffix |
| Prefix / KV cache | rebuilt | shared prefix |
| Fit | independent / context-free / long-horizon | related, parallel, cache-friendly continuation |
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

The sandbox is the one deliberate ownership handoff: fork middleware never
releases it. `ForkResult` carries the effective `sandbox_id` into the parent
`Command`, so the lead state remains the sole lifecycle owner and releases the
provider lease only after the parent run finishes. This also covers siblings
that lazily acquire the sandbox before the lead has used a file tool.

## Shared workspace

Forks reuse the parent sandbox/workspace and **may read and write files**. Tool schemas stay on the model (prefix cache). `ForkExecutionGuardMiddleware` only denies nested agents and user interrupts when `context.is_fork` is set:

- blocked: `fork_task`, `task`, `ask_clarification`, `setup_agent`, `update_agent`, `present_files`
- allowed: read/search/web, `write_file`, `str_replace`, and bash

Same-path mutations serialize through the sandbox file lock and read-before-write gate. Sibling forks should still prefer different output paths.

## Runtime

- Tool: `fork_task(prompt)`
- Enable: `configurable.fork_enabled` (defaults to `subagent_enabled`)
- Host graph: `ForkHostMiddleware` attaches the compiled lead graph and republishes it in `wrap_tool_call` (tool-node ContextVar), not only `before_agent`
- SDK factory customization: `RuntimeFeatures(fork=<middleware>)` adds the custom policy middleware while retaining the mandatory `ForkHostMiddleware` and `ForkExecutionGuardMiddleware`
- Concurrency: counted together with `task` by `SubagentLimitMiddleware`
- Recursion: conversational cap `DEFAULT_FORK_MAX_TURNS = 25`, mapped to a higher LangGraph `recursion_limit` because that counter includes middleware nodes. A usable partial answer is kept if the cap still trips.
- Persistence: ephemeral copy of the lead graph with `checkpointer=False`
- Isolation: child `RunnableConfig` omits `thread_id` / checkpoint coordinates (same stream-leak contract as subagents)
- Side effects: memory and title middlewares no-op when `is_fork` is true

## Token usage in the UI

Each `fork_task` card shows that branch's tokens, prompt-cache hits, and cache-hit rate in the expanded footer badge.

- After the branch finishes, collector records are reported to the parent `RunJournal`, so the **header thread total includes parallel-task tokens**
- Lead `usage_metadata` (per-turn / debug) still excludes `fork_task` — `TokenUsageMiddleware` does not merge those ToolMessages into the dispatching AIMessage
- Prompt-cache hits are included on the card when the provider reports `cache_read`
