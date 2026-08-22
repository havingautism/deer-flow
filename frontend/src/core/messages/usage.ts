import type { Message } from "@langchain/langgraph-sdk";

export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  /** Prompt-cache hits. Sparse: omitted when the provider reported none. */
  cacheReadTokens?: number;
}

/**
 * Extract usage_metadata from an AI message if present.
 * The field is added by the backend (PR #1218) but not typed in the SDK.
 */
export function getUsageMetadata(message: Message): TokenUsage | null {
  if (message.type !== "ai") {
    return null;
  }
  const usage =
    ((message as Record<string, unknown>).usage_metadata as
      | {
          input_tokens?: number;
          output_tokens?: number;
          total_tokens?: number;
        }
      | undefined) ??
    (message.additional_kwargs?.usage_metadata as
      | {
          input_tokens?: number;
          output_tokens?: number;
          total_tokens?: number;
        }
      | undefined);
  if (!usage) {
    return null;
  }
  return {
    inputTokens: usage.input_tokens ?? 0,
    outputTokens: usage.output_tokens ?? 0,
    totalTokens: usage.total_tokens ?? 0,
  };
}

/**
 * Accumulate token usage across AI messages.
 *
 * UI rendering may place the same AI message in more than one group, such as
 * when a message contains both reasoning and final answer content. Token usage
 * is attached to the AI message itself, so a message id should only contribute
 * once to any aggregate.
 */
export function accumulateUsage(messages: Message[]): TokenUsage | null {
  const cumulative: TokenUsage = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
  };
  let hasUsage = false;
  let cacheReadTokens = 0;
  const countedMessageIds = new Set<string>();

  for (const message of messages) {
    const usage = getUsageMetadata(message);
    if (!usage) {
      continue;
    }

    if (message.id) {
      if (countedMessageIds.has(message.id)) {
        continue;
      }
      countedMessageIds.add(message.id);
    }

    hasUsage = true;
    cumulative.inputTokens += usage.inputTokens;
    cumulative.outputTokens += usage.outputTokens;
    cumulative.totalTokens += usage.totalTokens;
    cacheReadTokens += usage.cacheReadTokens ?? 0;
  }
  if (!hasUsage) {
    return null;
  }
  return cacheReadTokens > 0
    ? { ...cumulative, cacheReadTokens }
    : cumulative;
}

/**
 * Validate a raw `{input,output,total}_tokens` object into {@link TokenUsage}.
 *
 * The single shared validator for both sub-agent usage surfaces — the live
 * `task_running` event (`core/tasks/lifecycle.ts`) and the terminal ToolMessage
 * metadata (`core/tasks/subtask-result.ts`). Keeping one function stops the two
 * from drifting (e.g. one accepting an extra token field the other rejects).
 * The three required keys must be finite, non-negative numbers or the whole
 * snapshot is rejected as `undefined`. `cache_read_tokens` is additive and
 * sparse: a valid positive number is kept, anything else is omitted.
 */
export function normalizeTokenUsage(value: unknown): TokenUsage | undefined {
  if (typeof value !== "object" || value === null) {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  const inputTokens = nonNegativeNumber(record.input_tokens);
  const outputTokens = nonNegativeNumber(record.output_tokens);
  const totalTokens = nonNegativeNumber(record.total_tokens);
  if (
    inputTokens === undefined ||
    outputTokens === undefined ||
    totalTokens === undefined
  ) {
    return undefined;
  }
  const cacheReadTokens = nonNegativeNumber(record.cache_read_tokens);
  return cacheReadTokens && cacheReadTokens > 0
    ? { inputTokens, outputTokens, totalTokens, cacheReadTokens }
    : { inputTokens, outputTokens, totalTokens };
}

function nonNegativeNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

export function hasNonZeroUsage(
  usage: TokenUsage | null | undefined,
): usage is TokenUsage {
  return (
    usage !== null &&
    usage !== undefined &&
    (usage.inputTokens > 0 || usage.outputTokens > 0 || usage.totalTokens > 0)
  );
}

export function addUsage(base: TokenUsage, delta: TokenUsage): TokenUsage {
  const cacheReadTokens =
    (base.cacheReadTokens ?? 0) + (delta.cacheReadTokens ?? 0);
  return {
    inputTokens: base.inputTokens + delta.inputTokens,
    outputTokens: base.outputTokens + delta.outputTokens,
    totalTokens: base.totalTokens + delta.totalTokens,
    ...(cacheReadTokens > 0 ? { cacheReadTokens } : {}),
  };
}

export function selectHeaderTokenUsage({
  backendUsage,
  messages,
  pendingMessages = [],
}: {
  backendUsage?: TokenUsage | null;
  messages: Message[];
  pendingMessages?: Message[];
}): TokenUsage | null {
  if (hasNonZeroUsage(backendUsage)) {
    const pendingUsage = accumulateUsage(pendingMessages);
    return pendingUsage ? addUsage(backendUsage, pendingUsage) : backendUsage;
  }
  return accumulateUsage(messages);
}

/**
 * Format a token count for display: 1234 -> "1,234", 12345 -> "12.3K"
 */
export function formatTokenCount(count: number): string {
  if (count < 10_000) {
    return count.toLocaleString();
  }
  return `${(count / 1000).toFixed(1)}K`;
}
