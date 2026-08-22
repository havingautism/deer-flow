import type { Message } from "@langchain/langgraph-sdk";
import { expect, test } from "@rstest/core";

import {
  accumulateUsage,
  formatCacheHitRate,
  selectHeaderTokenUsage,
  uniqueTokenCount,
} from "@/core/messages/usage";
import {
  getAssistantTurnUsageMessages,
  getMessageGroups,
} from "@/core/messages/utils";

test("accumulates each AI message usage only once by message id", () => {
  const aiMessage = {
    id: "ai-1",
    type: "ai",
    content: "Answer",
    usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
  } as Message;

  expect(accumulateUsage([aiMessage, aiMessage])).toEqual({
    inputTokens: 10,
    outputTokens: 5,
    totalTokens: 15,
  });
});

test("counts later usage-bearing snapshots for the same AI message id", () => {
  const earlySnapshot = {
    id: "ai-1",
    type: "ai",
    content: "Streaming...",
  } as Message;
  const completedSnapshot = {
    id: "ai-1",
    type: "ai",
    content: "Complete answer",
    usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
  } as Message;

  expect(accumulateUsage([earlySnapshot, completedSnapshot])).toEqual({
    inputTokens: 10,
    outputTokens: 5,
    totalTokens: 15,
  });
});

test("reads usage metadata from additional kwargs when the SDK nests it there", () => {
  const aiMessage = {
    id: "ai-1",
    type: "ai",
    content: "Answer",
    additional_kwargs: {
      usage_metadata: {
        input_tokens: 8,
        output_tokens: 3,
        total_tokens: 11,
      },
    },
  } as unknown as Message;

  expect(accumulateUsage([aiMessage])).toEqual({
    inputTokens: 8,
    outputTokens: 3,
    totalTokens: 11,
  });
});

test("keeps header and per-turn aggregation consistent for a reasoning+answer message", () => {
  // A single AI message carrying both reasoning (here via inline <think>) and
  // answer text now lands in exactly one assistant group (#3868), so its usage
  // is counted once both in the per-turn aggregation and against the header
  // total. The by-id dedupe (see "accumulates each AI message usage only once")
  // remains the defence-in-depth guard if any future grouping reintroduces a
  // duplicate.
  const messages = [
    {
      id: "human-1",
      type: "human",
      content: "Explain this",
    },
    {
      id: "ai-1",
      type: "ai",
      content: "<think>checking context</think>Final answer",
      usage_metadata: { input_tokens: 20, output_tokens: 7, total_tokens: 27 },
    },
  ] as Message[];

  const groups = getMessageGroups(messages);
  const usageMessagesByGroupIndex = getAssistantTurnUsageMessages(groups);
  const turnUsageMessages = usageMessagesByGroupIndex.at(-1);

  expect(groups.map((group) => group.type)).toEqual(["human", "assistant"]);
  expect(turnUsageMessages?.map((message) => message.id)).toEqual(["ai-1"]);
  expect(accumulateUsage(messages)).toEqual(
    accumulateUsage(turnUsageMessages!),
  );
  expect(accumulateUsage(turnUsageMessages!)).toEqual({
    inputTokens: 20,
    outputTokens: 7,
    totalTokens: 27,
  });
});

test("prefers backend thread usage for header totals", () => {
  const messages = [
    {
      id: "ai-visible",
      type: "ai",
      content: "Visible answer",
      usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    },
  ] as Message[];

  expect(
    selectHeaderTokenUsage({
      backendUsage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      messages,
    }),
  ).toEqual({
    inputTokens: 100,
    outputTokens: 50,
    totalTokens: 150,
  });
});

test("adds current in-flight message usage to backend header totals", () => {
  const completedMessages = [
    {
      id: "ai-completed",
      type: "ai",
      content: "Completed answer",
      usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    },
    {
      id: "ai-pending",
      type: "ai",
      content: "Streaming answer",
      usage_metadata: { input_tokens: 4, output_tokens: 6, total_tokens: 10 },
    },
  ] as Message[];

  expect(
    selectHeaderTokenUsage({
      backendUsage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      messages: completedMessages,
      pendingMessages: [completedMessages[1]!],
    }),
  ).toEqual({
    inputTokens: 104,
    outputTokens: 56,
    totalTokens: 160,
  });
});

test("falls back to visible messages when backend usage is unavailable or zero", () => {
  const messages = [
    {
      id: "ai-visible",
      type: "ai",
      content: "Visible answer",
      usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    },
  ] as Message[];

  expect(
    selectHeaderTokenUsage({
      backendUsage: null,
      messages,
    }),
  ).toEqual({
    inputTokens: 10,
    outputTokens: 5,
    totalTokens: 15,
  });
  expect(
    selectHeaderTokenUsage({
      backendUsage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      messages,
    }),
  ).toEqual({
    inputTokens: 10,
    outputTokens: 5,
    totalTokens: 15,
  });
});

test("header fallback includes fork_task tool usage when backend totals are missing", () => {
  const messages = [
    {
      id: "ai-1",
      type: "ai",
      content: "forking",
      usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    },
    {
      id: "tool-fork-1",
      type: "tool",
      name: "fork_task",
      tool_call_id: "fork-1",
      content: "Fork Succeeded. Result: A",
      additional_kwargs: {
        subagent_token_usage: {
          input_tokens: 100,
          output_tokens: 10,
          total_tokens: 110,
          cache_read_tokens: 80,
        },
      },
    },
  ] as Message[];

  expect(
    selectHeaderTokenUsage({
      backendUsage: null,
      messages,
    }),
  ).toEqual({
    inputTokens: 110,
    outputTokens: 15,
    totalTokens: 125,
    cacheReadTokens: 80,
  });
});

test("header backend totals are not double-counted with fork tool usage", () => {
  const messages = [
    {
      id: "ai-1",
      type: "ai",
      content: "forking",
      usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
    },
    {
      id: "tool-fork-1",
      type: "tool",
      name: "fork_task",
      tool_call_id: "fork-1",
      content: "Fork Succeeded. Result: A",
      additional_kwargs: {
        subagent_token_usage: {
          input_tokens: 100,
          output_tokens: 10,
          total_tokens: 110,
        },
      },
    },
  ] as Message[];

  expect(
    selectHeaderTokenUsage({
      backendUsage: { inputTokens: 125, outputTokens: 15, totalTokens: 140 },
      messages,
    }),
  ).toEqual({
    inputTokens: 125,
    outputTokens: 15,
    totalTokens: 140,
  });
});

test("formats cache hit rate against input tokens", () => {
  expect(
    formatCacheHitRate({
      inputTokens: 10_000,
      outputTokens: 100,
      totalTokens: 10_100,
      cacheReadTokens: 8_000,
    }),
  ).toBe("80%");
  expect(
    formatCacheHitRate({
      inputTokens: 100,
      outputTokens: 10,
      totalTokens: 110,
      cacheReadTokens: 150,
    }),
  ).toBe("100%");
  expect(
    formatCacheHitRate({
      inputTokens: 10_000,
      outputTokens: 100,
      totalTokens: 10_100,
    }),
  ).toBeUndefined();
});

test("formats unique tokens after subtracting cache hits", () => {
  expect(
    uniqueTokenCount({
      inputTokens: 10_000,
      outputTokens: 100,
      totalTokens: 10_100,
      cacheReadTokens: 8_000,
    }),
  ).toBe(2_100);
});
