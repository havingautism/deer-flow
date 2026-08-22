import {
  formatCacheHitRate,
  formatTokenCount,
  type TokenUsage,
} from "@/core/messages/usage";
import type { Model } from "@/core/models/types";

/** Return the user-facing label for a configured subagent model. */
export function resolveSubtaskModelLabel(
  modelName: string | undefined,
  models: Model[],
): string | undefined {
  if (!modelName) {
    return undefined;
  }
  return (
    models.find((model) => model.name === modelName)?.display_name ?? modelName
  );
}

export function formatSubtaskTokenUsage(
  usage: TokenUsage | undefined,
  cacheLabel?: string,
): string | undefined {
  if (!usage) {
    return undefined;
  }
  const total = formatTokenCount(usage.totalTokens);
  if (!usage.cacheReadTokens || usage.cacheReadTokens <= 0 || !cacheLabel) {
    return total;
  }
  const parts = [
    total,
    `${formatTokenCount(usage.cacheReadTokens)} ${cacheLabel}`,
  ];
  const cacheRate = formatCacheHitRate(usage);
  if (cacheRate) {
    parts.push(cacheRate);
  }
  return parts.join(" · ");
}
