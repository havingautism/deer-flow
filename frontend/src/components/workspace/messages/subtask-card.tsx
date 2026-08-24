import {
  CheckCircleIcon,
  ChevronUp,
  ClipboardListIcon,
  CoinsIcon,
  Loader2Icon,
  SparklesIcon,
  WrenchIcon,
  XCircleIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";
import {
  formatCacheHitRate,
  formatTokenCount,
  uniqueTokenCount,
} from "@/core/messages/usage";
import { useModels } from "@/core/models/hooks";
import {
  streamdownPluginsWithoutRawHtml,
  streamdownWordAnimation,
} from "@/core/streamdown";
import {
  SafeStreamdown,
  toStreamdownComponents,
} from "@/core/streamdown/components";
import { fetchSubtaskSteps } from "@/core/tasks/api";
import { useSubtask, useUpdateSubtask } from "@/core/tasks/context";
import { stepsForDisplay } from "@/core/tasks/steps";
import { cn } from "@/lib/utils";

import { CitationLink } from "../citations/citation-link";

import { MarkdownContent } from "./markdown-content";

function StatusGlyph({
  status,
  className,
}: {
  status: "completed" | "failed" | "in_progress" | "pending";
  className?: string;
}) {
  const iconClass = cn("text-muted-foreground size-3.5", className);
  if (status === "completed") {
    return <CheckCircleIcon className={iconClass} strokeWidth={1.75} />;
  }
  if (status === "failed") {
    return <XCircleIcon className={iconClass} strokeWidth={1.75} />;
  }
  if (status === "in_progress") {
    return <Loader2Icon className={cn(iconClass, "animate-spin")} />;
  }
  return null;
}

export function SubtaskCard({
  className,
  taskId,
  threadId,
  runId,
  isLoading,
}: {
  className?: string;
  taskId: string;
  threadId?: string;
  runId?: string;
  isLoading: boolean;
}) {
  const { t } = useI18n();
  const [collapsed, setCollapsed] = useState(true);
  const task = useSubtask(taskId, runId)!;
  const { tokenUsageEnabled } = useModels();
  const updateSubtask = useUpdateSubtask();
  const isFork = task.subagent_type === "fork";
  const statusLabel = isFork
    ? {
        in_progress: t.subtasks.fork_in_progress,
        completed: t.subtasks.fork_completed,
        failed: t.subtasks.fork_failed,
      }[task.status]
    : t.subtasks[task.status];
  const cacheLabel =
    tokenUsageEnabled && task.usage?.cacheReadTokens
      ? `${formatTokenCount(task.usage.cacheReadTokens)} ${t.tokenUsage.cache}`
      : undefined;
  const uniqueLabel =
    tokenUsageEnabled && task.usage?.cacheReadTokens
      ? `${formatTokenCount(uniqueTokenCount(task.usage))} ${t.tokenUsage.unique}`
      : undefined;
  const cacheRate =
    tokenUsageEnabled && task.usage
      ? formatCacheHitRate(task.usage)
      : undefined;
  const tokenTotal =
    tokenUsageEnabled && task.usage
      ? formatTokenCount(task.usage.totalTokens)
      : undefined;

  // The card shows the subagent's step timeline (#3779): its reasoning turns
  // (AI text) interleaved with the tools it ran (by name). See stepsForDisplay
  // for what is kept/dropped.
  const displaySteps = stepsForDisplay(task.steps, task.status);

  // Backfill step history on expand for historical runs (#3779). Live runs
  // already have steps from SSE, so the `steps.length` guard skips the fetch.
  const stepsCount = task.steps?.length ?? 0;
  const backfilledRef = useRef(false);
  useEffect(() => {
    if (collapsed || backfilledRef.current || stepsCount > 0) {
      return;
    }
    if (!threadId || !runId) {
      return;
    }
    backfilledRef.current = true;
    fetchSubtaskSteps(threadId, runId, taskId)
      .then((steps) => {
        if (steps.length > 0) {
          updateSubtask({ id: taskId, runId, steps });
        }
      })
      .catch(() => {
        // Allow a retry on the next expand if the fetch failed.
        backfilledRef.current = false;
      });
  }, [collapsed, stepsCount, threadId, runId, taskId, updateSubtask]);
  const statusIcon = useMemo(
    () => <StatusGlyph status={task.status} />,
    [task.status],
  );
  return (
    <ChainOfThought
      className={cn(
        "relative w-full min-w-0 overflow-hidden gap-2 rounded-lg border py-0",
        className,
      )}
      open={!collapsed}
    >
      <div className="bg-background/95 flex w-full min-w-0 flex-col rounded-lg">
        <div className="flex w-full min-w-0 items-center justify-between p-0.5">
          <Button
            className="w-full min-w-0 items-start justify-start overflow-hidden text-left"
            variant="ghost"
            onClick={() => setCollapsed(!collapsed)}
          >
            <div className="flex w-full min-w-0 items-center justify-between gap-2">
              <ChainOfThoughtStep
                className="min-w-0 flex-1 overflow-hidden font-normal"
                label={
                  <span className="block truncate" title={task.description}>
                    {task.status === "in_progress" ? (
                      <Shimmer className="max-w-full" duration={3} spread={3}>
                        {task.description}
                      </Shimmer>
                    ) : (
                      task.description
                    )}
                  </span>
                }
                icon={<ClipboardListIcon />}
              ></ChainOfThoughtStep>
              <div className="text-muted-foreground flex shrink-0 items-center gap-1.5">
                <span className="text-xs font-normal">{statusLabel}</span>
                {statusIcon}
                <ChevronUp
                  className={cn(
                    "size-4",
                    !collapsed ? "" : "rotate-180",
                  )}
                />
              </div>
            </div>
          </Button>
        </div>
        <ChainOfThoughtContent className="px-4 pb-3">
          {task.prompt && (
            <ChainOfThoughtStep
              className="min-w-0"
              label={
                <div className="min-w-0 break-words [overflow-wrap:anywhere]">
                  <SafeStreamdown
                    {...streamdownPluginsWithoutRawHtml}
                    animated={streamdownWordAnimation}
                    components={toStreamdownComponents({ a: CitationLink })}
                    isAnimating={isLoading}
                  >
                    {task.prompt}
                  </SafeStreamdown>
                </div>
              }
            ></ChainOfThoughtStep>
          )}
          {displaySteps.map((step, i) => {
            const isLastWhileRunning =
              task.status === "in_progress" && i === displaySteps.length - 1;
            const icon = isLastWhileRunning ? (
              <Loader2Icon className="text-muted-foreground size-4 animate-spin" />
            ) : step.kind === "tool" ? (
              <WrenchIcon className="text-muted-foreground size-4" />
            ) : (
              <SparklesIcon className="text-muted-foreground size-4" />
            );
            return (
              <ChainOfThoughtStep
                key={`${step.message_index}-${i}`}
                label={
                  step.kind === "tool" ? (
                    (step.tool_name ?? statusLabel)
                  ) : (
                    <div className="text-muted-foreground line-clamp-3 text-sm">
                      <MarkdownContent content={step.text} isLoading={false} />
                    </div>
                  )
                }
                icon={icon}
              />
            );
          })}
          {task.status === "completed" && (
            <>
              <ChainOfThoughtStep
                label={
                  isFork
                    ? t.subtasks.fork_completed
                    : t.subtasks.completed
                }
                icon={<CheckCircleIcon className="text-muted-foreground size-4" />}
              ></ChainOfThoughtStep>
              <ChainOfThoughtStep
                label={
                  task.result ? (
                    <MarkdownContent content={task.result} isLoading={false} />
                  ) : null
                }
              ></ChainOfThoughtStep>
            </>
          )}
          {task.status === "failed" && (
            <ChainOfThoughtStep
              label={
                <div className="text-muted-foreground">{task.error}</div>
              }
              icon={<XCircleIcon className="text-muted-foreground size-4" />}
            ></ChainOfThoughtStep>
          )}
          {tokenUsageEnabled && tokenTotal && (
            <div className="mt-3 flex justify-end">
              <div
                className="text-muted-foreground bg-background/70 flex h-auto items-center gap-1.5 rounded-full border px-2 py-1 text-xs font-normal"
                title={[
                  `${t.tokenUsage.label} ${tokenTotal}`,
                  uniqueLabel,
                  cacheLabel,
                  cacheRate
                    ? `${t.tokenUsage.cacheRate} ${cacheRate}`
                    : undefined,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              >
                <CoinsIcon size={14} />
                <span>{t.tokenUsage.label}</span>
                <span className="font-mono">{tokenTotal}</span>
                {uniqueLabel && (
                  <span className="text-muted-foreground/80 border-l pl-1.5 font-mono">
                    {uniqueLabel}
                  </span>
                )}
                {cacheLabel && (
                  <span className="text-muted-foreground/80 border-l pl-1.5 font-mono">
                    {cacheLabel}
                  </span>
                )}
                {cacheRate && (
                  <span className="text-muted-foreground/80 border-l pl-1.5 font-mono">
                    {cacheRate}
                  </span>
                )}
              </div>
            </div>
          )}
        </ChainOfThoughtContent>
      </div>
    </ChainOfThought>
  );
}
