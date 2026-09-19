/**
 * `StageTimings` — where the time went, shown rather than hidden.
 *
 * The contract is "command → rendered result, under 3 s on average", so latency is a first-class
 * number in this interface. Every response carries a per-stage breakdown twice: in the JSON body
 * (`timings`) and in the `Server-Timing` header, both produced by `aicls.core.timing.StageTimer`.
 * This component renders whichever it is given as a labelled bar chart with the numbers written
 * out, and marks the total when it crosses the contractual budget.
 *
 * The bars are decorative (`aria-hidden`); the numbers are a definition list, so a screen reader
 * reads "anlama 12 ms, ayrıştırma 1 ms, …" instead of describing rectangles.
 */
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import {
  TOTAL_STAGE,
  formatMs,
  sortStages,
  timingsToStages,
  totalMs,
  type StageTiming,
} from "@/lib/serverTiming";
import styles from "./StageTimings.module.css";

/** Contractual end-to-end budget in milliseconds (PROJECT_BRIEF, "Response time < 3 s"). */
export const LATENCY_BUDGET_MS = 3000;

/** Turkish label of each pipeline stage. */
const STAGE_LABEL: Record<string, MessageKey> = {
  ocr: "timings.stage.ocr",
  asr: "timings.stage.asr",
  decode: "timings.stage.decode",
  nlu: "timings.stage.nlu",
  parse: "timings.stage.parse",
  cas: "timings.stage.cas",
  book: "timings.stage.book",
  render: "timings.stage.render",
  app: "timings.stage.total",
};

function labelFor(name: string): string {
  const key = STAGE_LABEL[name];
  return key === undefined ? name : t(key);
}

export interface StageTimingsProps {
  /** Stages from the `Server-Timing` header (`serverTimingOf(response)`). */
  stages?: readonly StageTiming[];
  /** …or the `timings` object of the response body; both may be given. */
  timings?: Readonly<Record<string, unknown>>;
  /** Client-measured render time, appended as its own stage when supplied. */
  renderMs?: number;
  /** Hide the bars and show one line of text (for a dense header). */
  compact?: boolean;
  className?: string;
}

/**
 * Render the latency breakdown.
 *
 * @example
 * ```tsx
 * <StageTimings timings={outcome.timings} stages={result.stages} />
 * ```
 */
export function StageTimings({
  stages,
  timings,
  renderMs,
  compact = false,
  className,
}: StageTimingsProps): JSX.Element | null {
  const fromBody = timingsToStages(timings);
  const merged = new Map<string, StageTiming>();
  for (const stage of fromBody) {
    merged.set(stage.name, stage);
  }
  for (const stage of stages ?? []) {
    merged.set(stage.name, stage);
  }
  if (renderMs !== undefined && Number.isFinite(renderMs)) {
    merged.set("render", { name: "render", ms: renderMs });
  }
  const all = sortStages([...merged.values()]);
  if (all.length === 0) {
    return null;
  }

  const total = totalMs(all);
  const parts = all.filter((stage) => stage.name !== TOTAL_STAGE);
  const scale = Math.max(total, ...parts.map((stage) => stage.ms), 1);
  const overBudget = total > LATENCY_BUDGET_MS;

  if (compact) {
    return (
      <p className={cx(styles.compact, overBudget && styles.over, className)}>
        {t("timings.totalMs", { ms: formatMs(total) })}
      </p>
    );
  }

  return (
    <div className={cx(styles.wrapper, className)}>
      <p className={cx(styles.total, overBudget && styles.over)}>
        {t("timings.totalMs", { ms: formatMs(total) })}
        {overBudget ? <span className={styles.budget}> · {t("timings.overBudget")}</span> : null}
      </p>
      <dl className={styles.list}>
        {parts.map((stage) => (
          <div key={stage.name} className={styles.row}>
            <dt className={styles.name}>{labelFor(stage.name)}</dt>
            <dd className={styles.value}>
              <span
                className={styles.bar}
                style={{ width: `${String(Math.min(100, (stage.ms / scale) * 100))}%` }}
                aria-hidden="true"
              />
              <span className={styles.number}>{t("timings.ms", { ms: formatMs(stage.ms) })}</span>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
