/**
 * Rendering the free-form JSON the administration endpoints carry.
 *
 * `ModelVersionOut.metrics` and `EvalRunOut.metrics` are `additionalProperties: true` in the
 * schema — the backend stores whatever a training or evaluation run produced — so the client is
 * handed `unknown` values and must render them without pretending to know their shape. Numbers are
 * rounded to four decimals (enough for a CER or a WER), everything else is shown as it is.
 */
import { t } from "@/i18n";

/** Format one metric value for display. */
export function formatMetricValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : String(Math.round(value * 10_000) / 10_000);
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "boolean") {
    return value ? t("common.yes") : t("common.no");
  }
  if (value === null || value === undefined) {
    return t("common.none");
  }
  return JSON.stringify(value) ?? t("common.unknown");
}

/** One metric, ready to render. */
export interface MetricEntry {
  readonly name: string;
  readonly value: string;
}

/** Turn a metrics object into a stable, sorted list of `name = value` entries. */
export function metricEntries(metrics: Readonly<Record<string, unknown>>): MetricEntry[] {
  return Object.entries(metrics)
    .map(([name, value]) => ({ name, value: formatMetricValue(value) }))
    .sort((a, b) => a.name.localeCompare(b.name, "tr"));
}
