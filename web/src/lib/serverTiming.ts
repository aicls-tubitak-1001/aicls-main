/**
 * `Server-Timing` parsing — the honest half of the latency story (ADR-0014).
 *
 * `aicls.api.middleware.RequestContextMiddleware` stamps every response, error documents included,
 * with the header its {@link https://www.rfc-editor.org/rfc/rfc9211 RFC 9211} form:
 *
 * ```text
 * Server-Timing: nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4, app;dur=58.2
 * ```
 *
 * The names are the stages `StageTimer` recorded (`ocr`, `asr`, `nlu`, `parse`, `cas`, `render`,
 * `book`, `decode`) plus the trailing `app`, which is the whole request. The lecturer is shown
 * this breakdown rather than a spinner, so a slow answer can be explained on the spot instead of
 * looking like a hang.
 */

/** One measured stage. */
export interface StageTiming {
  /** Stage name as the server spelled it (`nlu`, `parse`, `cas`, `render`, `app`, …). */
  readonly name: string;
  /** Duration in milliseconds. */
  readonly ms: number;
  /** Optional `desc` parameter of the metric. */
  readonly description?: string;
}

/** Header the server writes the metrics to. */
export const SERVER_TIMING_HEADER = "Server-Timing";

/** Name of the metric that carries the whole request duration (`StageTimer.server_timing_header`). */
export const TOTAL_STAGE = "app";

/**
 * Display order of the pipeline stages.
 *
 * It follows the contractual pipeline (handwriting/voice → understanding → parsing → computation →
 * drawing), so the breakdown reads left to right the way the lecture does. Unknown stages keep
 * their server-side order and are appended.
 */
export const STAGE_ORDER: readonly string[] = [
  "ocr",
  "asr",
  "decode",
  "nlu",
  "parse",
  "cas",
  "book",
  "render",
  TOTAL_STAGE,
];

const METRIC_SEPARATOR = /\s*,\s*/;
const PARAM_SEPARATOR = /\s*;\s*/;

function parseDuration(raw: string): number | null {
  const value = Number.parseFloat(raw);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

function unquote(raw: string): string {
  if (raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"')) {
    return raw.slice(1, -1);
  }
  return raw;
}

/**
 * Parse a `Server-Timing` header value.
 *
 * Metrics without a `dur` parameter are dropped (they carry no latency information); malformed
 * entries are skipped rather than throwing, because a broken header must never cost the lecturer
 * their result.
 *
 * @param header - Raw header value, or `null` when the response carried none.
 * @returns The metrics in the order the server wrote them.
 */
export function parseServerTiming(header: string | null | undefined): StageTiming[] {
  if (header === null || header === undefined || header.trim() === "") {
    return [];
  }
  const stages: StageTiming[] = [];
  for (const metric of header.split(METRIC_SEPARATOR)) {
    const parts = metric.split(PARAM_SEPARATOR).filter((part) => part !== "");
    const name = parts[0]?.trim();
    if (name === undefined || name === "") {
      continue;
    }
    let ms: number | null = null;
    let description: string | undefined;
    for (const part of parts.slice(1)) {
      const equals = part.indexOf("=");
      if (equals === -1) {
        continue;
      }
      const key = part.slice(0, equals).trim().toLowerCase();
      const value = part.slice(equals + 1).trim();
      if (key === "dur") {
        ms = parseDuration(value);
      } else if (key === "desc") {
        description = unquote(value);
      }
    }
    if (ms === null) {
      continue;
    }
    stages.push(description === undefined ? { name, ms } : { name, ms, description });
  }
  return stages;
}

/** Read and parse the `Server-Timing` header of a response. */
export function serverTimingOf(response: Response | null | undefined): StageTiming[] {
  return parseServerTiming(response?.headers.get(SERVER_TIMING_HEADER));
}

/**
 * Turn the JSON `timings` object of a response body into the same list.
 *
 * `TimingsOut` allows unknown stages (`additionalProperties: true`), so anything numeric is kept
 * and ordered by {@link STAGE_ORDER}. `total` is renamed to {@link TOTAL_STAGE} so a body-derived
 * breakdown and a header-derived one are directly comparable.
 */
export function timingsToStages(
  timings: Readonly<Record<string, unknown>> | undefined,
): StageTiming[] {
  if (timings === undefined) {
    return [];
  }
  const stages: StageTiming[] = [];
  for (const [key, value] of Object.entries(timings)) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      continue;
    }
    stages.push({ name: key === "total" ? TOTAL_STAGE : key, ms: value });
  }
  return sortStages(stages);
}

/** Sort stages by {@link STAGE_ORDER}, keeping unknown names in their original relative order. */
export function sortStages(stages: readonly StageTiming[]): StageTiming[] {
  const rank = (name: string): number => {
    const index = STAGE_ORDER.indexOf(name);
    return index === -1 ? STAGE_ORDER.length : index;
  };
  return [...stages]
    .map((stage, index) => ({ stage, index }))
    .sort((a, b) => rank(a.stage.name) - rank(b.stage.name) || a.index - b.index)
    .map((entry) => entry.stage);
}

/**
 * Merge a body breakdown with a header breakdown.
 *
 * The header is authoritative when both carry a stage (it is measured one layer further out), but
 * the body may carry stages the header does not, and vice versa.
 */
export function mergeStages(
  primary: readonly StageTiming[],
  secondary: readonly StageTiming[],
): StageTiming[] {
  const merged = new Map<string, StageTiming>();
  for (const stage of secondary) {
    merged.set(stage.name, stage);
  }
  for (const stage of primary) {
    merged.set(stage.name, stage);
  }
  return sortStages([...merged.values()]);
}

/** Total duration: the `app` metric when present, otherwise the sum of the other stages. */
export function totalMs(stages: readonly StageTiming[]): number {
  const total = stages.find((stage) => stage.name === TOTAL_STAGE);
  if (total !== undefined) {
    return total.ms;
  }
  return stages.reduce((sum, stage) => sum + stage.ms, 0);
}

/**
 * Format a duration for the screen.
 *
 * Sub-millisecond stages keep one decimal so a `0.4 ms` render does not read as `0 ms`; anything
 * larger is rounded, because at four metres a decimal point is noise.
 */
export function formatMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) {
    return "0";
  }
  return ms < 10 ? ms.toFixed(1) : String(Math.round(ms));
}
