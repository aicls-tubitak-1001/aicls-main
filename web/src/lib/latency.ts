/**
 * Client-side latency telemetry (ADR-0014, api.md §3.10).
 *
 * The contractual metric is "command entered → result **displayed**", averaged. A server stopwatch
 * cannot see the render, so the browser marks two instants with `performance.now()` — submit and
 * first paint of the result — and posts the pair to `POST /telemetry/latency`, where it is joined
 * with the `Server-Timing` stages into `latency_samples`.
 *
 * Marks live in a bounded map keyed by operation id: an operation that never renders (navigation,
 * error) leaves at most one stale entry, and the oldest entries are evicted first.
 */
import { api } from "@/api/client";
import type { LatencySampleIn, Modality } from "@/api/types";

/** One completed measurement, ready to be posted. */
export type LatencySample = LatencySampleIn;

interface PendingMark {
  submitNow: number;
  submitEpoch: number;
  modality: Modality;
}

/** How many un-rendered operations to remember before evicting the oldest. */
const MAX_PENDING = 32;

const pending = new Map<string, PendingMark>();

function now(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

function epochMs(monotonic: number): number {
  if (typeof performance === "undefined" || typeof performance.timeOrigin !== "number") {
    return Date.now();
  }
  return performance.timeOrigin + monotonic;
}

/**
 * Mark the moment a command was submitted.
 *
 * @param opId - Operation id (the ULID returned by the API, or a temporary client id).
 * @param modality - How the command was entered; defaults to `"text"`.
 */
export function markSubmit(opId: string, modality: Modality = "text"): void {
  if (pending.size >= MAX_PENDING) {
    const oldest = pending.keys().next();
    if (!oldest.done) {
      pending.delete(oldest.value);
    }
  }
  const monotonic = now();
  pending.set(opId, { submitNow: monotonic, submitEpoch: epochMs(monotonic), modality });
}

/**
 * Mark the moment the result became visible and build the sample.
 *
 * @param opId - The id passed to {@link markSubmit}.
 * @returns The completed sample, or `null` when no submit mark exists (nothing to measure).
 */
export function markRendered(opId: string): LatencySample | null {
  const mark = pending.get(opId);
  if (mark === undefined) {
    return null;
  }
  pending.delete(opId);
  const monotonic = now();
  return {
    operation_id: opId,
    modality: mark.modality,
    client_submit_ts: Math.round(mark.submitEpoch),
    client_render_ts: Math.round(epochMs(monotonic)),
    render_ms: Math.round((monotonic - mark.submitNow) * 1000) / 1000,
  };
}

/**
 * Post a sample to the telemetry endpoint.
 *
 * Telemetry must never break the UI: transport failures are swallowed (the sample is lost, the
 * result stays on screen).
 */
export async function reportLatency(sample: LatencySample): Promise<void> {
  try {
    await api.POST("/telemetry/latency", { body: sample });
  } catch {
    // Ignored on purpose — see the docstring.
  }
}

/** Mark the render and post the sample in one call (the common case in `ResultCard`). */
export async function markRenderedAndReport(opId: string): Promise<LatencySample | null> {
  const sample = markRendered(opId);
  if (sample !== null) {
    await reportLatency(sample);
  }
  return sample;
}

/** Drop every outstanding mark (used by tests and on logout). */
export function resetLatencyMarks(): void {
  pending.clear();
}
