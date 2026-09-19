/**
 * Closing half of the contractual latency measurement (ADR-0014, api.md §3.10).
 *
 * The metric the proposal signs up to is "command entered → result **displayed**, average < 3 s".
 * A server stopwatch cannot see the display, so the submit instant is marked when the request
 * leaves (`submitTextCommand`, `useVoiceCommand`) under a provisional key, and this hook closes the
 * pair in the effect that runs right after the answer is committed to the DOM.
 *
 * Two details make the sample honest:
 *
 * 1. The mark is keyed by the *provisional* id, because the server's ULID does not exist yet when
 *    the clock starts; the posted sample carries the real `operation_id` so the backend can join it
 *    with its own stage timings (a client-made id would be rejected as unknown).
 * 2. `markRendered` consumes the mark, so a re-render — or React 18's double-invoked effects in
 *    development — can never report the same command twice.
 *
 * Telemetry never interferes with the lecture: a failed post is swallowed by `reportLatency`.
 */
import { useEffect } from "react";

import { markRendered, reportLatency } from "@/lib/latency";

/**
 * Report "this answer is now on screen".
 *
 * @param key - The provisional key the submit was marked under; `null` disables the measurement.
 * @param operationId - The server's operation id, sent instead of the provisional key.
 * @param enabled - Set to `false` while the answer is not actually displayed yet.
 */
export function useRenderedLatency(
  key: string | null | undefined,
  operationId: string | null | undefined,
  enabled = true,
): void {
  useEffect(() => {
    if (!enabled) {
      return;
    }
    const mark = key ?? operationId ?? null;
    if (mark === null || mark === "") {
      return;
    }
    const sample = markRendered(mark);
    if (sample === null) {
      return;
    }
    void reportLatency({ ...sample, operation_id: operationId ?? null });
  }, [key, operationId, enabled]);
}
