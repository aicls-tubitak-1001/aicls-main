/**
 * One function for "send this Turkish sentence to the pipeline".
 *
 * The workspace controller, the command bar's own fallback path, the clarification candidates and
 * the history's "yeniden çalıştır" all go through here, so the request body, the latency mark and
 * the error translation are written exactly once (api.md §3.4, ADR-0014).
 *
 * The latency key is opened *before* the request leaves and returned to the caller; the panel that
 * finally renders the answer closes the measurement, because the contractual metric ends at the
 * displayed result, not at the response.
 */
import { api, call, type AiclsProblem } from "@/api/client";
import type { ComputeOut, ComputeTextIn, Modality } from "@/api/types";
import { t } from "@/i18n";
import { markSubmit } from "@/lib/latency";
import type { StageTiming } from "@/lib/serverTiming";

/** Everything one submission produced. */
export interface SubmitResult {
  /** The backend's answer, or `null` when the call failed. */
  readonly outcome: ComputeOut | null;
  /** The Turkish failure, or `null` on success. */
  readonly problem: AiclsProblem | null;
  /** `Server-Timing` stages of the request. */
  readonly stages: StageTiming[];
  /** Provisional key the submit instant was marked under. */
  readonly latencyKey: string;
}

/** Call-site options. */
export interface SubmitOptions {
  /** Workspace to record the operation in. */
  workspaceId?: string | null;
  /** Expression on the board, sent as `context.last_latex`. */
  lastLatex?: string | null;
  /** Variable the session is working with, sent as `context.active_var`. */
  activeVar?: string | null;
  /** Intent chosen from a clarification (`confirm_intent`). */
  confirmIntent?: string;
  /** How the command was entered; only affects the telemetry modality. */
  modality?: Modality;
}

let counter = 0;

/** A provisional operation key: the server has not named the operation yet. */
export function nextLatencyKey(prefix = "pending"): string {
  counter += 1;
  return `${prefix}-${String(counter)}-${String(Date.now())}`;
}

/**
 * Send a Turkish command to `POST /compute/text`.
 *
 * @param text - The sentence as the lecturer wrote or spoke it (trimmed by the caller).
 * @param options - Workspace, board context and the confirmation echo.
 * @returns The outcome or a renderable problem, plus the timings and the latency key.
 */
export async function submitTextCommand(
  text: string,
  options: SubmitOptions = {},
): Promise<SubmitResult> {
  const { workspaceId = null, lastLatex = null, activeVar = null, confirmIntent } = options;
  const latencyKey = nextLatencyKey();
  markSubmit(latencyKey, options.modality ?? "text");

  const context = {
    ...(lastLatex === null || lastLatex === "" ? {} : { last_latex: lastLatex }),
    ...(activeVar === null || activeVar === "" ? {} : { active_var: activeVar }),
  };
  const body: ComputeTextIn = {
    text,
    ...(workspaceId === null || workspaceId === "" ? {} : { workspace_id: workspaceId }),
    ...(Object.keys(context).length === 0 ? {} : { context }),
    ...(confirmIntent === undefined ? {} : { confirm_intent: confirmIntent }),
  };

  const { data, problem, stages } = await call(api.POST("/compute/text", { body }), {
    fallbackDetailTr: t("errors.network"),
  });
  return { outcome: data, problem, stages, latencyKey };
}
