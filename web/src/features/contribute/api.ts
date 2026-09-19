/**
 * Every call the "Model Gelişimine Katkıda Bulun" tab makes (api.md §3.8).
 *
 * The screens never touch `openapi-fetch` directly: each function here wraps one endpoint with
 * {@link call}, so the caller always gets `{ data, problem }` and never a thrown error. Two things
 * are handled once, in this module, instead of in every component:
 *
 * * **the participant header** — `X-Participant-Token` identifies an anonymous volunteer's speaker
 *   row; it is attached whenever this tab holds one (see {@link file://./participant.ts});
 * * **the ethics gate** — while `AICLS_ETHICS_APPROVAL_REF` is unset the whole router answers
 *   `503 GATE_BLOCKED` (`aicls.api.routers.contribute.contribution_gate`). {@link isGateBlocked}
 *   recognises that one code so the page can render a calm "henüz açık değil" panel instead of an
 *   error; withdrawal is deliberately outside the gate and keeps working.
 */
import { api, call, filePart, formDataSerializer, type ApiResult } from "@/api/client";
import type { AiclsProblem } from "@/api/problem";
import type { Schema } from "@/api/schema";
import type {
  ConsentTextOut,
  DemographicsIn,
  ProgressOut,
  PromptOut,
  RecordingOut,
  SpeakerOut,
} from "@/api/types";
import { t } from "@/i18n";

/** Header name the backend reads the participant token from (`PARTICIPANT_TOKEN_HEADER`). */
export const PARTICIPANT_TOKEN_HEADER = "X-Participant-Token";

/** How many sentences one `GET /contribute/prompts/next` asks for. */
export const PROMPT_BATCH_SIZE = 5;

/** Accepted recordings that make one comfortable sitting; the volunteer may stop at any time. */
export const SESSION_GOAL = 10;

/** Whether a failure is the ethics gate rather than something the volunteer did. */
export function isGateBlocked(problem: AiclsProblem | null): boolean {
  return problem !== null && problem.code === "GATE_BLOCKED";
}

/** Whether a failure means "this speaker has not consented (yet)". */
export function isConsentRequired(problem: AiclsProblem | null): boolean {
  return problem !== null && problem.code === "CONSENT_REQUIRED";
}

/** Build the optional participant header block for a request. */
function tokenHeader(token: string | null): { [PARTICIPANT_TOKEN_HEADER]: string } | undefined {
  return token === null || token === "" ? undefined : { [PARTICIPANT_TOKEN_HEADER]: token };
}

/** The consent text a volunteer must accept, with its version and hash. */
export async function fetchConsentText(): Promise<ApiResult<ConsentTextOut>> {
  return await call(api.GET("/contribute/consent/current", {}), {
    fallbackDetailTr: t("errors.network"),
  });
}

/**
 * Record an acceptance and receive the pseudonym plus the participant token.
 *
 * @param version - `ConsentTextOut.version` exactly as it was displayed.
 * @param demographics - Band-only demographics; the API rejects any other field (KVKK).
 */
export async function submitConsent(
  version: string,
  demographics: DemographicsIn,
): Promise<ApiResult<SpeakerOut>> {
  return await call(
    api.POST("/contribute/consent", { body: { version, accepted: true, demographics } }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/** The next sentences this participant should read (least recorded, scenario balanced). */
export async function fetchNextPrompts(
  token: string | null,
  n: number = PROMPT_BATCH_SIZE,
): Promise<ApiResult<PromptOut[]>> {
  const header = tokenHeader(token);
  return await call(
    api.GET("/contribute/prompts/next", {
      params: { query: { n }, ...(header === undefined ? {} : { header }) },
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/**
 * Upload one take.
 *
 * The verdict is part of the 200 payload: a rejected recording is **not** an error, it is a
 * `RecordingOut` with `accepted: false` and Turkish `qc.reasons_tr` (api.md §2). Nothing is written
 * to disk on the server for a rejected take, so retrying costs the volunteer nothing.
 */
export async function uploadRecording(
  promptId: string,
  wav: Blob,
  token: string | null,
): Promise<ApiResult<RecordingOut>> {
  const header = tokenHeader(token);
  return await call(
    api.POST("/contribute/recordings", {
      ...(header === undefined ? {} : { params: { header } }),
      body: { audio: filePart(wav, "kayit.wav"), prompt_id: promptId },
      bodySerializer: formDataSerializer,
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/** Soft-delete one of this participant's own recordings (204). */
export async function deleteRecording(
  recordingId: string,
  token: string | null,
): Promise<ApiResult<never>> {
  const header = tokenHeader(token);
  return await call(
    api.DELETE("/contribute/recordings/{recording_id}", {
      params: {
        path: { recording_id: recordingId },
        ...(header === undefined ? {} : { header }),
      },
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/** How much this participant has contributed so far. */
export async function fetchProgress(token: string | null): Promise<ApiResult<ProgressOut>> {
  const header = tokenHeader(token);
  return await call(
    api.GET("/contribute/me/progress", {
      ...(header === undefined ? {} : { params: { header } }),
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/**
 * The collection dashboard.
 *
 * `GET /contribute/stats` is an **instructor/admin** endpoint (`StaffUser`), so the community
 * counter is only fetched for those roles; a volunteer gets no counter rather than a 403 they
 * cannot act on.
 */
export async function fetchContributionStats(): Promise<ApiResult<Schema<"ContribStatsOut">>> {
  return await call(api.GET("/contribute/stats", {}), {
    fallbackDetailTr: t("errors.network"),
  });
}

/**
 * KVKK withdrawal: delete the audio, keep the (pseudonymous) proof of consent.
 *
 * Never gated — a volunteer must be able to withdraw even while collection itself is closed.
 */
export async function withdrawConsent(token: string | null): Promise<ApiResult<never>> {
  const header = tokenHeader(token);
  return await call(
    api.POST("/contribute/withdraw", {
      ...(header === undefined ? {} : { params: { header } }),
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}
