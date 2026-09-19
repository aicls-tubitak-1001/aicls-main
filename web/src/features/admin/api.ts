/**
 * Every administration call (api.md §3.11).
 *
 * The surface is split by role exactly as the backend splits it — `StaffUser` for the quality
 * queue, the contribution dashboard and the latency figures; `AdminUser` for the model registry,
 * the evaluation history, the user list and the book loader — so a screen can hide what a role
 * cannot use instead of showing a button that answers 403.
 *
 * One deliberate exception to "everything comes from the generated types": `POST /admin/book/load`
 * reads its archive straight from the request form (`aicls.api.bookload.load_book_archive`) and is
 * therefore documented with **no** request body, which `openapi-typescript` renders as
 * `requestBody?: never`. {@link uploadBookArchive} narrows `api.POST` once, in one place, to send
 * the multipart part the loader actually reads; the response type still comes from the schema and
 * the request still travels through the client's auth and refresh middleware.
 */
import { api, call, type ApiResult } from "@/api/client";
import type { AiclsProblem } from "@/api/problem";
import type { Schema } from "@/api/schema";
import type { ModelVersionOut, Role, UserOut } from "@/api/types";
import { t } from "@/i18n";

/** One recording waiting for a human decision. */
export type QcQueueItem = Schema<"QcQueueItemOut">;

/** The human verdict of `POST /admin/qc/{id}/review`. */
export type QcVerdictValue = Schema<"QcReviewIn">["verdict"];

/** The collection dashboard. */
export type ContribStats = Schema<"ContribStatsOut">;

/** Mean/p50/p95 of the measured end-to-end time. */
export type LatencyStats = Schema<"LatencyStatsOut">;

/** One persisted evaluation report. */
export type EvalRun = Schema<"EvalRunOut">;

/** Result of a book ingest. */
export type LoadReport = Schema<"LoadReportOut">;

/** Which model component a registry row belongs to (`ActivateIn.component`). */
export type ModelComponent = Schema<"ActivateIn">["component"];

/** Review-queue statuses the backend stores (`recordings.review_status`). */
export const REVIEW_STATUSES = ["pending", "accepted", "rejected", "fixed"] as const;

/** One of {@link REVIEW_STATUSES}. */
export type ReviewStatus = (typeof REVIEW_STATUSES)[number];

/* ------------------------------------------------------------------ quality control (staff) --- */

/** Recordings awaiting a human decision, oldest first. */
export async function fetchQcQueue(status: string, limit = 50): Promise<ApiResult<QcQueueItem[]>> {
  return await call(api.GET("/admin/qc/queue", { params: { query: { status, limit } } }), {
    fallbackDetailTr: t("admin.loadFailed"),
  });
}

/** Apply a human quality decision to one recording (audited, 204). */
export async function reviewRecording(
  recordingId: string,
  verdict: QcVerdictValue,
  transcriptFix?: string,
): Promise<ApiResult<never>> {
  return await call(
    api.POST("/admin/qc/{recording_id}/review", {
      params: { path: { recording_id: recordingId } },
      body: {
        verdict,
        ...(transcriptFix === undefined || transcriptFix === ""
          ? {}
          : { transcript_fix: transcriptFix }),
      },
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/** The İP-3 collection dashboard from the administration side. */
export async function fetchContributionStats(): Promise<ApiResult<ContribStats>> {
  return await call(api.GET("/admin/contributions/stats", {}), {
    fallbackDetailTr: t("admin.loadFailed"),
  });
}

/** The contractual "< 3 s" figure, optionally per modality. */
export async function fetchLatencyStats(modality?: string): Promise<ApiResult<LatencyStats>> {
  return await call(
    api.GET("/admin/metrics/latency", {
      ...(modality === undefined || modality === "" ? {} : { params: { query: { modality } } }),
    }),
    { fallbackDetailTr: t("admin.loadFailed") },
  );
}

/* ----------------------------------------------------------------- registry & eval (admin) --- */

/** Every registered checkpoint, newest first. */
export async function fetchModels(): Promise<ApiResult<ModelVersionOut[]>> {
  return await call(api.GET("/admin/models", {}), { fallbackDetailTr: t("admin.loadFailed") });
}

/** Promote one checkpoint and reload the registry (audited, 204). */
export async function activateModel(
  component: ModelComponent,
  version: string,
): Promise<ApiResult<never>> {
  return await call(api.POST("/admin/models/activate", { body: { component, version } }), {
    fallbackDetailTr: t("errors.network"),
  });
}

/** Persisted evaluation reports, newest first. */
export async function fetchEvalRuns(limit = 50): Promise<ApiResult<EvalRun[]>> {
  return await call(api.GET("/admin/eval-runs", { params: { query: { limit } } }), {
    fallbackDetailTr: t("admin.loadFailed"),
  });
}

/* --------------------------------------------------------------------- users (admin only) --- */

/** One page of accounts, newest first. */
export async function fetchUsers(
  cursor?: string,
  limit = 50,
): Promise<ApiResult<{ items: UserOut[]; next_cursor?: string | null }>> {
  return await call(
    api.GET("/admin/users", {
      params: { query: { limit, ...(cursor === undefined ? {} : { cursor }) } },
    }),
    { fallbackDetailTr: t("admin.loadFailed") },
  );
}

/** Change a user's role (audited, 204). */
export async function setUserRole(userId: string, role: Role): Promise<ApiResult<never>> {
  return await call(
    api.PATCH("/admin/users/{user_id}/role", {
      params: { path: { user_id: userId } },
      body: { role },
    }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/* ------------------------------------------------------------------ book loading (admin) --- */

/** What `api.POST` resolves to for the book loader (the schema's own response model). */
interface BookLoadOutcome {
  data?: LoadReport | undefined;
  error?: unknown;
  response: Response;
}

/** The multipart init the loader needs; `file` is the part `load_book_archive` reads. */
interface BookLoadInit {
  body: FormData;
  bodySerializer: (body: FormData) => FormData;
}

/**
 * Upload a zipped Markdown book directory.
 *
 * The backend refuses the whole batch when tagging coverage is below 100 % (İP-8) and lists the
 * offending headings in the problem's `details`; see {@link untaggedHeadingsOf}.
 */
export async function uploadBookArchive(archive: File): Promise<ApiResult<LoadReport>> {
  const form = new FormData();
  form.append("file", archive);
  const post = api.POST as unknown as (
    path: "/admin/book/load",
    init: BookLoadInit,
  ) => Promise<BookLoadOutcome>;
  return await call(
    post("/admin/book/load", { body: form, bodySerializer: (body: FormData) => body }),
    { fallbackDetailTr: t("errors.network") },
  );
}

/**
 * The untagged headings a refused book load reported.
 *
 * `aicls.book.ingest.validators.UntaggedHeadingError` puts `untagged_headings` (the heading texts)
 * into the problem document; anything else is treated as "no list available".
 */
export function untaggedHeadingsOf(problem: AiclsProblem | null): string[] {
  const raw = problem?.extra.untagged_headings;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.filter((item): item is string => typeof item === "string");
}

/** The tagging ratio a refused book load reported, or `null` when it carried none. */
export function taggedRatioOf(problem: AiclsProblem | null): number | null {
  const raw = problem?.extra.tagged_ratio;
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}
