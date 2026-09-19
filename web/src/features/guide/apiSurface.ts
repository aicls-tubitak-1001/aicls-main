/**
 * The API surface, summarised for the technical guide.
 *
 * The counts below are the ones in `web/src/api/generated/openapi.json` (56 paths, 62 operations,
 * 72 schemas). They are written out rather than computed at runtime because importing the whole
 * OpenAPI document into the bundle to count its keys would cost every reader a few hundred
 * kilobytes for a number. The guarantee that they stay true is a unit test instead: it reads the
 * generated document and fails if any count here drifts (`src/test/guide.test.tsx`).
 *
 * The error codes are typed as {@link ErrorCode}, so a code that does not exist in
 * `aicls.core.errors.ErrorCode` cannot be listed here at all — the table is complete by
 * construction, not by proofreading.
 */
import type { ErrorCode } from "@/api/problem";
import type { MessageKey } from "@/i18n";

/** Totals of the committed OpenAPI document. */
export const API_TOTALS = { paths: 56, operations: 62, schemas: 72 } as const;

/** One tag of the REST surface. */
export interface ApiTagRow {
  /** Tag name as it appears in the OpenAPI document. */
  readonly tag: string;
  /** Number of operations carrying that tag. */
  readonly count: number;
  readonly purposeKey: MessageKey;
}

/** Every tag in the document, in the order a reader meets them. */
export const API_TAGS: readonly ApiTagRow[] = [
  { tag: "health", count: 3, purposeKey: "guide.tech.api.tag.health" },
  { tag: "auth", count: 5, purposeKey: "guide.tech.api.tag.auth" },
  { tag: "users", count: 2, purposeKey: "guide.tech.api.tag.users" },
  { tag: "workspaces", count: 8, purposeKey: "guide.tech.api.tag.workspaces" },
  { tag: "compute", count: 4, purposeKey: "guide.tech.api.tag.compute" },
  { tag: "ocr", count: 2, purposeKey: "guide.tech.api.tag.ocr" },
  { tag: "asr", count: 3, purposeKey: "guide.tech.api.tag.asr" },
  { tag: "book", count: 8, purposeKey: "guide.tech.api.tag.book" },
  { tag: "contribute", count: 9, purposeKey: "guide.tech.api.tag.contribute" },
  { tag: "feedback", count: 3, purposeKey: "guide.tech.api.tag.feedback" },
  { tag: "telemetry", count: 1, purposeKey: "guide.tech.api.tag.telemetry" },
  { tag: "admin", count: 14, purposeKey: "guide.tech.api.tag.admin" },
];

/** One row of the RFC 9457 error table. */
export interface ErrorRow {
  readonly code: ErrorCode;
  readonly status: number;
  readonly whenKey: MessageKey;
}

/** The complete error-code table (`docs/mimari/api.md` §2). */
export const ERROR_ROWS: readonly ErrorRow[] = [
  { code: "VALIDATION_ERROR", status: 422, whenKey: "guide.tech.api.err.validation" },
  { code: "PARSE_ERROR", status: 422, whenKey: "guide.tech.api.err.parse" },
  {
    code: "UNSUPPORTED_CONSTRUCT",
    status: 422,
    whenKey: "guide.tech.api.err.unsupportedConstruct",
  },
  { code: "UNSUPPORTED_PROBLEM", status: 422, whenKey: "guide.tech.api.err.unsupportedProblem" },
  { code: "NLU_NO_INTENT", status: 422, whenKey: "guide.tech.api.err.nluNoIntent" },
  { code: "CAS_TIMEOUT", status: 504, whenKey: "guide.tech.api.err.casTimeout" },
  { code: "OCR_INPUT_INVALID", status: 400, whenKey: "guide.tech.api.err.ocrInput" },
  { code: "ASR_INPUT_INVALID", status: 400, whenKey: "guide.tech.api.err.asrInput" },
  { code: "UPLOAD_INVALID", status: 400, whenKey: "guide.tech.api.err.uploadInvalid" },
  { code: "AUTH_INVALID", status: 401, whenKey: "guide.tech.api.err.authInvalid" },
  { code: "FORBIDDEN", status: 403, whenKey: "guide.tech.api.err.forbidden" },
  { code: "CONSENT_REQUIRED", status: 403, whenKey: "guide.tech.api.err.consentRequired" },
  { code: "NOT_FOUND", status: 404, whenKey: "guide.tech.api.err.notFound" },
  { code: "RATE_LIMITED", status: 429, whenKey: "guide.tech.api.err.rateLimited" },
  { code: "GATE_BLOCKED", status: 503, whenKey: "guide.tech.api.err.gateBlocked" },
  { code: "INTERNAL", status: 500, whenKey: "guide.tech.api.err.internal" },
];

/**
 * A problem document, as the API sends one.
 *
 * Copied from `docs/mimari/api.md` §2 so a reader can see the shape their own client will parse.
 */
export const PROBLEM_EXAMPLE = `{
  "type": "https://ai-cls.example/errors/PARSE_ERROR",
  "title": "LaTeX ifadesi ayrıştırılamadı",
  "status": 422,
  "detail_tr": "Konum 8: ifade tamamlanmamış, } bekleniyor",
  "code": "PARSE_ERROR",
  "instance": "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  "position": 8
}`;

/** The `Server-Timing` header the compute endpoints return (api.md §1). */
export const SERVER_TIMING_EXAMPLE =
  "Server-Timing: nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4, app;dur=58.2";

/** The handwriting stream's message vocabulary (api.md §4.1). */
export const INK_PROTOCOL = `istemci → sunucu:  stroke{points:[{x,y,t}], canvas:{w,h}} | pen_up{} | clear{} | finalize{} | ping{}
sunucu → istemci:  partial{latex, confidence} | final{latex, alternatives, recognition_id}
                   | error{problem} | slow_down{retry_after_ms} | pong{}`;

/** The speech stream's message vocabulary (api.md §4.2). */
export const AUDIO_PROTOCOL = `istemci → sunucu:  start{sample_rate:16000, format:"pcm16"} | ikili çerçeve (250 ms PCM16)
                   | stop{} | cancel{}
sunucu → istemci:  partial{text} | final{text, confidence, transcript_id}
                   | command{outcome: ComputeOut} | error{problem} | slow_down{}`;
