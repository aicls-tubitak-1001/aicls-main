/**
 * RFC 9457 `application/problem+json` parsing (docs/mimari/api.md §2).
 *
 * Failure is a normal state in a lecture, not a crash: every error the UI shows must be a calm
 * Turkish sentence, and the backend already ships one in `detail_tr` for every problem it raises.
 * This module turns the raw JSON body into a typed {@link AiclsProblem} with `detail_tr`
 * guaranteed to be non-empty, keeps the structured extras (`position` for a parse error,
 * `construct` for an unsupported construct, `retry_after` for a rate limit) and remembers the
 * `X-Request-Id` of the response so a lecturer can quote it in a bug report.
 *
 * The problem document is **not** part of the OpenAPI schema (FastAPI documents only the success
 * models), so the shape here is transcribed from `src/aicls/api/schemas/common.py::ProblemOut` and
 * `src/aicls/core/errors.py::AiclsError.to_problem`. The code list is transcribed from
 * `aicls.core.errors.ErrorCode`; nothing is invented.
 */

import { t } from "@/i18n";

/** Header carrying the correlation id of a request (`aicls.api.errors.REQUEST_ID_HEADER`). */
export const REQUEST_ID_HEADER = "X-Request-Id";

/** Media type of every error response (`aicls.api.errors.PROBLEM_MEDIA_TYPE`). */
export const PROBLEM_MEDIA_TYPE = "application/problem+json";

/**
 * The stable machine-readable codes of `aicls.core.errors.ErrorCode`.
 *
 * Kept as a runtime array as well as a type so the client can tell a code it understands from one
 * a newer backend introduced (see {@link isKnownErrorCode}) without ever inventing one.
 */
export const ERROR_CODES = [
  "PARSE_ERROR",
  "UNSUPPORTED_CONSTRUCT",
  "UNSUPPORTED_PROBLEM",
  "CAS_TIMEOUT",
  "NLU_NO_INTENT",
  "OCR_INPUT_INVALID",
  "ASR_INPUT_INVALID",
  "UPLOAD_INVALID",
  "CONSENT_REQUIRED",
  "AUTH_INVALID",
  "FORBIDDEN",
  "NOT_FOUND",
  "RATE_LIMITED",
  "VALIDATION_ERROR",
  "GATE_BLOCKED",
  "INTERNAL",
] as const;

/** One of the backend's error codes. */
export type ErrorCode = (typeof ERROR_CODES)[number];

/**
 * Marker used when the request never reached the API (DNS, offline, aborted fetch).
 *
 * It is a **client-side** value with `status: 0` and is deliberately spelled so it can never be
 * confused with a backend code: the server has no such code and never sends one.
 */
export const TRANSPORT_FAILURE_CODE = "CLIENT_TRANSPORT" as const;

/**
 * Either a backend error code, an unknown code from a newer backend, or the transport marker.
 *
 * The `string & Record<never, never>` arm keeps editor completion for the known codes while still
 * accepting a code this client version has not heard of — an unknown code must render, not throw.
 */
export type ProblemCode =
  ErrorCode | typeof TRANSPORT_FAILURE_CODE | (string & Record<never, never>);

/** One entry of a 422 validation failure (`details.errors`). */
export interface ProblemValidationEntry {
  loc: (string | number)[];
  msg: string;
}

/**
 * A parsed problem document.
 *
 * `detail_tr` is always a usable Turkish sentence: the parser falls back to `title`, then to the
 * caller-supplied default, so a component can render it without a null check.
 */
export interface AiclsProblem {
  /** `https://ai-cls.example/errors/<CODE>`. */
  readonly type: string;
  /** Short Turkish title (`aicls.core.messages_tr.TITLES`). */
  readonly title: string;
  /** HTTP status; `0` for a transport failure that never reached the API. */
  readonly status: number;
  /** Stable machine-readable code. */
  readonly code: ProblemCode;
  /** Turkish user-facing sentence; never empty. */
  readonly detail_tr: string;
  /** Correlation id (`instance`, echoed in `X-Request-Id`). */
  readonly instance: string | null;
  /** Character offset of a parse failure, when the backend located one. */
  readonly position?: number;
  /** Name of the LaTeX construct the grammar refuses. */
  readonly construct?: string;
  /** Whole seconds to wait after a 429. */
  readonly retry_after?: number;
  /** Turkish suggestions offered with `NLU_NO_INTENT`. */
  readonly suggestions_tr?: string[];
  /** Per-field messages of a 422. */
  readonly errors?: ProblemValidationEntry[];
  /** Every other structured field the raising error carried in `details`. */
  readonly extra: Readonly<Record<string, unknown>>;
}

const RESERVED_KEYS = new Set([
  "type",
  "title",
  "status",
  "code",
  "detail_tr",
  "instance",
  "position",
  "construct",
  "retry_after",
  "suggestions_tr",
  "errors",
]);

/** Base of the `type` URI (`aicls.core.errors.ERROR_TYPE_BASE`). */
const ERROR_TYPE_BASE = "https://ai-cls.example/errors";

const KNOWN_CODES: ReadonlySet<string> = new Set<string>(ERROR_CODES);

/** Whether `code` is one of the codes this client version knows about. */
export function isKnownErrorCode(code: string): code is ErrorCode {
  return KNOWN_CODES.has(code);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const items = value.filter((item): item is string => typeof item === "string");
  return items.length === value.length ? items : undefined;
}

function asValidationEntries(value: unknown): ProblemValidationEntry[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const entries: ProblemValidationEntry[] = [];
  for (const item of value) {
    if (!isRecord(item)) {
      continue;
    }
    const loc = Array.isArray(item.loc)
      ? item.loc.filter(
          (part): part is string | number => typeof part === "string" || typeof part === "number",
        )
      : [];
    entries.push({ loc, msg: asString(item.msg) ?? "" });
  }
  return entries.length > 0 ? entries : undefined;
}

/**
 * Structural test for a problem document.
 *
 * A document must carry at least the two fields every `AiclsError` renders: `code` and `status`.
 */
export function isProblemDocument(value: unknown): value is Record<string, unknown> {
  return isRecord(value) && typeof value.code === "string" && typeof value.status === "number";
}

/** Read the correlation id of a response, if the API stamped one. */
export function requestIdOf(response: Response | null | undefined): string | null {
  return response?.headers.get(REQUEST_ID_HEADER) ?? null;
}

/**
 * Parse an error payload into a typed problem.
 *
 * @param value - The body `openapi-fetch` returned in `error` (already JSON-parsed when possible).
 * @param options.response - The response it came from, used for the status and the request id.
 * @param options.fallbackDetailTr - Turkish sentence used when the payload carries none.
 * @returns A problem whose `detail_tr` is always renderable.
 */
export function parseProblem(
  value: unknown,
  options: { response?: Response | null; fallbackDetailTr?: string } = {},
): AiclsProblem {
  const { response = null, fallbackDetailTr } = options;
  const requestId = requestIdOf(response);
  const fallback = fallbackDetailTr ?? t("errors.unknown");

  if (!isProblemDocument(value)) {
    const status = response?.status ?? 0;
    const detail = asString(typeof value === "string" ? value : undefined) ?? fallback;
    return {
      type: `${ERROR_TYPE_BASE}/${TRANSPORT_FAILURE_CODE}`,
      title: fallback,
      status,
      code: status === 0 ? TRANSPORT_FAILURE_CODE : "INTERNAL",
      detail_tr: detail,
      instance: requestId,
      extra: {},
    };
  }

  const extra: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    if (!RESERVED_KEYS.has(key)) {
      extra[key] = item;
    }
  }

  const code = value.code as string;
  const title = asString(value.title) ?? fallback;
  const position = asNumber(value.position);
  const construct = asString(value.construct);
  const retryAfter = asNumber(value.retry_after);
  const suggestions = asStringArray(value.suggestions_tr);
  const errors = asValidationEntries(value.errors);

  return {
    type: asString(value.type) ?? `${ERROR_TYPE_BASE}/${code}`,
    title,
    status: value.status as number,
    code,
    detail_tr: asString(value.detail_tr) ?? title,
    instance: asString(value.instance) ?? requestId,
    ...(position === undefined ? {} : { position }),
    ...(construct === undefined ? {} : { construct }),
    ...(retryAfter === undefined ? {} : { retry_after: retryAfter }),
    ...(suggestions === undefined ? {} : { suggestions_tr: suggestions }),
    ...(errors === undefined ? {} : { errors }),
    extra,
  };
}

/**
 * Build the problem for a request that never reached the API.
 *
 * @param detailTr - The Turkish sentence to show (from `i18n/tr.json`, e.g. `errors.network`).
 */
export function transportProblem(detailTr: string, cause?: unknown): AiclsProblem {
  const reason = cause instanceof Error ? cause.message : undefined;
  return {
    type: `${ERROR_TYPE_BASE}/${TRANSPORT_FAILURE_CODE}`,
    title: detailTr,
    status: 0,
    code: TRANSPORT_FAILURE_CODE,
    detail_tr: detailTr,
    instance: null,
    extra: reason === undefined ? {} : { reason },
  };
}

/** Whether the failure is the client's own transport marker rather than a backend answer. */
export function isTransportProblem(problem: AiclsProblem): boolean {
  return problem.code === TRANSPORT_FAILURE_CODE;
}

/**
 * The Turkish sentence to display for any error value.
 *
 * Kept for call sites that only need the text; anything richer should use {@link parseProblem}.
 */
export function problemMessage(error: unknown, fallback: string): string {
  if (isProblemDocument(error)) {
    return asString(error.detail_tr) ?? asString(error.title) ?? fallback;
  }
  return fallback;
}
