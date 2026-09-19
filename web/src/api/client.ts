/**
 * The typed REST client (`openapi-fetch`) and everything that happens around a request.
 *
 * `baseUrl` is `/api/v1` on the document's own origin, so the browser always talks to itself: Vite
 * proxies both prefixes in development, Caddy proxies them in production (ADR-0018). Cookies are
 * sent (`credentials: "include"`) because the refresh token lives in the httpOnly `aicls_refresh`
 * cookie and is never visible to JavaScript (api.md §1).
 *
 * Three behaviours are layered on top of the raw client:
 *
 * 1. **Bearer + correlation** — the in-memory access token is attached to every request and the
 *    `X-Request-Id` of every response is remembered, so a lecturer can quote one id for support.
 * 2. **Silent re-authentication** — a 401 triggers a single `POST /auth/refresh` (shared by all
 *    concurrent requests), then the original request is replayed exactly once. If the refresh
 *    fails the session is dropped. A lecture must not stop because a 15-minute token expired.
 * 3. **Problem details** — {@link call} converts every failure into a typed
 *    {@link AiclsProblem} carrying a Turkish `detail_tr`, the request id, and the `Server-Timing`
 *    stage breakdown, so error and success are rendered from the same information.
 */
import createClient from "openapi-fetch";

import type { AiclsProblem } from "./problem";
import {
  isProblemDocument,
  parseProblem,
  problemMessage,
  requestIdOf,
  transportProblem,
} from "./problem";
import { t } from "@/i18n";
import { API_BASE_PATH, type paths } from "./schema";
import { serverTimingOf, type StageTiming } from "@/lib/serverTiming";

export { API_BASE_PATH };
export type { AiclsProblem };
export { parseProblem, problemMessage, transportProblem };

/**
 * Base URL handed to `openapi-fetch`.
 *
 * Same-origin by construction. The request goes exactly where a bare `"/api/v1"` would, but a
 * `Request` built from it is also valid where relative URLs are not (jsdom and Node's `undici`,
 * i.e. the unit tests).
 */
export const API_BASE_URL =
  typeof globalThis.location === "undefined"
    ? API_BASE_PATH
    : `${globalThis.location.origin}${API_BASE_PATH}`;

/** Header set on a replayed request so it is never refreshed and replayed a second time. */
const RETRY_HEADER = "X-Aicls-Retry";

/** Endpoints that must never trigger the refresh dance (they *are* the dance, or precede it). */
const NO_REFRESH_PATHS: ReadonlySet<string> = new Set([
  "/auth/login",
  "/auth/register",
  "/auth/refresh",
  "/auth/logout",
]);

/** What the session layer plugs into the client (registered by `app/auth.ts`). */
export interface SessionBridge {
  /** Current access token, or `null` when signed out. */
  getAccessToken: () => string | null;
  /** Store a token minted by `POST /auth/refresh`. */
  onTokenRefreshed: (accessToken: string, expiresInSeconds: number) => void;
  /** Drop the session: the refresh cookie is gone or was revoked. */
  onSignOut: (problem: AiclsProblem | null) => void;
}

const NULL_BRIDGE: SessionBridge = {
  getAccessToken: () => null,
  onTokenRefreshed: () => undefined,
  onSignOut: () => undefined,
};

let bridge: SessionBridge = NULL_BRIDGE;
let lastRequestIdValue: string | null = null;
let refreshInFlight: Promise<boolean> | null = null;

/**
 * Connect the client to the session store.
 *
 * Injected rather than imported so the two modules do not form a cycle, and so a test can drive
 * the whole refresh path without touching the real store.
 */
export function setSessionBridge(next: Partial<SessionBridge>): void {
  bridge = { ...NULL_BRIDGE, ...next };
}

/** Restore the inert bridge (used by tests and on teardown). */
export function resetSessionBridge(): void {
  bridge = NULL_BRIDGE;
  refreshInFlight = null;
}

/** Register only the token source (kept for call sites that need nothing else). */
export function setAuthTokenProvider(provider: () => string | null): void {
  bridge = { ...bridge, getAccessToken: provider };
}

/** Register only the sign-out callback. */
export function setUnauthorizedHandler(handler: (problem: AiclsProblem | null) => void): void {
  bridge = { ...bridge, onSignOut: handler };
}

/**
 * Correlation id of the most recent response.
 *
 * Shown in the support footer of an error so a report can be tied to a server log line.
 */
export function lastRequestId(): string | null {
  return lastRequestIdValue;
}

/** Type guard for RFC 9457 problem documents (re-exported for call sites that only need it). */
export const isProblemDetail = isProblemDocument;

/** The typed API client. Every path and body comes from the generated schema. */
export const api = createClient<paths>({
  baseUrl: API_BASE_URL,
  credentials: "include",
  // Resolve `fetch` per call instead of capturing it at module load: the client is created while
  // the module graph is still loading, and a late-installed implementation (a test double, a
  // service-worker shim) must be honoured.
  fetch: (request) => globalThis.fetch(request),
});

/** Requests kept alive (unread clones) so a 401 can be replayed after a refresh. */
const inFlight = new Map<string, Request>();

/**
 * Run `POST /auth/refresh` at most once at a time.
 *
 * Several requests can fail with 401 in the same tick (the workspace loads history, the book panel
 * searches, telemetry posts); they all await the same promise and then replay individually.
 */
async function refreshAccessToken(): Promise<boolean> {
  const pending =
    refreshInFlight ??
    (async (): Promise<boolean> => {
      try {
        const { data, error, response } = await api.POST("/auth/refresh", {});
        if (error !== undefined || data === undefined) {
          bridge.onSignOut(parseProblem(error, { response }));
          return false;
        }
        bridge.onTokenRefreshed(data.access_token, data.expires_in);
        return true;
      } catch (cause) {
        bridge.onSignOut(transportProblem(t("errors.network"), cause));
        return false;
      } finally {
        refreshInFlight = null;
      }
    })();
  refreshInFlight = pending;
  return pending;
}

api.use({
  onRequest({ request, schemaPath, id }) {
    const token = bridge.getAccessToken();
    if (token !== null && token !== "") {
      request.headers.set("Authorization", `Bearer ${token}`);
    }
    if (!request.headers.has("Accept")) {
      request.headers.set("Accept", "application/json, application/problem+json");
    }
    if (!NO_REFRESH_PATHS.has(schemaPath)) {
      // An unread clone: the original is about to be consumed by `fetch`, and a replay needs a
      // body that has never been read.
      inFlight.set(id, request.clone());
    }
    return request;
  },

  async onResponse({ request, response, schemaPath, id, options }) {
    const replay = inFlight.get(id);
    inFlight.delete(id);
    lastRequestIdValue = requestIdOf(response) ?? lastRequestIdValue;

    if (response.status !== 401) {
      return response;
    }
    // The auth endpoints own their own failures: a refused login is not a dropped session, and a
    // refused refresh is reported once by `refreshAccessToken`.
    if (NO_REFRESH_PATHS.has(schemaPath)) {
      return response;
    }
    if (replay === undefined || request.headers.has(RETRY_HEADER)) {
      bridge.onSignOut(await readProblem(response));
      return response;
    }

    const refreshed = await refreshAccessToken();
    if (!refreshed) {
      return response;
    }

    const token = bridge.getAccessToken();
    if (token !== null && token !== "") {
      replay.headers.set("Authorization", `Bearer ${token}`);
    }
    replay.headers.set(RETRY_HEADER, "1");
    try {
      const retried = await options.fetch(replay);
      lastRequestIdValue = requestIdOf(retried) ?? lastRequestIdValue;
      if (retried.status === 401) {
        bridge.onSignOut(await readProblem(retried));
      }
      return retried;
    } catch {
      // The replay itself failed to reach the server: hand the original 401 back so the caller
      // renders one honest error instead of two.
      return response;
    }
  },
});

/** Read a problem document from a response without disturbing the body the caller will parse. */
async function readProblem(response: Response): Promise<AiclsProblem | null> {
  try {
    const body: unknown = await response.clone().json();
    return isProblemDocument(body) ? parseProblem(body, { response }) : null;
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------------- multipart --- */

/**
 * Body serializer for the `multipart/form-data` endpoints (`/ocr/recognize`, `/asr/transcribe`,
 * `/contribute/recordings`).
 *
 * `openapi-typescript` types a `format: binary` field as `string`, so the typed body carries the
 * file through {@link filePart} and this serializer turns the object into a `FormData` — the
 * browser then sets the boundary itself.
 */
export function formDataSerializer(body: Readonly<Record<string, unknown>>): FormData {
  const form = new FormData();
  for (const [key, value] of Object.entries(body)) {
    if (value === undefined || value === null) {
      continue;
    }
    if (value instanceof Blob) {
      form.append(key, value);
    } else if (typeof value === "string") {
      form.append(key, value);
    } else if (typeof value === "number" || typeof value === "boolean") {
      form.append(key, String(value));
    } else {
      form.append(key, JSON.stringify(value));
    }
  }
  return form;
}

/**
 * Put a file into a typed multipart body.
 *
 * The cast is the one place the client admits that `format: binary` is a `string` in the generated
 * types and a `File` on the wire; {@link formDataSerializer} unpacks it again.
 */
export function filePart(blob: Blob, filename: string): string {
  const file = blob instanceof File ? blob : new File([blob], filename, { type: blob.type });
  return file as unknown as string;
}

/** What a wrapped call returns: never a thrown error, always something renderable. */
export interface ApiResult<T> {
  /** The parsed success body, or `null` when the call failed. */
  readonly data: T | null;
  /** The failure, with a Turkish `detail_tr`, or `null` on success. */
  readonly problem: AiclsProblem | null;
  /** `X-Request-Id` of the response, for support. */
  readonly requestId: string | null;
  /** `Server-Timing` breakdown of the request (empty when the header was absent). */
  readonly stages: StageTiming[];
  /** The raw response, for the rare caller that needs a header or the status. */
  readonly response: Response | null;
}

/** The shape `openapi-fetch` resolves to, narrowed to what {@link call} needs. */
type FetchOutcome<T> = {
  data?: T | undefined;
  error?: unknown;
  response: Response;
};

/**
 * Wrap an `openapi-fetch` call so the UI never has to branch on `error` or catch.
 *
 * @param promise - The result of `api.GET(...)` / `api.POST(...)`.
 * @param options.fallbackDetailTr - Turkish sentence for a failure that carries no `detail_tr`
 *   (a transport error, or a body that is not a problem document).
 * @returns The success body or a typed problem, plus the request id and the stage timings.
 *
 * @example
 * ```ts
 * const { data, problem, stages } = await call(api.POST("/compute/text", { body }), {
 *   fallbackDetailTr: t("errors.network"),
 * });
 * ```
 */
export async function call<T>(
  promise: Promise<FetchOutcome<T>>,
  options: { fallbackDetailTr: string },
): Promise<ApiResult<T>> {
  try {
    const { data, error, response } = await promise;
    const requestId = requestIdOf(response);
    const stages = serverTimingOf(response);
    // A 204 (logout, telemetry) resolves with neither `data` nor `error`; only the status decides
    // whether the call succeeded.
    if (!response.ok || error !== undefined) {
      return {
        data: null,
        problem: parseProblem(error, {
          response,
          fallbackDetailTr: options.fallbackDetailTr,
        }),
        requestId,
        stages,
        response,
      };
    }
    return { data: data ?? null, problem: null, requestId, stages, response };
  } catch (cause) {
    return {
      data: null,
      problem: transportProblem(options.fallbackDetailTr, cause),
      requestId: null,
      stages: [],
      response: null,
    };
  }
}
