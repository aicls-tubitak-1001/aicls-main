/**
 * Tiny `fetch` router for component tests.
 *
 * The typed client resolves `globalThis.fetch` per call (see `api/client.ts`), so a stub installed
 * here intercepts every request without touching the network. Routes are keyed `"<METHOD> <path>"`
 * exactly as they appear in `docs/mimari/api.md`, which keeps the tests readable next to the spec.
 */
import { vi, type Mock } from "vitest";

/** A route value: a ready JSON body (object or array), or a function building the response. */
export type RouteHandler = ((request: Request) => Response | Promise<Response>) | object;

/** Build a JSON response with the content type `openapi-fetch` expects. */
export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Build an RFC 9457 problem response (api.md §2). */
export function problemResponse(code: string, status: number, detailTr: string): Response {
  return new Response(
    JSON.stringify({
      type: `https://ai-cls.example/errors/${code}`,
      title: detailTr,
      status,
      code,
      detail_tr: detailTr,
    }),
    { status, headers: { "content-type": "application/problem+json" } },
  );
}

/**
 * Install a `fetch` stub that answers the given routes.
 *
 * @param routes - Map of `"POST /compute/text"` to a body or a handler.
 * @returns The mock, so tests can assert on the requests that were made.
 */
export function installApiMock(routes: Record<string, RouteHandler>): Mock {
  const fetchMock = vi.fn(async (input: Request | string): Promise<Response> => {
    const request = typeof input === "string" ? new Request(input) : input;
    const key = `${request.method} ${new URL(request.url).pathname.replace("/api/v1", "")}`;
    const handler = routes[key];
    if (handler === undefined) {
      return problemResponse("NOT_FOUND", 404, `Sahte uç nokta yok: ${key}`);
    }
    if (typeof handler === "function") {
      return await (handler as (r: Request) => Response | Promise<Response>)(request);
    }
    return jsonResponse(handler);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** Parse the JSON body of a recorded request. */
export async function requestBody(mock: Mock, index = 0): Promise<unknown> {
  const request = mock.mock.calls[index]?.[0] as Request | undefined;
  if (request === undefined) {
    throw new Error("istek kaydedilmedi");
  }
  return await request.clone().json();
}
