/**
 * The wire contract, checked against the backend's own OpenAPI document.
 *
 * TypeScript already stops the client calling a path that is not in the generated schema — that is
 * the whole point of `src/api/schema.ts`. What TypeScript cannot see is (a) whether the generated
 * schema still matches the document the backend actually publishes, and (b) the two call sites
 * that deliberately step around the generated types because `openapi-typescript` cannot express
 * their bodies (a free-form ink object, a multipart part FastAPI reads off the raw request).
 *
 * This file closes both gaps at runtime:
 *
 * 1. `web/src/api/generated/openapi.json` must be byte-identical to `docs/api/openapi.json`. A
 *    backend change that was not regenerated fails here instead of failing in a lecture.
 * 2. Every `METHOD path` the client sends — the typed calls scraped out of the source, plus the
 *    handful of untyped ones listed explicitly below — must exist in that document. A path the
 *    backend renames or drops fails here by name.
 *
 * Scraping the source rather than maintaining a hand-written list is deliberate: a new
 * `api.POST("/…")` anywhere in `src/` is picked up by the next run with no bookkeeping, so the
 * test cannot quietly fall behind the client it guards.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { API_BASE_PATH } from "@/api/schema";

/** Vitest runs with `web/` as the working directory (that is where `vite.config.ts` lives). */
const WEB_ROOT = resolve(".");
const GENERATED_SPEC = resolve(WEB_ROOT, "src/api/generated/openapi.json");
const COMMITTED_SPEC = resolve(WEB_ROOT, "../docs/api/openapi.json");
const SOURCE_ROOT = resolve(WEB_ROOT, "src");

/** HTTP methods an OpenAPI path item may carry. */
const METHODS = ["get", "put", "post", "delete", "options", "head", "patch", "trace"] as const;

/**
 * Calls that cannot be scraped because they do not go through the typed `api.<METHOD>(…)` form.
 *
 * Each one is a documented cast in the client; listing it here means a rename in the OpenAPI
 * document still breaks the build rather than the lecture.
 */
const UNTYPED_CALLS: readonly { readonly call: string; readonly why: string }[] = [
  {
    // `src/features/admin/api.ts` — FastAPI reads the archive off the raw `Request`, so the
    // document declares no `requestBody` and `openapi-fetch` has no body type to check.
    call: "POST /admin/book/load",
    why: "multipart archive read from the raw request (aicls.api.bookload.load_book_archive)",
  },
];

/** Every `.ts`/`.tsx` file under `src/`, excluding the generated schema and the tests themselves. */
function sourceFiles(directory: string): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(directory)) {
    const full = join(directory, entry);
    if (statSync(full).isDirectory()) {
      if (entry === "generated" || entry === "test") {
        continue;
      }
      found.push(...sourceFiles(full));
      continue;
    }
    if (entry.endsWith(".ts") || entry.endsWith(".tsx")) {
      found.push(full);
    }
  }
  return found;
}

/** One `api.<METHOD>("<path>")` occurrence. */
interface ClientCall {
  /** `"POST /compute/text"`, relative to {@link API_BASE_PATH}. */
  readonly call: string;
  /** Where it was found, for a failure message that names the file. */
  readonly where: string;
}

const CALL_RE = /\bapi\.(GET|PUT|POST|DELETE|OPTIONS|HEAD|PATCH|TRACE)\(\s*"([^"]+)"/g;

/** Scrape every typed call site out of the client's own source. */
function scrapeCalls(): ClientCall[] {
  const calls: ClientCall[] = [];
  for (const file of sourceFiles(SOURCE_ROOT)) {
    const text = readFileSync(file, "utf-8");
    for (const match of text.matchAll(CALL_RE)) {
      calls.push({
        call: `${match[1] ?? ""} ${match[2] ?? ""}`,
        where: relative(WEB_ROOT, file).replaceAll("\\", "/"),
      });
    }
  }
  return calls;
}

/** `"POST /api/v1/compute/text"` for every operation the document declares. */
function specOperations(spec: {
  paths: Record<string, Record<string, unknown>>;
}): ReadonlySet<string> {
  const operations = new Set<string>();
  for (const [path, item] of Object.entries(spec.paths)) {
    for (const method of METHODS) {
      if (method in item) {
        operations.add(`${method.toUpperCase()} ${path}`);
      }
    }
  }
  return operations;
}

const generatedText = readFileSync(GENERATED_SPEC, "utf-8");
const spec = JSON.parse(generatedText) as { paths: Record<string, Record<string, unknown>> };
const OPERATIONS = specOperations(spec);

describe("API sözleşmesi", () => {
  it("keeps the generated schema identical to the committed OpenAPI document", () => {
    // Compared as parsed JSON so a line ending or a trailing newline is not reported as drift;
    // any difference in content still fails, and `npm run gen:api` is the fix.
    expect(JSON.parse(generatedText)).toEqual(
      JSON.parse(readFileSync(COMMITTED_SPEC, "utf-8")) as unknown,
    );
  });

  it("finds the calls it is supposed to guard", () => {
    // A regex that silently stops matching would turn this whole file into a no-op.
    const calls = scrapeCalls();
    expect(calls.length).toBeGreaterThan(30);
    expect(calls.map((entry) => entry.call)).toContain("POST /compute/text");
  });

  it("calls no endpoint that is absent from the schema", () => {
    const missing = scrapeCalls()
      .map((entry) => {
        const [method = "", path = ""] = entry.call.split(" ");
        return { ...entry, full: `${method} ${API_BASE_PATH}${path}` };
      })
      .filter((entry) => !OPERATIONS.has(entry.full))
      .map((entry) => `${entry.full}  (${entry.where})`);

    expect(missing, "şemada bulunmayan uç noktalar").toEqual([]);
  });

  it("keeps the untyped call sites pointing at real endpoints", () => {
    for (const { call, why } of UNTYPED_CALLS) {
      const [method = "", path = ""] = call.split(" ");
      expect(OPERATIONS.has(`${method} ${API_BASE_PATH}${path}`), `${call} — ${why}`).toBe(true);
    }
  });

  it("uses only methods the schema declares for each path", () => {
    // A path that exists under a different verb is the failure this catches: `DELETE` on a
    // collection that only supports `GET`, for instance.
    const paths = new Set(Object.keys(spec.paths));
    for (const entry of scrapeCalls()) {
      const [, path = ""] = entry.call.split(" ");
      expect(paths.has(`${API_BASE_PATH}${path}`), `${entry.call} (${entry.where})`).toBe(true);
    }
  });
});
