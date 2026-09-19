/**
 * The single bridge between the generated OpenAPI schema and the rest of the client.
 *
 * `src/api/generated/schema.d.ts` is produced by `npm run gen:api`
 * (`openapi-typescript ../docs/api/openapi.json`) and is never edited by hand. Everything the UI
 * knows about the wire format is derived from it here, so a backend change that is regenerated
 * turns into a TypeScript error rather than into a runtime surprise.
 *
 * One transformation happens: the generated path keys carry the `/api/v1` prefix
 * (`"/api/v1/compute/text"`), while the client is constructed with that prefix as its `baseUrl`.
 * {@link paths} therefore re-keys the generated table without the prefix, which keeps the call
 * sites short (`api.POST("/compute/text")`) *and* keeps them checked against the real schema — a
 * path the backend drops disappears from this type and every caller stops compiling.
 */
import type {
  components as GeneratedComponents,
  operations as GeneratedOperations,
  paths as GeneratedPaths,
} from "./generated/schema";

/** Path prefix of the versioned REST API (docs/mimari/api.md §1). */
export const API_BASE_PATH = "/api/v1";

/** WebSocket prefix; the two streams are not part of the OpenAPI document (`app.py`). */
export const WS_BASE_PATH = "/ws/v1";

/** Remove the `/api/v1` prefix from every key of the generated path table. */
type WithoutApiPrefix<T> = {
  [K in keyof T as K extends `/api/v1${infer Rest}` ? Rest : never]: T[K];
};

/** The operation table `openapi-fetch` is instantiated with, keyed relative to {@link API_BASE_PATH}. */
export type paths = WithoutApiPrefix<GeneratedPaths>;

/** Every schema object of the OpenAPI document (`components.schemas.*`). */
export type components = GeneratedComponents;

/** Every operation of the OpenAPI document, keyed by `operationId`. */
export type operations = GeneratedOperations;

/** Shorthand for one named response/request model: `Schema<"ComputeOut">`. */
export type Schema<K extends keyof GeneratedComponents["schemas"]> =
  GeneratedComponents["schemas"][K];
