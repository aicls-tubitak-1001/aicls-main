/**
 * Turning an RFC 9457 problem into per-field form errors.
 *
 * The backend is the authority on what is wrong with a credential: a 422 carries
 * `errors: [{loc, msg}]` (`aicls.api.errors` renders FastAPI's validation failures into the problem
 * document), and everything else — a refused login, an address that is already registered — arrives
 * as one Turkish `detail_tr` that belongs above the form rather than under a field. This module
 * does that split and nothing else, so no screen invents a validation rule of its own.
 *
 * The only checks the client performs are "this required box is empty", which exists so a
 * lecturer is not sent on a round trip to be told they typed nothing.
 */
import type { AiclsProblem } from "@/api/problem";

/** Field errors keyed by the API's own field names (`email`, `password`, `display_name`, …). */
export type FieldErrors = Readonly<Record<string, string>>;

/**
 * Extract per-field messages from a problem.
 *
 * @param problem - The parsed failure, or `null`.
 * @returns A map from field name to the message the backend supplied; empty when the failure was
 *   not a field-level one.
 *
 * @example
 * ```ts
 * fieldErrorsOf(problem).password; // "Parola en az 10 karakter olmalı." (from the API)
 * ```
 */
export function fieldErrorsOf(problem: AiclsProblem | null): FieldErrors {
  if (problem === null || problem.errors === undefined) {
    return {};
  }
  const errors: Record<string, string> = {};
  for (const entry of problem.errors) {
    // `loc` is ["body", "<field>"] for a body field; the last string segment names the field.
    const field = [...entry.loc].reverse().find((part) => typeof part === "string");
    if (typeof field !== "string" || field === "body" || entry.msg === "") {
      continue;
    }
    errors[field] ??= entry.msg;
  }
  return errors;
}

/**
 * Whether the failure belongs above the form rather than under a field.
 *
 * A 422 whose every message was attributed to a field is already shown there; anything else — a
 * wrong password (`AUTH_INVALID`), a locked account, a network failure — is a form-level message.
 */
export function isFormLevel(problem: AiclsProblem | null): boolean {
  if (problem === null) {
    return false;
  }
  return Object.keys(fieldErrorsOf(problem)).length === 0;
}

/**
 * Whether the failure is "you are not (or no longer) signed in".
 *
 * `AUTH_INVALID` covers an expired or missing token and `FORBIDDEN` a role that may not perform
 * the action; both are worth offering a sign-in link for, and neither is worth throwing away a
 * filled-in form over.
 */
export function isAuthProblem(problem: AiclsProblem | null): boolean {
  if (problem === null) {
    return false;
  }
  return problem.code === "AUTH_INVALID" || problem.code === "FORBIDDEN" || problem.status === 401;
}
