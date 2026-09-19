/**
 * RFC 9457 parsing (api.md §2).
 *
 * The rule these tests pin down: whatever arrives, the UI gets a Turkish sentence it can render.
 * A well-formed problem keeps its `detail_tr` and its structured extras; a malformed one falls
 * back without throwing; a transport failure is marked as a client-side condition and never
 * pretends to be a backend error code.
 */
import { describe, expect, it } from "vitest";

import {
  ERROR_CODES,
  isKnownErrorCode,
  isProblemDocument,
  parseProblem,
  problemMessage,
  requestIdOf,
  transportProblem,
  TRANSPORT_FAILURE_CODE,
} from "@/api/problem";

function response(status: number, headers: Record<string, string> = {}): Response {
  return new Response(null, { status, headers });
}

describe("parseProblem", () => {
  it("keeps every documented field of a parse error", () => {
    const problem = parseProblem({
      type: "https://ai-cls.example/errors/PARSE_ERROR",
      title: "LaTeX ifadesi ayrıştırılamadı",
      status: 422,
      code: "PARSE_ERROR",
      detail_tr: "Konum 8: beklenmeyen simge.",
      instance: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
      position: 8,
    });

    expect(problem.code).toBe("PARSE_ERROR");
    expect(problem.status).toBe(422);
    expect(problem.detail_tr).toBe("Konum 8: beklenmeyen simge.");
    expect(problem.instance).toBe("01J8R2ZQ7G4V0X9K3M2N5P6Q7R");
    expect(problem.position).toBe(8);
    expect(problem.extra).toEqual({});
  });

  it("keeps the unsupported-construct name and the contract sentence", () => {
    const problem = parseProblem({
      type: "https://ai-cls.example/errors/UNSUPPORTED_PROBLEM",
      title: "Desteklenmeyen problem tipi",
      status: 422,
      code: "UNSUPPORTED_PROBLEM",
      detail_tr: "Bu problem tipi şu an için desteklenmemektedir.",
      construct: "triple_integral",
    });

    expect(problem.detail_tr).toBe("Bu problem tipi şu an için desteklenmemektedir.");
    expect(problem.construct).toBe("triple_integral");
  });

  it("collects unknown structured fields into `extra` instead of dropping them", () => {
    const problem = parseProblem({
      title: "Çok fazla istek",
      status: 429,
      code: "RATE_LIMITED",
      detail_tr: "Çok fazla istek gönderildi.",
      retry_after: 3,
      rule: "compute_user",
      limit: 30,
      window_s: 60,
    });

    expect(problem.retry_after).toBe(3);
    expect(problem.extra).toEqual({ rule: "compute_user", limit: 30, window_s: 60 });
  });

  it("normalises the validation entries of a 422", () => {
    const problem = parseProblem({
      title: "Geçersiz istek",
      status: 422,
      code: "VALIDATION_ERROR",
      detail_tr: "Girdi doğrulanamadı.",
      errors: [{ loc: ["body", "text"], msg: "field required" }, "nonsense"],
    });

    expect(problem.errors).toEqual([{ loc: ["body", "text"], msg: "field required" }]);
  });

  it("falls back to the title when detail_tr is missing", () => {
    const problem = parseProblem({ title: "Bulunamadı", status: 404, code: "NOT_FOUND" });

    expect(problem.detail_tr).toBe("Bulunamadı");
  });

  it("uses the caller's Turkish fallback for a body that is not a problem document", () => {
    const problem = parseProblem(
      { detail: "Internal Server Error" },
      { response: response(500), fallbackDetailTr: "Beklenmeyen bir sunucu hatası oluştu." },
    );

    expect(problem.status).toBe(500);
    expect(problem.code).toBe("INTERNAL");
    expect(problem.detail_tr).toBe("Beklenmeyen bir sunucu hatası oluştu.");
  });

  it("takes the request id from the response when the document has no instance", () => {
    const problem = parseProblem(
      { title: "Hata", status: 500, code: "INTERNAL", detail_tr: "Sunucu hatası." },
      { response: response(500, { "X-Request-Id": "req-1" }) },
    );

    expect(problem.instance).toBe("req-1");
  });

  it("marks a payload-less failure as a client-side transport condition", () => {
    const problem = parseProblem(null, { fallbackDetailTr: "Sunucuya ulaşılamadı." });

    expect(problem.code).toBe(TRANSPORT_FAILURE_CODE);
    expect(problem.status).toBe(0);
    expect(isKnownErrorCode(problem.code)).toBe(false);
  });

  it("renders a code from a newer backend rather than throwing", () => {
    const problem = parseProblem({
      title: "Yeni hata",
      status: 418,
      code: "SOMETHING_NEW",
      detail_tr: "Yeni bir hata türü.",
    });

    expect(problem.code).toBe("SOMETHING_NEW");
    expect(isKnownErrorCode(problem.code)).toBe(false);
    expect(problem.detail_tr).toBe("Yeni bir hata türü.");
  });
});

describe("helpers", () => {
  it("lists exactly the backend's error codes", () => {
    expect(ERROR_CODES).toContain("UNSUPPORTED_PROBLEM");
    expect(ERROR_CODES).toContain("GATE_BLOCKED");
    expect(ERROR_CODES).not.toContain(TRANSPORT_FAILURE_CODE);
  });

  it("recognises a problem document by code and status", () => {
    expect(isProblemDocument({ code: "NOT_FOUND", status: 404 })).toBe(true);
    expect(isProblemDocument({ code: "NOT_FOUND" })).toBe(false);
    expect(isProblemDocument("hata")).toBe(false);
  });

  it("reads the correlation header", () => {
    expect(requestIdOf(response(200, { "X-Request-Id": "abc" }))).toBe("abc");
    expect(requestIdOf(response(200))).toBeNull();
    expect(requestIdOf(null)).toBeNull();
  });

  it("builds a transport problem that carries the reason without leaking it to the user", () => {
    const problem = transportProblem("Sunucuya ulaşılamadı.", new Error("ECONNREFUSED"));

    expect(problem.detail_tr).toBe("Sunucuya ulaşılamadı.");
    expect(problem.extra).toEqual({ reason: "ECONNREFUSED" });
  });

  it("problemMessage prefers detail_tr, then title, then the fallback", () => {
    expect(problemMessage({ code: "X", status: 400, detail_tr: "bir" }, "yedek")).toBe("bir");
    expect(problemMessage({ code: "X", status: 400, title: "iki" }, "yedek")).toBe("iki");
    expect(problemMessage(undefined, "yedek")).toBe("yedek");
  });
});
