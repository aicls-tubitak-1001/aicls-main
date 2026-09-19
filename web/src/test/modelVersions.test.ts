/**
 * Stub detection (scientific integrity).
 *
 * The version strings here are the real ones the backend reports:
 * `aicls.ocr.stub.STUB_MODEL_VERSION == "stub-0.1.0"` and
 * `aicls.asr.stub.STUB_MODEL_VERSION == "asr-stub-1"`, plus the `"override"` that
 * `ModelRegistry.set_recognizer` records for a model injected at runtime. If any of them ever
 * stops being classified as "not a trained model", a demonstration could be mistaken for a
 * trained system — which is exactly what this test exists to prevent.
 */
import { describe, expect, it } from "vitest";

import { classifyModelVersion, classifyModelVersions, hasStubModel } from "@/lib/modelVersions";

describe("classifyModelVersion", () => {
  it("recognises the OCR and ASR stand-ins the backend ships", () => {
    expect(classifyModelVersion("stub-0.1.0")).toBe("stub");
    expect(classifyModelVersion("asr-stub-1")).toBe("stub");
  });

  it("is case-insensitive and tolerates surrounding whitespace", () => {
    expect(classifyModelVersion("  OCR-STUB-2  ")).toBe("stub");
  });

  it("flags a runtime-injected model as unverified rather than trained", () => {
    expect(classifyModelVersion("override")).toBe("override");
  });

  it("treats a promoted checkpoint as trained", () => {
    expect(classifyModelVersion("ocr-2027.03")).toBe("trained");
    expect(classifyModelVersion("whisper-small-tr-2027.06")).toBe("trained");
  });

  it("reports a missing version as unknown, never as trained", () => {
    expect(classifyModelVersion(null)).toBe("unknown");
    expect(classifyModelVersion(undefined)).toBe("unknown");
    expect(classifyModelVersion("   ")).toBe("unknown");
  });
});

describe("classifyModelVersions", () => {
  it("keeps a stable order so the badge does not reshuffle between answers", () => {
    const statuses = classifyModelVersions({
      zzz: "x-1",
      asr: "asr-stub-1",
      ocr: "stub-0.1.0",
      aaa: "y-1",
    });

    expect(statuses.map((status) => status.component)).toEqual(["ocr", "asr", "aaa", "zzz"]);
  });

  it("normalises a missing version to null", () => {
    expect(classifyModelVersions({ ocr: null })).toEqual([
      { component: "ocr", version: null, kind: "unknown" },
    ]);
    expect(classifyModelVersions(undefined)).toEqual([]);
  });

  it("answers whether any component answered with a stand-in", () => {
    expect(hasStubModel({ ocr: "stub-0.1.0", asr: "asr-2027.04" })).toBe(true);
    expect(hasStubModel({ ocr: "ocr-2027.03", asr: "asr-2027.04" })).toBe(false);
    expect(hasStubModel({ ocr: null })).toBe(false);
  });
});
