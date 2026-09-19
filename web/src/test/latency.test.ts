/**
 * Client-side latency marks (ADR-0014).
 *
 * The contractual metric is measured from the browser's own clock, so the sample must carry both a
 * monotonic duration (`render_ms`) and wall-clock endpoints the server can join with its stage
 * timings.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { markRendered, markSubmit, resetLatencyMarks } from "@/lib/latency";

beforeEach(() => {
  resetLatencyMarks();
});

afterEach(() => {
  vi.restoreAllMocks();
  resetLatencyMarks();
});

describe("latency marks", () => {
  it("measures the submit → render duration", () => {
    const nowSpy = vi.spyOn(performance, "now");
    nowSpy.mockReturnValueOnce(1_000).mockReturnValueOnce(1_250.5);

    markSubmit("op-1", "text");
    const sample = markRendered("op-1");

    expect(sample).not.toBeNull();
    expect(sample?.operation_id).toBe("op-1");
    expect(sample?.modality).toBe("text");
    expect(sample?.render_ms).toBeCloseTo(250.5, 3);
    expect(sample?.client_render_ts).toBeGreaterThanOrEqual(sample?.client_submit_ts ?? 0);
  });

  it("defaults the modality to text", () => {
    markSubmit("op-2");

    expect(markRendered("op-2")?.modality).toBe("text");
  });

  it("keeps the modality of a voice command", () => {
    markSubmit("op-3", "audio");

    expect(markRendered("op-3")?.modality).toBe("audio");
  });

  it("returns null when the operation was never submitted", () => {
    expect(markRendered("missing")).toBeNull();
  });

  it("consumes the mark so a second render does not report twice", () => {
    markSubmit("op-4");

    expect(markRendered("op-4")).not.toBeNull();
    expect(markRendered("op-4")).toBeNull();
  });

  it("evicts the oldest mark instead of growing without bound", () => {
    for (let i = 0; i < 40; i += 1) {
      markSubmit(`op-${String(i)}`);
    }

    expect(markRendered("op-0")).toBeNull();
    expect(markRendered("op-39")).not.toBeNull();
  });
});
