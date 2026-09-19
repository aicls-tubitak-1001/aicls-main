/**
 * `Server-Timing` parsing (ADR-0014).
 *
 * The header is the client's only view of where a slow answer went, and it is written by
 * `StageTimer.server_timing_header()`; these tests pin the exact format that function emits, plus
 * the tolerance a malformed header needs — a broken header must cost the lecturer a breakdown,
 * never a result.
 */
import { describe, expect, it } from "vitest";

import {
  formatMs,
  mergeStages,
  parseServerTiming,
  serverTimingOf,
  sortStages,
  timingsToStages,
  totalMs,
  TOTAL_STAGE,
} from "@/lib/serverTiming";

/** Exactly what `StageTimer.server_timing_header()` produces. */
const HEADER = "nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4, app;dur=58.2";

describe("parseServerTiming", () => {
  it("parses the header StageTimer writes", () => {
    expect(parseServerTiming(HEADER)).toEqual([
      { name: "nlu", ms: 12.3 },
      { name: "parse", ms: 1.1 },
      { name: "cas", ms: 41 },
      { name: "render", ms: 0.4 },
      { name: "app", ms: 58.2 },
    ]);
  });

  it("keeps a desc parameter and tolerates extra whitespace", () => {
    expect(parseServerTiming('  cas;dur=41.0;desc="sympy" ,  app;dur=58.2 ')).toEqual([
      { name: "cas", ms: 41, description: "sympy" },
      { name: "app", ms: 58.2 },
    ]);
  });

  it("drops metrics without a duration and survives nonsense", () => {
    expect(parseServerTiming("cache;desc=hit, cas;dur=41.0, ;dur=1, bad;dur=abc")).toEqual([
      { name: "cas", ms: 41 },
    ]);
  });

  it("returns an empty list for a missing header", () => {
    expect(parseServerTiming(null)).toEqual([]);
    expect(parseServerTiming(undefined)).toEqual([]);
    expect(parseServerTiming("   ")).toEqual([]);
  });

  it("reads the header off a response", () => {
    const response = new Response(null, { headers: { "Server-Timing": "cas;dur=2" } });

    expect(serverTimingOf(response)).toEqual([{ name: "cas", ms: 2 }]);
    expect(serverTimingOf(null)).toEqual([]);
  });
});

describe("body timings", () => {
  it("renames `total` to the header's own total metric", () => {
    expect(timingsToStages({ nlu: 12.3, parse: 1.1, cas: 41, total: 58.2 })).toEqual([
      { name: "nlu", ms: 12.3 },
      { name: "parse", ms: 1.1 },
      { name: "cas", ms: 41 },
      { name: TOTAL_STAGE, ms: 58.2 },
    ]);
  });

  it("ignores null stages and non-numeric values (TimingsOut allows both)", () => {
    expect(timingsToStages({ nlu: null, parse: "x", cas: 41 })).toEqual([{ name: "cas", ms: 41 }]);
    expect(timingsToStages(undefined)).toEqual([]);
  });

  it("orders stages the way the pipeline runs, unknown ones last", () => {
    const sorted = sortStages([
      { name: "render", ms: 1 },
      { name: "mystery", ms: 2 },
      { name: "ocr", ms: 3 },
      { name: "cas", ms: 4 },
    ]);

    expect(sorted.map((stage) => stage.name)).toEqual(["ocr", "cas", "render", "mystery"]);
  });
});

describe("merge and totals", () => {
  it("lets the header win over the body for a stage both report", () => {
    const merged = mergeStages(
      [{ name: "cas", ms: 41 }],
      [
        { name: "cas", ms: 39 },
        { name: "nlu", ms: 12 },
      ],
    );

    expect(merged).toEqual([
      { name: "nlu", ms: 12 },
      { name: "cas", ms: 41 },
    ]);
  });

  it("uses the app metric as the total, or sums the stages when it is absent", () => {
    expect(totalMs(parseServerTiming(HEADER))).toBeCloseTo(58.2, 5);
    expect(
      totalMs([
        { name: "nlu", ms: 10 },
        { name: "cas", ms: 30 },
      ]),
    ).toBe(40);
  });

  it("keeps one decimal under 10 ms so a fast stage does not read as zero", () => {
    expect(formatMs(0.4)).toBe("0.4");
    expect(formatMs(58.2)).toBe("58");
    expect(formatMs(-1)).toBe("0");
  });
});
