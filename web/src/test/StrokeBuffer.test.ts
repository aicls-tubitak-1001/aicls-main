/**
 * Stroke buffer and ink serialisation (ADR-0013).
 *
 * The InkML shape is what `aicls.ocr.read_inkml` parses and what MathWriting uses, so the layout is
 * asserted explicitly rather than through a snapshot.
 */
import { describe, expect, it } from "vitest";

import { InkTooLargeError, StrokeBuffer } from "@/features/workspace/StrokeBuffer";
import { countPoints, strokesToCanvasJson, strokesToInkML } from "@/lib/inkml";

function buildBuffer(): StrokeBuffer {
  const buffer = new StrokeBuffer({ canvas: { w: 400, h: 120 } });
  buffer.push({ x: 10, y: 20, t: 1000 });
  buffer.push({ x: 12.345, y: 21, t: 1016 });
  buffer.endStroke();
  buffer.push({ x: 40, y: 20, t: 1100 });
  buffer.endStroke();
  return buffer;
}

describe("StrokeBuffer", () => {
  it("normalises timestamps to the first sample", () => {
    const json = buildBuffer().toCanvasJson();

    expect(json.strokes[0]?.t).toEqual([0, 16]);
    expect(json.strokes[1]?.t).toEqual([100]);
  });

  it("rounds coordinates to two decimals", () => {
    const json = buildBuffer().toCanvasJson();

    expect(json.strokes[0]?.x).toEqual([10, 12.35]);
  });

  it("emits one InkML trace per stroke with the canvas annotations", () => {
    const inkml = buildBuffer().toInkML();

    expect(inkml.match(/<trace>/g)).toHaveLength(2);
    expect(inkml).toContain('<annotation type="canvasWidth">400</annotation>');
    expect(inkml).toContain('<annotation type="canvasHeight">120</annotation>');
    expect(inkml).toContain("<trace>10 20 0, 12.35 21 16</trace>");
    expect(inkml.endsWith("</ink>\n")).toBe(true);
  });

  it("discards an empty stroke and counts only completed ones", () => {
    const buffer = new StrokeBuffer();
    buffer.endStroke();

    expect(buffer.strokeCount).toBe(0);
    expect(buffer.isEmpty).toBe(true);
  });

  it("drops samples closer than minDistance", () => {
    const buffer = new StrokeBuffer({ minDistance: 5 });
    buffer.push({ x: 0, y: 0, t: 0 });
    buffer.push({ x: 1, y: 0, t: 1 });
    buffer.push({ x: 10, y: 0, t: 2 });
    buffer.endStroke();

    expect(countPoints(buffer.getStrokes())).toBe(2);
  });

  it("refuses to grow past the contract's point budget", () => {
    const buffer = new StrokeBuffer({ maxPoints: 2 });
    buffer.push({ x: 0, y: 0, t: 0 });
    buffer.push({ x: 1, y: 1, t: 1 });

    expect(() => buffer.push({ x: 2, y: 2, t: 2 })).toThrow(InkTooLargeError);
  });

  it("clears every trace and the time origin", () => {
    const buffer = buildBuffer();
    buffer.clear();
    buffer.push({ x: 5, y: 5, t: 9999 });
    buffer.endStroke();

    expect(buffer.strokeCount).toBe(1);
    expect(buffer.toCanvasJson().strokes[0]?.t).toEqual([0]);
  });

  it("hands out defensive copies of its strokes", () => {
    const buffer = buildBuffer();
    const strokes = buffer.getStrokes();
    strokes[0]?.points.push({ x: 999, y: 999, t: 999 });

    expect(countPoints(buffer.getStrokes())).toBe(3);
  });
});

describe("inkml helpers", () => {
  it("escapes the optional label annotation", () => {
    const inkml = strokesToInkML(
      [{ points: [{ x: 1, y: 2, t: 0 }] }],
      { w: 10, h: 10 },
      { label: "a<b&c" },
    );

    expect(inkml).toContain('<annotation type="label">a&lt;b&amp;c</annotation>');
  });

  it("skips empty traces in both serialisations", () => {
    const strokes = [{ points: [] }, { points: [{ x: 1, y: 1, t: 0 }] }];

    expect(strokesToInkML(strokes, { w: 4, h: 4 }).match(/<trace>/g)).toHaveLength(1);
    expect(strokesToCanvasJson(strokes, { w: 4, h: 4 }).strokes).toHaveLength(1);
  });
});
