/**
 * `InkCanvas.test.tsx` — contractual test from `docs/mimari/moduller.md#web--sahibi-sg`:
 * two pointer strokes must produce InkML with two `<trace>` elements (ADR-0013).
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { InkCanvas } from "@/features/workspace/InkCanvas";
import type { StrokeBuffer } from "@/features/workspace/StrokeBuffer";

/** Draw one stroke as a pointer-down, two moves and a pointer-up. */
function drawStroke(canvas: HTMLElement, from: { x: number; y: number }): void {
  fireEvent.pointerDown(canvas, { clientX: from.x, clientY: from.y, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: from.x + 10, clientY: from.y + 4, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: from.x + 20, clientY: from.y + 9, pointerId: 1 });
  fireEvent.pointerUp(canvas, { clientX: from.x + 20, clientY: from.y + 9, pointerId: 1 });
}

describe("InkCanvas", () => {
  it("turns two pointer strokes into InkML with two traces", () => {
    const onStrokeEnd = vi.fn<(buffer: StrokeBuffer) => void>();
    render(<InkCanvas onStrokeEnd={onStrokeEnd} />);
    const canvas = screen.getByTestId("ink-canvas");

    drawStroke(canvas, { x: 12, y: 30 });
    drawStroke(canvas, { x: 60, y: 30 });

    expect(onStrokeEnd).toHaveBeenCalledTimes(2);
    const buffer = onStrokeEnd.mock.calls[1]?.[0];
    expect(buffer).toBeDefined();
    const inkml = buffer?.toInkML() ?? "";

    const traces = inkml.match(/<trace>/g) ?? [];
    expect(traces).toHaveLength(2);
    expect(inkml).toContain('xmlns="http://www.w3.org/2003/InkML"');
    expect(inkml).toContain("<traceFormat>");
    // Each trace carries "x y t" triples separated by commas.
    expect(inkml).toMatch(/<trace>[-\d.]+ [-\d.]+ \d+(, [-\d.]+ [-\d.]+ \d+)+<\/trace>/);
  });

  it("records three sampled points per stroke in the Canvas JSON payload", () => {
    const onStrokeEnd = vi.fn<(buffer: StrokeBuffer) => void>();
    render(<InkCanvas width={640} height={200} onStrokeEnd={onStrokeEnd} />);
    const canvas = screen.getByTestId("ink-canvas");

    drawStroke(canvas, { x: 5, y: 5 });

    const json = onStrokeEnd.mock.calls[0]?.[0].toCanvasJson();
    expect(json?.version).toBe(1);
    expect(json?.canvas).toEqual({ w: 640, h: 200 });
    expect(json?.strokes).toHaveLength(1);
    expect(json?.strokes[0]?.x).toEqual([5, 15, 25]);
    expect(json?.strokes[0]?.y).toEqual([5, 9, 14]);
  });

  it("reports the stroke count and clears the buffer", () => {
    const onClear = vi.fn();
    render(<InkCanvas onClear={onClear} />);
    const canvas = screen.getByTestId("ink-canvas");

    drawStroke(canvas, { x: 10, y: 10 });
    expect(screen.getByTestId("stroke-count")).toHaveTextContent("1");

    fireEvent.click(screen.getByRole("button", { name: /temizle/i }));
    expect(onClear).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("stroke-count")).toHaveTextContent("0");
  });

  it("does not record strokes while disabled", () => {
    const onStrokeEnd = vi.fn<(buffer: StrokeBuffer) => void>();
    render(<InkCanvas disabled onStrokeEnd={onStrokeEnd} />);

    drawStroke(screen.getByTestId("ink-canvas"), { x: 10, y: 10 });

    expect(onStrokeEnd).not.toHaveBeenCalled();
  });
});
