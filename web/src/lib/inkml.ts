/**
 * Ink serialisation: browser strokes → InkML (W3C) and → Canvas JSON.
 *
 * ADR-0013 keeps both handwriting transports alive behind one `Recognizer` interface: online traces
 * (InkML, the MathWriting-native format) and offline raster (PNG). This module owns the two
 * text formats; rasterising lives in `StrokeBuffer.toPNG`.
 *
 * The Canvas JSON shape is the payload of `POST /ocr/recognize` (`{ink: CanvasInkJson}`) and is
 * consumed server-side by `aicls.ocr.ink_from_canvas_json`. Coordinates are CSS pixels relative to
 * the canvas element, timestamps are milliseconds relative to the first sample of the first stroke,
 * so a trace is device- and clock-independent.
 */

/** One sampled pen position. */
export interface InkPoint {
  /** X in CSS pixels, relative to the canvas' left edge. */
  x: number;
  /** Y in CSS pixels, relative to the canvas' top edge. */
  y: number;
  /** Milliseconds since the first sample of the first stroke. */
  t: number;
}

/** A single pen-down → pen-up trace. */
export interface InkStroke {
  points: InkPoint[];
}

/** Size of the drawing surface in CSS pixels. */
export interface CanvasSize {
  w: number;
  h: number;
}

/** Canvas JSON payload accepted by `POST /ocr/recognize` and `/ws/v1/ink`. */
export interface InkJson {
  /** Schema version; bumped only with an ADR. */
  version: 1;
  canvas: CanvasSize;
  /** Parallel coordinate arrays per trace (compact and JSON-friendly). */
  strokes: { x: number[]; y: number[]; t: number[] }[];
}

/** Upload limit from api.md §1 ("InkML/Canvas JSON ≤ 1 MB ve ≤ 20 000 nokta"). */
export const MAX_INK_POINTS = 20_000;

/** Decimal places kept for coordinates (sub-pixel precision the recogniser can use). */
const COORD_PRECISION = 2;

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Drop empty traces — a tap without movement carries no signal. */
function usableStrokes(strokes: readonly InkStroke[]): InkStroke[] {
  return strokes.filter((stroke) => stroke.points.length > 0);
}

/**
 * Serialise strokes as W3C InkML.
 *
 * The document declares an explicit `traceFormat` (X, Y, T) and one `<trace>` element per stroke
 * with `x y t` triples separated by commas — the layout `aicls.ocr.read_inkml` and the MathWriting
 * corpus use.
 *
 * @param strokes - Traces in drawing order.
 * @param canvas - Canvas size, recorded as annotations so the server can normalise.
 * @param options.label - Ground-truth LaTeX, written as `<annotation type="label">` (data
 *   collection only; never set for user input).
 * @returns InkML text with LF line endings.
 */
export function strokesToInkML(
  strokes: readonly InkStroke[],
  canvas: CanvasSize,
  options: { label?: string } = {},
): string {
  const traces = usableStrokes(strokes).map((stroke) => {
    const body = stroke.points
      .map(
        (p) =>
          `${String(round(p.x, COORD_PRECISION))} ${String(round(p.y, COORD_PRECISION))} ${String(Math.round(p.t))}`,
      )
      .join(", ");
    return `  <trace>${body}</trace>`;
  });
  const annotations = [
    `  <annotation type="creator">AI-CLS web</annotation>`,
    `  <annotation type="canvasWidth">${String(Math.round(canvas.w))}</annotation>`,
    `  <annotation type="canvasHeight">${String(Math.round(canvas.h))}</annotation>`,
  ];
  if (options.label !== undefined) {
    annotations.push(`  <annotation type="label">${escapeXml(options.label)}</annotation>`);
  }
  return [
    `<?xml version="1.0" encoding="UTF-8"?>`,
    `<ink xmlns="http://www.w3.org/2003/InkML">`,
    ...annotations,
    `  <traceFormat>`,
    `    <channel name="X" type="decimal"/>`,
    `    <channel name="Y" type="decimal"/>`,
    `    <channel name="T" type="integer"/>`,
    `  </traceFormat>`,
    ...traces,
    `</ink>`,
    "",
  ].join("\n");
}

/** Serialise strokes as the compact Canvas JSON payload. */
export function strokesToCanvasJson(strokes: readonly InkStroke[], canvas: CanvasSize): InkJson {
  return {
    version: 1,
    canvas: { w: Math.round(canvas.w), h: Math.round(canvas.h) },
    strokes: usableStrokes(strokes).map((stroke) => ({
      x: stroke.points.map((p) => round(p.x, COORD_PRECISION)),
      y: stroke.points.map((p) => round(p.y, COORD_PRECISION)),
      t: stroke.points.map((p) => Math.round(p.t)),
    })),
  };
}

/** Total number of sampled points (checked against {@link MAX_INK_POINTS} before upload). */
export function countPoints(strokes: readonly InkStroke[]): number {
  return strokes.reduce((total, stroke) => total + stroke.points.length, 0);
}

/**
 * Draw strokes onto a 2D context (shared by the live canvas and the PNG export).
 *
 * @param ctx - Destination context; already cleared by the caller.
 * @param strokes - Traces to draw.
 * @param options.scale - Device/export scale factor.
 * @param options.color - Stroke colour (CSS).
 * @param options.width - Stroke width in CSS pixels before scaling.
 */
export function drawStrokes(
  ctx: CanvasRenderingContext2D,
  strokes: readonly InkStroke[],
  options: { scale?: number; color?: string; width?: number } = {},
): void {
  const scale = options.scale ?? 1;
  ctx.save();
  ctx.scale(scale, scale);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = options.color ?? "#12303a";
  ctx.lineWidth = options.width ?? 2.5;
  for (const stroke of usableStrokes(strokes)) {
    const [first, ...rest] = stroke.points;
    if (first === undefined) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(first.x, first.y);
    if (rest.length === 0) {
      // A dot: a zero-length segment with a round cap still paints a mark.
      ctx.lineTo(first.x + 0.01, first.y);
    }
    for (const point of rest) {
      ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
  }
  ctx.restore();
}
