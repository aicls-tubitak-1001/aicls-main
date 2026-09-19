/**
 * In-memory pen-stroke buffer for the handwriting canvas (İP-2 / ADR-0013).
 *
 * `InkCanvas` feeds pointer samples in; the buffer owns the trace list and produces the three
 * transports the backend accepts: InkML, Canvas JSON and PNG. Timestamps are normalised to the
 * first sample so a trace never leaks wall-clock time.
 */
import {
  countPoints,
  drawStrokes,
  MAX_INK_POINTS,
  strokesToCanvasJson,
  strokesToInkML,
  type CanvasSize,
  type InkJson,
  type InkPoint,
  type InkStroke,
} from "@/lib/inkml";

/**
 * Raised when a trace would exceed the contract's point budget (api.md §1).
 *
 * The message is English because it is developer-facing: what the lecturer reads is
 * `workspace.inkTooLong` from the catalogue, interpolated with {@link maxPoints}.
 */
export class InkTooLargeError extends Error {
  /** The budget that was exhausted, so the UI can name it without importing the constant. */
  readonly maxPoints: number;

  constructor(maxPoints: number) {
    super(`Ink trace exhausted its point budget of ${String(maxPoints)} points.`);
    this.name = "InkTooLargeError";
    this.maxPoints = maxPoints;
  }
}

/** Options for {@link StrokeBuffer}. */
export interface StrokeBufferOptions {
  /** Canvas size in CSS pixels; updated by the component on resize. */
  canvas?: CanvasSize;
  /** Minimum squared pixel distance between kept samples (cheap sampling filter). */
  minDistance?: number;
  /** Point budget; defaults to the contract limit. */
  maxPoints?: number;
}

const DEFAULT_CANVAS: CanvasSize = { w: 800, h: 300 };

/**
 * Collects pointer samples into traces.
 *
 * Usage: `push()` while the pen is down, `endStroke()` on pen-up, then one of the serialisers.
 */
export class StrokeBuffer {
  private strokes: InkStroke[] = [];
  private current: InkPoint[] = [];
  private originT: number | null = null;
  private canvasSize: CanvasSize;
  private readonly minDistance: number;
  private readonly maxPoints: number;

  constructor(options: StrokeBufferOptions = {}) {
    this.canvasSize = options.canvas ?? DEFAULT_CANVAS;
    this.minDistance = options.minDistance ?? 0;
    this.maxPoints = options.maxPoints ?? MAX_INK_POINTS;
  }

  /** Number of completed traces (an unfinished stroke is not counted). */
  get strokeCount(): number {
    return this.strokes.length;
  }

  /** True when nothing has been drawn. */
  get isEmpty(): boolean {
    return this.strokes.length === 0 && this.current.length === 0;
  }

  /** Update the drawing surface size recorded in the serialised payloads. */
  setCanvasSize(size: CanvasSize): void {
    this.canvasSize = size;
  }

  /**
   * Append one pointer sample to the stroke in progress.
   *
   * The first sample ever seen defines t = 0. Samples closer than `minDistance` to the previous one
   * are dropped.
   *
   * @throws {InkTooLargeError} When the point budget is exhausted.
   */
  push(point: { x: number; y: number; t: number }): void {
    if (countPoints(this.strokes) + this.current.length >= this.maxPoints) {
      throw new InkTooLargeError(this.maxPoints);
    }
    this.originT ??= point.t;
    const last = this.current[this.current.length - 1];
    const relative: InkPoint = { x: point.x, y: point.y, t: point.t - this.originT };
    if (last !== undefined && this.minDistance > 0) {
      const dx = relative.x - last.x;
      const dy = relative.y - last.y;
      if (dx * dx + dy * dy < this.minDistance * this.minDistance) {
        return;
      }
    }
    this.current.push(relative);
  }

  /** Close the stroke in progress (pen-up). A stroke with no samples is discarded. */
  endStroke(): void {
    if (this.current.length > 0) {
      this.strokes.push({ points: this.current });
      this.current = [];
    }
  }

  /** All completed traces (defensive copy; the stroke in progress is excluded). */
  getStrokes(): InkStroke[] {
    return this.strokes.map((stroke) => ({ points: [...stroke.points] }));
  }

  /** Serialise as W3C InkML — one `<trace>` per completed stroke. */
  toInkML(): string {
    return strokesToInkML(this.strokes, this.canvasSize);
  }

  /** Serialise as the compact Canvas JSON payload of `POST /ocr/recognize`. */
  toCanvasJson(): InkJson {
    return strokesToCanvasJson(this.strokes, this.canvasSize);
  }

  /**
   * Rasterise the traces to a PNG blob (offline OCR path, ADR-0013 option 1).
   *
   * @param scale - Pixel scale factor (2 gives a retina-quality raster).
   * @returns The PNG bytes.
   * @throws {Error} When the browser exposes no 2D canvas context.
   */
  async toPNG(scale = 2): Promise<Blob> {
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(this.canvasSize.w * scale));
    canvas.height = Math.max(1, Math.round(this.canvasSize.h * scale));
    const ctx = canvas.getContext("2d");
    if (ctx === null) {
      throw new Error("This browser exposes no 2D canvas context.");
    }
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawStrokes(ctx, this.strokes, { scale });
    return await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob === null) {
          reject(new Error("Canvas.toBlob produced no PNG."));
          return;
        }
        resolve(blob);
      }, "image/png");
    });
  }

  /** Drop every trace and reset the time origin. */
  clear(): void {
    this.strokes = [];
    this.current = [];
    this.originT = null;
  }
}
