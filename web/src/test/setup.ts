/**
 * Vitest setup shared by every unit test.
 *
 * jsdom implements neither the canvas 2D context nor `HTMLCanvasElement.toBlob`, and installing the
 * native `canvas` package would add a compiled dependency to CI for no benefit: the components under
 * test are judged by the data they produce (InkML, Canvas JSON, WAV bytes), not by pixels. Both are
 * therefore replaced with plain no-op stubs — deliberately **not** `vi.fn()`, because the suite runs
 * with `restoreMocks`, which would strip their implementations after the first test. Pixel-level
 * behaviour is covered by the Playwright specs in `e2e/`, which run in real browsers.
 */
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

const noop = (): void => undefined;

/** A 2D context stub that satisfies every call the canvas code makes. */
function createContextStub(): CanvasRenderingContext2D {
  const stub = {
    lineCap: "butt",
    lineJoin: "miter",
    lineWidth: 1,
    strokeStyle: "#000000",
    fillStyle: "#000000",
    beginPath: noop,
    closePath: noop,
    moveTo: noop,
    lineTo: noop,
    stroke: noop,
    fill: noop,
    fillRect: noop,
    clearRect: noop,
    save: noop,
    restore: noop,
    scale: noop,
    translate: noop,
    setTransform: noop,
    drawImage: noop,
  };
  return stub as unknown as CanvasRenderingContext2D;
}

HTMLCanvasElement.prototype.getContext = function getContext(): CanvasRenderingContext2D {
  return createContextStub();
} as unknown as HTMLCanvasElement["getContext"];

HTMLCanvasElement.prototype.toBlob = function toBlob(callback: BlobCallback): void {
  callback(new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], { type: "image/png" }));
};

/**
 * jsdom has no `PointerEvent`, so Testing Library would fall back to a bare `Event` and drop
 * `clientX`/`clientY`. A `MouseEvent` subclass restores the coordinate fields the ink canvas reads —
 * the same interface the browser gives us for mouse, pen and touch.
 */
class PointerEventPolyfill extends MouseEvent {
  readonly pointerId: number;
  readonly pointerType: string;
  readonly pressure: number;
  readonly isPrimary: boolean;

  constructor(type: string, params: PointerEventInit = {}) {
    super(type, params);
    this.pointerId = params.pointerId ?? 1;
    this.pointerType = params.pointerType ?? "mouse";
    this.pressure = params.pressure ?? 0.5;
    this.isPrimary = params.isPrimary ?? true;
  }
}

if (typeof globalThis.PointerEvent === "undefined") {
  (globalThis as unknown as { PointerEvent: unknown }).PointerEvent = PointerEventPolyfill;
}

afterEach(() => {
  cleanup();
});
