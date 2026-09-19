/**
 * Handwriting surface (Canvas API, proposal A.1.2).
 *
 * Pointer events — mouse, pen and touch through the unified Pointer Events API — are sampled into a
 * {@link StrokeBuffer}. Each pen-up closes a trace and notifies the parent, which forwards the ink
 * over `/ws/v1/ink` (streaming recognition, ADR-0014) or posts it to `POST /ocr/recognize`.
 *
 * Drawing is incremental: `pointerdown` opens a path, `pointermove` extends it. A full redraw only
 * happens on clear or resize, so a long expression stays smooth on a tablet.
 */
import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

import { Button } from "@/components/ui/Button";
import { t } from "@/i18n";
import { drawStrokes } from "@/lib/inkml";
import { InkTooLargeError, StrokeBuffer } from "./StrokeBuffer";
import styles from "./InkCanvas.module.css";

export interface InkCanvasProps {
  /** Logical canvas width in CSS pixels. */
  width?: number;
  /** Logical canvas height in CSS pixels. */
  height?: number;
  /** Block drawing (e.g. while a recognition is in flight). */
  disabled?: boolean;
  /** Called after every pen-up with the buffer holding all traces so far. */
  onStrokeEnd?: (buffer: StrokeBuffer) => void;
  /** Called when the user asks for recognition explicitly. */
  onRecognize?: (buffer: StrokeBuffer) => void;
  /** Called after the canvas is cleared. */
  onClear?: () => void;
  /**
   * Change this number to clear the canvas from outside (the `Alt + T` shortcut).
   *
   * A signal prop rather than an imperative handle: the page already owns the shortcut table, and
   * a number in the render tree is far easier to reason about than a ref that may be null.
   */
  clearToken?: number;
}

const DEFAULT_WIDTH = 800;
const DEFAULT_HEIGHT = 260;

/** Canvas plus its small toolbar. */
export function InkCanvas({
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  disabled = false,
  onStrokeEnd,
  onRecognize,
  onClear,
  clearToken = 0,
}: InkCanvasProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bufferRef = useRef<StrokeBuffer>(new StrokeBuffer({ canvas: { w: width, h: height } }));
  const drawingRef = useRef(false);
  const onClearRef = useRef(onClear);
  onClearRef.current = onClear;
  const [strokeCount, setStrokeCount] = useState(0);
  const [warning, setWarning] = useState<string | null>(null);
  const hintId = useId();

  const context = useCallback((): CanvasRenderingContext2D | null => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return null;
    }
    return canvas.getContext("2d");
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = context();
    if (canvas === null || ctx === null) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawStrokes(ctx, bufferRef.current.getStrokes());
  }, [context]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) {
      return;
    }
    const ratio = globalThis.devicePixelRatio ?? 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx?.setTransform(ratio, 0, 0, ratio, 0, 0);
    bufferRef.current.setCanvasSize({ w: width, h: height });
    redraw();
  }, [width, height, redraw]);

  const clear = useCallback((): void => {
    bufferRef.current.clear();
    setStrokeCount(0);
    setWarning(null);
    redraw();
    onClearRef.current?.();
  }, [redraw]);

  // The first render is not a clear request: only a *change* of the token wipes the board, so a
  // remount never throws away what the lecturer just wrote.
  const seenClearToken = useRef(clearToken);
  useEffect(() => {
    if (seenClearToken.current === clearToken) {
      return;
    }
    seenClearToken.current = clearToken;
    clear();
  }, [clearToken, clear]);

  const positionOf = (event: ReactPointerEvent<HTMLCanvasElement>): { x: number; y: number } => {
    const rect = event.currentTarget.getBoundingClientRect();
    const scaleX = rect.width === 0 ? 1 : width / rect.width;
    const scaleY = rect.height === 0 ? 1 : height / rect.height;
    return { x: (event.clientX - rect.left) * scaleX, y: (event.clientY - rect.top) * scaleY };
  };

  const pushPoint = (
    event: ReactPointerEvent<HTMLCanvasElement>,
  ): { x: number; y: number } | null => {
    const point = positionOf(event);
    try {
      bufferRef.current.push({ ...point, t: performance.now() });
    } catch (error) {
      if (error instanceof InkTooLargeError) {
        drawingRef.current = false;
        // The exception's own message is English (developer-facing); what the lecturer reads comes
        // from the catalogue like every other string in the interface.
        setWarning(t("workspace.inkTooLong", { max: error.maxPoints }));
        return null;
      }
      throw error;
    }
    return point;
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>): void => {
    if (disabled) {
      return;
    }
    const point = pushPoint(event);
    if (point === null) {
      return;
    }
    drawingRef.current = true;
    if (typeof event.currentTarget.setPointerCapture === "function") {
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch {
        // Pointer capture is best effort; drawing works without it.
      }
    }
    const ctx = context();
    if (ctx !== null) {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#12303a";
      ctx.beginPath();
      ctx.moveTo(point.x, point.y);
    }
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>): void => {
    if (!drawingRef.current || disabled) {
      return;
    }
    const point = pushPoint(event);
    if (point === null) {
      return;
    }
    const ctx = context();
    if (ctx !== null) {
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
    }
  };

  const endStroke = (): void => {
    if (!drawingRef.current) {
      return;
    }
    drawingRef.current = false;
    bufferRef.current.endStroke();
    setStrokeCount(bufferRef.current.strokeCount);
    onStrokeEnd?.(bufferRef.current);
  };

  return (
    <div className={styles.wrapper}>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        style={{ aspectRatio: `${String(width)} / ${String(height)}` }}
        role="img"
        aria-label={t("workspace.canvasLabel")}
        aria-describedby={hintId}
        data-testid="ink-canvas"
        data-disabled={disabled ? "true" : "false"}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endStroke}
        onPointerCancel={endStroke}
        onPointerLeave={endStroke}
      />
      <div className={styles.toolbar}>
        <Button
          variant="primary"
          size="small"
          disabled={disabled || strokeCount === 0}
          onClick={() => {
            onRecognize?.(bufferRef.current);
          }}
        >
          {t("workspace.recognize")}
        </Button>
        <Button variant="secondary" size="small" disabled={strokeCount === 0} onClick={clear}>
          {t("workspace.clear")}
        </Button>
        <span className={styles.count} data-testid="stroke-count">
          {t("workspace.strokeCount", { count: strokeCount })}
        </span>
      </div>
      <p id={hintId} className={styles.hint} {...(warning === null ? {} : { role: "alert" })}>
        {warning ?? t("workspace.canvasHint")}
      </p>
    </div>
  );
}
