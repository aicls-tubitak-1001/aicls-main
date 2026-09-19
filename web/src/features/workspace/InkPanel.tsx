/**
 * Handwriting: the canvas, the live recognition and the confirmation that stands between them.
 *
 * The stream is the point (ADR-0014). Every pen-up ships the completed trace over `/ws/v1/ink`, the
 * server debounces and answers `partial`, and `finalize` — or 1.5 s of idle — produces `final` with
 * alternatives and a `recognition_id`. Recognition therefore overlaps writing instead of following
 * it, which is what keeps "command → result" inside the contractual three seconds.
 *
 * What the lecturer sees is never the raw stream:
 *
 * - the recognised LaTeX is **always shown for confirmation** before it becomes a command, with the
 *   confidence spelled out as a percentage — a low number is the single most useful warning the
 *   OCR layer can give while the models are still stubs;
 * - the reading is editable, and an edit is worth data: with a `recognition_id` in hand it is sent
 *   to `POST /ocr/feedback` as a corrected sample (İP-3's correction loop);
 * - alternatives are offered as one-tap chips rather than hidden behind a menu.
 *
 * When the socket is not available — signed out, offline, a refused ticket — the same canvas still
 * works through the one-shot `POST /ocr/recognize` with the Canvas JSON of ADR-0013.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { api, call } from "@/api/client";
import { transportProblem, type AiclsProblem } from "@/api/problem";
import { useInkSocket } from "@/api/sockets";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import type { InkJson } from "@/lib/inkml";
import { MathView } from "@/features/results/MathView";
import { InkCanvas } from "./InkCanvas";
import type { StrokeBuffer } from "./StrokeBuffer";
import styles from "./InkPanel.module.css";

/** Turkish label of each socket state (`api/ws.ts::SocketStatus`). */
const STATUS_LABEL: Record<string, MessageKey> = {
  connecting: "ws.connecting",
  open: "ws.connected",
  reconnecting: "ws.reconnecting",
  closed: "ws.closed",
};

/** One reading of the board, from the stream or from the one-shot endpoint. */
interface Recognition {
  latex: string;
  confidence: number | null;
  alternatives: readonly [string, number][];
  recognitionId: string | null;
  /** True while this is only a partial reading (the pen may still be moving). */
  partial: boolean;
}

export interface InkPanelProps {
  /** Open the `/ws/v1/ink` stream (a signed-in session on the workspace screen). */
  streamEnabled?: boolean;
  /** Workspace the one-shot recognition belongs to. */
  workspaceId?: string | null;
  /** Accept the reading: the LaTeX becomes the expression on the board. */
  onAccept: (latex: string) => void;
  /** Report a failure to the page so it lands in the one notice area. */
  onProblem?: (problem: AiclsProblem) => void;
  /** Change this number to clear the canvas from the shortcut table. */
  clearToken?: number;
  /** Render over the result instead of beside it. */
  overlay?: boolean;
  /** Toggle the overlay/beside placement. */
  onToggleOverlay?: () => void;
  /** Close the panel. */
  onClose?: () => void;
}

/**
 * Put the Canvas JSON document into the request body.
 *
 * `openapi-typescript` renders the endpoint's untyped `ink` object as `Record<string, never>`
 * (FastAPI documents it as a free-form object), so this is the single place that admits the payload
 * is really an {@link InkJson} — the shape `aicls.ocr.ink_from_canvas_json` parses.
 */
function inkBody(json: InkJson): Record<string, never> {
  return json as unknown as Record<string, never>;
}

/** Percentage form of a confidence in `[0, 1]`. */
function percent(confidence: number | null): number {
  return Math.round((confidence ?? 0) * 100);
}

/** Canvas, live recognition, confirmation and the OCR correction loop. */
export function InkPanel({
  streamEnabled = false,
  workspaceId = null,
  onAccept,
  onProblem,
  clearToken = 0,
  overlay = false,
  onToggleOverlay,
  onClose,
}: InkPanelProps): JSX.Element {
  const ink = useInkSocket({ enabled: streamEnabled });
  const [recognition, setRecognition] = useState<Recognition | null>(null);
  const [draft, setDraft] = useState("");
  const [edited, setEdited] = useState(false);
  const [correctionSent, setCorrectionSent] = useState(false);
  const [recognizing, setRecognizing] = useState(false);
  const bufferRef = useRef<StrokeBuffer | null>(null);
  const onProblemRef = useRef(onProblem);
  onProblemRef.current = onProblem;

  const applyRecognition = useCallback((next: Recognition): void => {
    setRecognition(next);
    setCorrectionSent(false);
    // A partial reading keeps overwriting the draft; once the lecturer edits it, it is theirs.
    setEdited((wasEdited) => {
      if (!wasEdited) {
        setDraft(next.latex);
      }
      return wasEdited;
    });
  }, []);

  useEffect(() => {
    if (ink.partialLatex !== null) {
      applyRecognition({
        latex: ink.partialLatex,
        confidence: ink.confidence,
        alternatives: [],
        recognitionId: null,
        partial: true,
      });
    }
  }, [ink.partialLatex, ink.confidence, applyRecognition]);

  useEffect(() => {
    if (ink.finalLatex !== null) {
      applyRecognition({
        latex: ink.finalLatex,
        confidence: ink.confidence,
        alternatives: ink.alternatives,
        recognitionId: ink.recognitionId,
        partial: false,
      });
    }
  }, [ink.finalLatex, ink.confidence, ink.alternatives, ink.recognitionId, applyRecognition]);

  useEffect(() => {
    if (ink.problem !== null) {
      onProblemRef.current?.(ink.problem);
    }
  }, [ink.problem]);

  /** One-shot recognition for the socketless path (`POST /ocr/recognize`, ADR-0013 option 2). */
  const recognizeOnce = useCallback(async (): Promise<void> => {
    const buffer = bufferRef.current;
    if (buffer === null || buffer.isEmpty) {
      return;
    }
    setRecognizing(true);
    const { data, problem } = await call(
      api.POST("/ocr/recognize", {
        body: {
          ink: inkBody(buffer.toCanvasJson()),
          ...(workspaceId === null || workspaceId === "" ? {} : { workspace_id: workspaceId }),
        },
      }),
      { fallbackDetailTr: t("errors.network") },
    );
    setRecognizing(false);
    if (data === null) {
      onProblemRef.current?.(problem ?? transportProblem(t("errors.network")));
      return;
    }
    applyRecognition({
      latex: data.latex,
      confidence: data.confidence,
      alternatives: data.alternatives ?? [],
      recognitionId: data.recognition_id ?? null,
      partial: false,
    });
  }, [applyRecognition, workspaceId]);

  const reset = useCallback((): void => {
    setRecognition(null);
    setDraft("");
    setEdited(false);
    setCorrectionSent(false);
  }, []);

  const sendCorrection = useCallback(async (): Promise<void> => {
    const recognitionId = recognition?.recognitionId ?? null;
    const corrected = draft.trim();
    if (recognitionId === null || corrected === "") {
      return;
    }
    const { problem } = await call(
      api.POST("/ocr/feedback", {
        body: { recognition_id: recognitionId, corrected_latex: corrected },
      }),
      { fallbackDetailTr: t("errors.network") },
    );
    if (problem !== null) {
      onProblemRef.current?.(problem);
      return;
    }
    setCorrectionSent(true);
  }, [draft, recognition]);

  const status = STATUS_LABEL[ink.status] ?? "ws.closed";
  const canCorrect = edited && recognition?.recognitionId !== null && !correctionSent;

  return (
    <Card
      title={t("workspace.penTitle")}
      className={cx(styles.card, overlay && styles.overlay)}
      actions={
        <div className={styles.headerActions}>
          <Chip
            tone={ink.status === "open" ? "ok" : "neutral"}
            dot
            title={t("workspace.socketLabel")}
          >
            {t(status)}
          </Chip>
          {ink.throttled ? <Chip tone="warn">{t("ws.throttled")}</Chip> : null}
          {onToggleOverlay === undefined ? null : (
            <Button variant="ghost" onClick={onToggleOverlay} aria-pressed={overlay}>
              {overlay ? t("workspace.penBeside") : t("workspace.penOverlay")}
            </Button>
          )}
          {onClose === undefined ? null : (
            <Button variant="ghost" onClick={onClose} aria-label={t("common.close")}>
              <span aria-hidden="true">×</span>
            </Button>
          )}
        </div>
      }
    >
      <div className={styles.panel}>
        <InkCanvas
          clearToken={clearToken}
          onStrokeEnd={(buffer) => {
            bufferRef.current = buffer;
            const strokes = buffer.getStrokes();
            const last = strokes[strokes.length - 1];
            if (last !== undefined) {
              ink.sendStroke(last, buffer.toCanvasJson().canvas);
            }
            ink.penUp();
          }}
          onRecognize={(buffer) => {
            bufferRef.current = buffer;
            if (ink.status === "open") {
              ink.finalize();
              return;
            }
            void recognizeOnce();
          }}
          onClear={() => {
            ink.clear();
            reset();
          }}
        />

        {recognition === null ? (
          <p className={styles.pending}>{t("workspace.recognitionPending")}</p>
        ) : (
          <div className={styles.recognition} data-testid="ink-recognition">
            <div className={styles.recognitionHead}>
              <span className={styles.label}>{t("workspace.recognitionTitle")}</span>
              <Chip tone={percent(recognition.confidence) < 70 ? "warn" : "ok"}>
                {t("workspace.recognitionConfidence", { percent: percent(recognition.confidence) })}
              </Chip>
              {recognition.partial ? (
                <Chip tone="neutral">{t("workspace.partialLabel")}</Chip>
              ) : null}
            </div>

            <div className={styles.preview}>
              <MathView latex={draft} display />
            </div>

            <label className="visually-hidden" htmlFor="ink-latex">
              {t("workspace.editInput")}
            </label>
            <input
              id="ink-latex"
              className={styles.latexInput}
              type="text"
              value={draft}
              spellCheck={false}
              onChange={(event) => {
                setDraft(event.target.value);
                setEdited(true);
              }}
            />

            {recognition.alternatives.length === 0 ? null : (
              <div className={styles.alternatives}>
                <span className={styles.label}>{t("workspace.alternatives")}</span>
                {recognition.alternatives.map(([latex, score]) => (
                  <button
                    key={latex}
                    type="button"
                    className={styles.alternative}
                    onClick={() => {
                      setDraft(latex);
                      setEdited(true);
                    }}
                  >
                    <MathView latex={latex} />
                    <span className={styles.score}>%{Math.round(score * 100)}</span>
                  </button>
                ))}
              </div>
            )}

            <div className={styles.actions}>
              <Button
                variant="primary"
                disabled={draft.trim() === ""}
                onClick={() => {
                  onAccept(draft.trim());
                }}
              >
                {t("workspace.useRecognized")}
              </Button>
              {canCorrect ? (
                <Button
                  variant="secondary"
                  onClick={() => {
                    void sendCorrection();
                  }}
                >
                  {t("workspace.saveInput")}
                </Button>
              ) : null}
              {recognizing ? (
                <span className={styles.status}>{t("workspace.recognizing")}</span>
              ) : null}
              {correctionSent ? (
                <span className={styles.sent} role="status">
                  {t("workspace.correctionSent")}
                </span>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
