/**
 * The result area — the visual centre of gravity of the whole screen.
 *
 * Everything about it is decided by the back row of a lecture hall: the answer is the largest
 * element on the board, it is announced through a polite live region rather than replacing itself
 * silently, the `plain` form the API returns sits under it as readable text (so the answer is not
 * only a picture of an equation), and the honest metadata — cache hit, stage timings, which model
 * produced the recognition — lives in the footer where it can be pointed at but never shouts.
 *
 * Two rules hold even when things go wrong:
 *
 * - **The previous answer stays.** `busy` shows a quiet marker in the header; it never blanks the
 *   card. A failure is rendered *next to* this card, not instead of it (`ProblemNotice`).
 * - **A stub is never mistaken for a trained model.** `ModelBadge` prints the Turkish warning
 *   whenever `model_versions` names a stand-in (`lib/modelVersions.ts`).
 *
 * The card also closes the contractual latency measurement the moment the answer is committed to
 * the DOM (`useRenderedLatency`, ADR-0014).
 */
import { useCallback, useState } from "react";

import type { ComputeOut, ResultKind } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Chip } from "@/components/ui/Chip";
import { ModelBadge } from "@/components/ui/ModelBadge";
import { Spinner } from "@/components/ui/Spinner";
import { StageTimings } from "@/components/ui/StageTimings";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import type { StageTiming } from "@/lib/serverTiming";
import { MathView } from "./MathView";
import { PlotView } from "./PlotView";
import { ResultFeedback } from "./ResultFeedback";
import { StepsView } from "./StepsView";
import { exportOperation, type ExportFormat } from "./exportOperation";
import { useRenderedLatency } from "./useRenderedLatency";
import styles from "./ResultCard.module.css";

/** Turkish name of each CAS result kind (`mathengine.Result.kind`). */
const KIND_LABEL: Record<ResultKind, MessageKey> = {
  expr: "results.kind.expr",
  number: "results.kind.number",
  set: "results.kind.set",
  bool: "results.kind.bool",
  plot: "results.kind.plot",
  steps: "results.kind.steps",
};

/** Turkish name of each client-side action the NLU can emit (`nlu.UiActionName`). */
const UI_ACTION_LABEL: Record<string, MessageKey> = {
  clear: "results.uiAction.clear",
  undo: "results.uiAction.undo",
  repeat_last: "results.uiAction.repeatLast",
  export: "results.uiAction.export",
};

export interface ResultCardProps {
  /** The compute outcome to display; `null` before the first command. */
  outcome: ComputeOut | null;
  /** `Server-Timing` stages of the request that produced it. */
  stages?: readonly StageTiming[];
  /** Provisional latency key opened at submit (see {@link useRenderedLatency}). */
  latencyKey?: string | null;
  /** True while the next command is in flight; the answer below stays on screen. */
  busy?: boolean;
  /** Workspace id, required for the export buttons. */
  workspaceId?: string | null;
}

/** Copy text to the clipboard, reporting whether the browser allowed it. */
async function copyText(value: string): Promise<boolean> {
  const clipboard = globalThis.navigator?.clipboard;
  if (clipboard === undefined || typeof clipboard.writeText !== "function") {
    return false;
  }
  try {
    await clipboard.writeText(value);
    return true;
  } catch {
    return false;
  }
}

/** Render one compute outcome and report its render latency. */
export function ResultCard({
  outcome,
  stages,
  latencyKey = null,
  busy = false,
  workspaceId = null,
}: ResultCardProps): JSX.Element {
  const [copied, setCopied] = useState<string | null>(null);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exported, setExported] = useState(false);

  const result = outcome?.result ?? null;
  const operationId = outcome?.operation_id ?? null;
  useRenderedLatency(latencyKey, operationId, result !== null);

  const copy = useCallback(async (value: string, badge: string): Promise<void> => {
    const ok = await copyText(value);
    setCopied(ok ? badge : t("results.copyFailed"));
    globalThis.setTimeout(() => {
      setCopied(null);
    }, 2500);
  }, []);

  const runExport = useCallback(
    async (format: ExportFormat): Promise<void> => {
      if (workspaceId === null || operationId === null) {
        return;
      }
      setExporting(format);
      setExportError(null);
      setExported(false);
      const problem = await exportOperation(workspaceId, operationId, format);
      setExporting(null);
      setExportError(problem?.detail_tr ?? null);
      setExported(problem === null);
    },
    [operationId, workspaceId],
  );

  const header = (
    <div className={styles.badges}>
      {busy ? (
        <span className={styles.busy}>
          <Spinner size="sm" inline />
          <span>{t("results.busy")}</span>
        </span>
      ) : null}
      {result === null ? null : <Chip tone="accent">{t(KIND_LABEL[result.kind])}</Chip>}
      {result?.cache_hit === true ? <Chip tone="success">{t("results.cacheHit")}</Chip> : null}
    </div>
  );

  if (outcome === null) {
    return (
      <Card title={t("results.title")} actions={header} emphasis className={cx(styles.card)}>
        <p className={styles.empty}>{t("results.empty")}</p>
      </Card>
    );
  }

  const canExport = workspaceId !== null && operationId !== null;

  return (
    <Card title={t("results.title")} actions={header} emphasis className={cx(styles.card)}>
      <div className={styles.body} data-testid="result-card">
        {/*
          The stand-in warning goes *above* the answer, not under it. A result with steps and a
          plot is taller than the board, and a warning below all of that is a warning a room can
          scroll past — in a lecture, and in a screenshot that ends up in a progress report. It
          renders nothing at all when every model that answered is a promoted checkpoint, so a
          normal result gains no line; the footer badge still lists the versions.
        */}
        <ModelBadge versions={outcome.model_versions} warningOnly />

        {outcome.transcript === null || outcome.transcript === undefined ? null : (
          <p className={styles.transcript}>
            <span className={styles.label}>{t("results.transcript")}</span>
            <span className={styles.transcriptText}>“{outcome.transcript}”</span>
          </p>
        )}

        {outcome.input_latex === null || outcome.input_latex === undefined ? null : (
          <div className={styles.input}>
            <span className={styles.label}>{t("results.input")}</span>
            <MathView latex={outcome.input_latex} />
          </div>
        )}

        <div className={styles.answer} aria-live="polite" aria-atomic="true">
          {result === null ? (
            <p className={styles.note}>
              {outcome.kind === "book"
                ? t("results.bookAnswer")
                : outcome.kind === "ui"
                  ? t("results.uiAction", { action: uiActionLabel(outcome) })
                  : t("results.empty")}
            </p>
          ) : (
            <>
              <span className="visually-hidden">{t("results.result")}: </span>
              <div className={styles.latex} data-testid="result-latex">
                <MathView latex={result.latex} display />
              </div>
              <p className={styles.plain} data-testid="result-plain">
                {result.plain}
              </p>
            </>
          )}
        </div>

        {result === null ? null : (
          <div className={styles.actions}>
            <Button
              variant="secondary"
              onClick={() => {
                void copy(result.plain, t("results.copied"));
              }}
            >
              {t("results.copyPlain")}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                void copy(result.latex, t("results.copied"));
              }}
            >
              {t("results.copyLatex")}
            </Button>
            {canExport ? (
              <>
                <Button
                  variant="ghost"
                  loading={exporting === "tex"}
                  onClick={() => {
                    void runExport("tex");
                  }}
                >
                  {t("results.exportTex")}
                </Button>
                <Button
                  variant="ghost"
                  loading={exporting === "pdf"}
                  onClick={() => {
                    void runExport("pdf");
                  }}
                >
                  {t("results.exportPdf")}
                </Button>
              </>
            ) : null}
            <span className={styles.copied} role="status">
              {exporting !== null
                ? t("results.exporting")
                : exported
                  ? t("results.exportDone")
                  : (copied ?? "")}
            </span>
          </div>
        )}

        {exportError === null ? null : (
          <p className={styles.exportError} role="alert">
            {exportError}
          </p>
        )}

        {result === null || (result.steps ?? []).length === 0 ? null : (
          <StepsView steps={result.steps ?? []} />
        )}

        {result !== null && result.plot_url !== null && result.plot_url !== undefined ? (
          <PlotView url={result.plot_url} alt={t("plot.altFor", { plain: result.plain })} />
        ) : null}

        <div className={styles.footer}>
          {/* The version list, without repeating the warning stated above the answer. Neither is
              secondary chrome: presentation mode is exactly when a demonstration could be mistaken
              for a trained system, so both stay on the wall. */}
          <ModelBadge versions={outcome.model_versions} showWarning={false} />
          {outcome.nlu === null || outcome.nlu === undefined ? null : (
            <p className={cx(styles.intent, "chrome-secondary")}>
              <span className={styles.label}>{t("results.intent")}</span>{" "}
              <span className={styles.intentValue}>{outcome.nlu.intent}</span>
            </p>
          )}
          <div className={cx(styles.timings, "chrome-secondary")}>
            <StageTimings stages={stages ?? []} timings={outcome.timings} />
          </div>
        </div>

        <ResultFeedback outcome={outcome} />
      </div>
    </Card>
  );
}

/** Turkish label of a `kind: "ui"` outcome's action, falling back to the raw name. */
function uiActionLabel(outcome: ComputeOut): string {
  const action = outcome.command?.action;
  if (typeof action !== "string") {
    return "—";
  }
  const key = UI_ACTION_LABEL[action];
  return key === undefined ? action : t(key);
}
