/**
 * "This is what I heard, and this is what I am about to run."
 *
 * The proposal's no-hallucination guarantee rests on the lecturer seeing the parsed command before
 * the CAS executes it, and api.md §3.6 makes the step mandatory on the voice path. In front of a
 * class a wrong command is far more expensive than a slow one, so the default is to ask — and the
 * panel offers the escape hatch (`Sesli komutlarda bir daha sorma`) that only ever applies above
 * the confidence threshold (`voicePreference.ts`).
 *
 * The panel is inline, never a modal: it appears above the result, the previous answer stays
 * visible behind it, and Escape dismisses it without navigating anywhere.
 *
 * It also closes the latency measurement when it renders a voice outcome: at that instant the
 * system has finished its work and the lecturer is looking at what it understood, so the seconds
 * they spend deciding are not charged to the contractual metric (ADR-0014).
 */
import { useEffect, useRef } from "react";

import type { CommandDto } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Chip } from "@/components/ui/Chip";
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import { formatCommand } from "@/lib/commandFormat";
import { MathView } from "@/features/results/MathView";
import { useRenderedLatency } from "@/features/results/useRenderedLatency";
import { CONFIRMATION_THRESHOLD } from "./voicePreference";
import styles from "./CommandPreview.module.css";

export { CONFIRMATION_THRESHOLD };

export interface CommandPreviewProps {
  /** The recognised input in LaTeX. */
  latex: string;
  /** The typed CAS command the backend intends to run. */
  command: CommandDto;
  /** NLU/ASR confidence in `[0, 1]`. */
  confidence: number;
  /** Run the command as shown. */
  onConfirm: () => void;
  /** Return to the command bar for editing. */
  onEdit: () => void;
  /** Voice input: shows the transcript and the "always confirmed" notice. */
  voice?: boolean;
  /** What the ASR heard, shown before the answer (api.md §3.6). */
  transcript?: string | null;
  /** Intent the NLU settled on, for the lecturer who wants to know why. */
  intent?: string | null;
  /** Current value of the "ask me every time" preference. */
  alwaysAsk?: boolean;
  /** Change the preference; the checkbox appears only when this is given. */
  onAlwaysAskChange?: (value: boolean) => void;
  /** Provisional latency key of the outcome being confirmed. */
  latencyKey?: string | null;
  /** Server operation id of the outcome being confirmed. */
  operationId?: string | null;
}

/** The confirm/edit panel. */
export function CommandPreview({
  latex,
  command,
  confidence,
  onConfirm,
  onEdit,
  voice = false,
  transcript = null,
  intent = null,
  alwaysAsk = true,
  onAlwaysAskChange,
  latencyKey = null,
  operationId = null,
}: CommandPreviewProps): JSX.Element {
  const lowConfidence = confidence < CONFIRMATION_THRESHOLD;
  const actionsRef = useRef<HTMLDivElement | null>(null);

  useRenderedLatency(latencyKey, operationId);

  useEffect(() => {
    // Move focus to the safe action so Enter confirms and Escape (handled by the page) cancels.
    // Focus is *moved*, never trapped: Tab still leaves the panel for the command bar.
    actionsRef.current?.querySelector("button")?.focus();
  }, []);

  return (
    <section className={styles.preview} aria-label={t("commandPreview.title")} role="group">
      <h2 className={styles.title}>{t("commandPreview.title")}</h2>
      <p className={styles.description}>{t("commandPreview.description")}</p>

      {transcript === null || transcript === "" ? null : (
        <p className={styles.transcript} data-testid="command-preview-transcript">
          <span className={styles.label}>{t("commandPreview.transcript")}</span>
          <span className={styles.transcriptText}>“{transcript}”</span>
        </p>
      )}

      <div className={styles.grid}>
        {latex === "" ? null : (
          <>
            <span className={styles.label}>{t("commandPreview.latex")}</span>
            <MathView latex={latex} />
          </>
        )}

        <span className={styles.label}>{t("commandPreview.command")}</span>
        <code className={styles.command} data-testid="command-preview-command">
          {formatCommand(command)}
        </code>

        {intent === null || intent === "" ? null : (
          <>
            <span className={styles.label}>{t("commandPreview.intent")}</span>
            <span className={styles.intent}>{intent}</span>
          </>
        )}

        <span className={styles.label}>{t("commandPreview.confidence")}</span>
        <span>
          <Chip tone={lowConfidence ? "warning" : "success"} large>
            {`%${String(Math.round(confidence * 100))}`}
          </Chip>
        </span>
      </div>

      {lowConfidence ? <p className={styles.notice}>{t("commandPreview.lowConfidence")}</p> : null}
      {voice ? <p className={styles.notice}>{t("commandPreview.voiceNotice")}</p> : null}

      <div className={styles.actions} ref={actionsRef}>
        <Button variant="primary" size="lg" onClick={onConfirm}>
          {t("commandPreview.confirm")}
        </Button>
        <Button variant="secondary" size="lg" onClick={onEdit}>
          {t("commandPreview.edit")}
        </Button>
      </div>

      {onAlwaysAskChange === undefined ? null : (
        <label className={cx(styles.skip, "chrome-secondary")}>
          <input
            type="checkbox"
            checked={!alwaysAsk}
            onChange={(event) => {
              onAlwaysAskChange(!event.target.checked);
            }}
          />
          <span>
            {t("commandPreview.skipNext")}
            <span className={styles.skipHint}> {t("commandPreview.skipHint")}</span>
          </span>
        </label>
      )}
    </section>
  );
}
