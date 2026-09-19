/**
 * The automatic quality verdict of one take (`RecordingOut.qc`).
 *
 * A rejected recording arrives as a **200** with `accepted:false` (api.md §2) — it is a verdict,
 * not an error — so it is rendered as one: the Turkish `reasons_tr` sentences the backend composed
 * (`aicls.core.messages_tr.MSG_QC_*`) in full size, the measurements underneath for anyone who
 * wants them, and a plain statement that the volunteer may simply read the same sentence again.
 * The audio of a rejected take is never written on the server, so a retry costs nothing.
 *
 * The panel is a polite live region: the verdict is announced without moving focus, which matters
 * because the volunteer's hands are on the record button, not on the keyboard.
 */
import type { QcCheckOut, RecordingOut } from "@/api/types";
import { Chip } from "@/components/ui";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import styles from "./QcVerdict.module.css";

/** Turkish label for each `aicls.asr.qc.QcCheck` name; an unknown name shows its raw key. */
const CHECK_LABELS: Record<string, MessageKey> = {
  duration_min: "contribute.check.durationMin",
  duration_max: "contribute.check.durationMax",
  clipping_ratio: "contribute.check.clippingRatio",
  snr_db: "contribute.check.snrDb",
  silence_ratio: "contribute.check.silenceRatio",
  dc_offset: "contribute.check.dcOffset",
};

function checkLabel(name: string): string {
  const key = CHECK_LABELS[name];
  return key === undefined ? name : t(key);
}

/** Format a measurement so small ratios stay readable next to durations in seconds. */
function formatValue(value: number): string {
  if (!Number.isFinite(value)) {
    return t("common.unknown");
  }
  const magnitude = Math.abs(value);
  const digits = magnitude >= 10 ? 1 : magnitude >= 1 ? 2 : 4;
  return value.toFixed(digits);
}

export interface QcVerdictProps {
  /** The upload response. */
  recording: RecordingOut;
  /** Delete this (accepted) recording; omitted when there is nothing to delete. */
  onDelete?: () => void;
  /** Label of the delete action. */
  deleteLabel?: string;
}

/** The verdict panel. */
export function QcVerdict({ recording, onDelete, deleteLabel }: QcVerdictProps): JSX.Element {
  const checks: QcCheckOut[] = recording.qc.checks ?? [];
  const reasons: string[] = recording.qc.reasons_tr ?? [];
  const accepted = recording.accepted;

  return (
    <div
      className={cx(styles.verdict, accepted ? styles.accepted : styles.rejected)}
      data-testid="qc-verdict"
      role="status"
      aria-live="polite"
    >
      <p className={styles.headline}>
        <Chip tone={accepted ? "ok" : "warn"} large>
          {accepted ? t("contribute.accepted") : t("contribute.rejected")}
        </Chip>
      </p>

      {accepted ? (
        <p className={styles.hint}>{t("contribute.qcAcceptedHint")}</p>
      ) : (
        <>
          <p className={styles.hint}>{t("contribute.qcRetryHint")}</p>
          {reasons.length === 0 ? null : (
            <ul className={styles.reasons}>
              {reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          )}
        </>
      )}

      {checks.length === 0 ? null : (
        <details className={styles.details}>
          <summary>{t("contribute.checksTitle")}</summary>
          <div className="scroll-x">
            <table className={styles.table}>
              <caption className="visually-hidden">{t("contribute.qcTitle")}</caption>
              <thead>
                <tr>
                  <th scope="col">{t("contribute.checkName")}</th>
                  <th scope="col">{t("contribute.checkValue")}</th>
                  <th scope="col">{t("contribute.checkThreshold")}</th>
                  <th scope="col">{t("contribute.checkStatus")}</th>
                </tr>
              </thead>
              <tbody>
                {checks.map((check) => (
                  <tr key={check.name}>
                    <th scope="row">{checkLabel(check.name)}</th>
                    <td>{formatValue(check.value)}</td>
                    <td>{formatValue(check.threshold)}</td>
                    <td>
                      <Chip tone={check.passed ? "ok" : "critical"}>
                        {check.passed ? t("contribute.checkPassed") : t("contribute.checkFailed")}
                      </Chip>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {onDelete === undefined || !accepted ? null : (
        <button type="button" className={styles.delete} onClick={onDelete}>
          {deleteLabel ?? t("contribute.deleteRecording")}
        </button>
      )}
    </div>
  );
}
