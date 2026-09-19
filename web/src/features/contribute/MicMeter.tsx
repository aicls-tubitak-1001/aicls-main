/**
 * Input-level meter for the microphone.
 *
 * A volunteer cannot hear their own input gain, so the only honest way to prevent a wasted take is
 * to show the level *while* they speak and to name the two failure modes in Turkish: too loud
 * (clipping, which `aicls.asr.qc` refuses at a ratio of 0.001) and too quiet (which fails the SNR
 * check). The bar is `role="meter"` with its value in percent, and it is deliberately **not** a
 * live region: a value that changes ten times a second must never be announced.
 */
import { useId } from "react";

import { Chip } from "@/components/ui";
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import styles from "./MicMeter.module.css";

export interface MicMeterProps {
  /** Current input level in [0, 1]. */
  level: number;
  /** True while a take is being captured (the bar turns into the accent colour). */
  recording?: boolean;
  /** Clipped-sample ratio of the current take; above the limit the warning appears. */
  clippingRatio?: number;
  /** Show the "too quiet" hint (peak stayed under the quiet threshold during a take). */
  quiet?: boolean;
}

/** A horizontal level meter with its two Turkish warnings. */
export function MicMeter({
  level,
  recording = false,
  clippingRatio = 0,
  quiet = false,
}: MicMeterProps): JSX.Element {
  const labelId = useId();
  const clamped = Math.max(0, Math.min(1, level));
  const percent = Math.round(clamped * 100);
  const clipping = clippingRatio > 0;
  return (
    <div className={styles.meterBlock}>
      <div className={styles.row}>
        <span className={styles.label} id={labelId}>
          {t("contribute.levelLabel")}
        </span>
        <div
          className={cx(styles.track, recording && styles.recording, clipping && styles.clipping)}
          role="meter"
          aria-labelledby={labelId}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          aria-valuetext={t("contribute.levelValue", { percent })}
        >
          <span className={styles.fill} style={{ width: `${String(percent)}%` }} />
          <span className={styles.clipZone} aria-hidden="true" />
        </div>
        <span className={styles.percent}>{t("common.percent", { value: percent })}</span>
      </div>

      <p className={styles.hint}>{t("contribute.levelHint")}</p>

      {clipping ? (
        <p className={styles.warning} role="status">
          <Chip tone="warn">{t("contribute.check.clippingRatio")}</Chip>{" "}
          {t("contribute.clippingWarning")}
        </p>
      ) : null}
      {quiet && !clipping ? (
        <p className={styles.warning} role="status">
          {t("contribute.quietWarning")}
        </p>
      ) : null}
    </div>
  );
}
