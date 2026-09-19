/**
 * Spinner primitive.
 *
 * A spinner never replaces content here: the last answer stays on screen while the next one is
 * computed, so this is always shown *beside* something, never instead of it. It carries
 * `role="status"` with a Turkish label so a screen reader announces the wait once, and it stops
 * animating under `prefers-reduced-motion` (the ring stays, the rotation goes).
 */
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import styles from "./Spinner.module.css";

export interface SpinnerProps {
  /** `sm` sits inside a button; `md` next to a panel title; `lg` on an empty screen. */
  size?: "sm" | "md" | "lg";
  /** Drop the status role and the label (the surrounding control already announces the wait). */
  inline?: boolean;
  /** Accessible label; defaults to the Turkish "Yükleniyor…". */
  label?: string;
  className?: string;
}

/** An indeterminate progress indicator. */
export function Spinner({
  size = "md",
  inline = false,
  label,
  className,
}: SpinnerProps): JSX.Element {
  const classes = cx(styles.spinner, styles[size], className);
  if (inline) {
    return <span className={classes} aria-hidden="true" />;
  }
  return (
    <span className={styles.wrapper} role="status" aria-live="polite">
      <span className={classes} aria-hidden="true" />
      <span className="visually-hidden">{label ?? t("common.loading")}</span>
    </span>
  );
}
