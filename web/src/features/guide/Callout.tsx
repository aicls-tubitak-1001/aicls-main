/**
 * A boxed remark: a contract sentence quoted verbatim, a provenance warning, a caveat.
 *
 * Tone is carried by a border, a background *and* a label, never by colour alone — the status
 * page's warnings are the part of this guide a reader must not be able to skim past, and some of
 * them will be read on a projector with poor colour reproduction.
 */
import type { ReactNode } from "react";

import { cx } from "@/lib/cx";
import styles from "./Callout.module.css";

/** Semantic weight of the remark. */
export type CalloutTone = "info" | "warn" | "critical";

export interface CalloutProps {
  readonly tone?: CalloutTone;
  /** Short label rendered above the body (already-translated Turkish). */
  readonly label?: string;
  readonly children: ReactNode;
  readonly className?: string;
}

const TONE_CLASS: Record<CalloutTone, string> = {
  info: "info",
  warn: "warn",
  critical: "critical",
};

/** A boxed remark. */
export function Callout({ tone = "info", label, children, className }: CalloutProps): JSX.Element {
  return (
    <div className={cx(styles.callout, styles[TONE_CLASS[tone]], className)}>
      {label === undefined ? null : <p className={styles.label}>{label}</p>}
      <div className={styles.body}>{children}</div>
    </div>
  );
}
