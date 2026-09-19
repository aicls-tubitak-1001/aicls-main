/**
 * Chip primitive — compact status labels (connection state, confidence, QC verdict, unit type).
 *
 * A chip is never the only carrier of its meaning: the colour is a second signal, the text is the
 * first, and a status chip that matters (the socket badge) also renders a dot so the state is
 * readable without colour perception.
 */
import type { ReactNode } from "react";

import { cx } from "@/lib/cx";
import styles from "./Chip.module.css";

/**
 * Semantic colour of a chip.
 *
 * `ok` / `warn` / `critical` are the design-system names; `success` / `warning` / `danger` are
 * accepted aliases from the first skeleton and resolve to the same tokens.
 */
export type ChipTone =
  "neutral" | "accent" | "info" | "ok" | "warn" | "critical" | "success" | "warning" | "danger";

export interface ChipProps {
  tone?: ChipTone;
  /** Show a leading status dot (used by the WebSocket badge). */
  dot?: boolean;
  /** Larger text for a chip that has to be read from the back row. */
  large?: boolean;
  /** Optional element role/title passthrough for a chip that carries live status. */
  title?: string;
  className?: string;
  children: ReactNode;
}

const TONE_CLASS: Record<Exclude<ChipTone, "neutral">, string> = {
  accent: "accent",
  info: "info",
  ok: "ok",
  warn: "warn",
  critical: "critical",
  success: "ok",
  warning: "warn",
  danger: "critical",
};

/** A small pill-shaped label. */
export function Chip({
  tone = "neutral",
  dot = false,
  large = false,
  title,
  className,
  children,
}: ChipProps): JSX.Element {
  const toneClass = tone === "neutral" ? undefined : styles[TONE_CLASS[tone]];
  return (
    <span
      className={cx(styles.chip, toneClass, large && styles.large, className)}
      {...(title === undefined ? {} : { title })}
    >
      {dot ? <span className={styles.dot} aria-hidden="true" /> : null}
      {children}
    </span>
  );
}
