/**
 * Tooltip primitive.
 *
 * Two rules make this safe for a touch board:
 *
 * 1. **Never the only carrier of information.** The tooltip text is wired to the trigger with
 *    `aria-describedby`, so it is always available to a screen reader and to a keyboard user even
 *    when nothing is hovered; the trigger itself must still have a visible label.
 * 2. **No hover-only affordance.** It opens on hover *and* on focus, and it stays open while the
 *    pointer is inside, so a pen tap (which focuses) shows it too.
 */
import { useId, useState, type ReactNode } from "react";

import { cx } from "@/lib/cx";
import styles from "./Tooltip.module.css";

export interface TooltipProps {
  /** The supplementary text. Never the primary label of the control. */
  content: string;
  /** Which side to render on; `top` by default. */
  placement?: "top" | "bottom";
  className?: string;
  /**
   * The trigger. It receives `aria-describedby`, so pass a real control (a button or a link) —
   * a bare `<span>` cannot be reached by keyboard.
   */
  children: (props: { "aria-describedby": string }) => ReactNode;
}

/** A hover/focus tooltip whose text is always exposed through `aria-describedby`. */
export function Tooltip({
  content,
  placement = "top",
  className,
  children,
}: TooltipProps): JSX.Element {
  const id = useId();
  const [visible, setVisible] = useState(false);
  return (
    <span
      className={cx(styles.wrapper, className)}
      onMouseEnter={() => {
        setVisible(true);
      }}
      onMouseLeave={() => {
        setVisible(false);
      }}
      onFocusCapture={() => {
        setVisible(true);
      }}
      onBlurCapture={() => {
        setVisible(false);
      }}
    >
      {children({ "aria-describedby": id })}
      <span
        id={id}
        role="tooltip"
        className={cx(styles.bubble, styles[placement], !visible && styles.hidden)}
      >
        {content}
      </span>
    </span>
  );
}
