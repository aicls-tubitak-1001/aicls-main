/**
 * Button primitive.
 *
 * No UI kit is used: four variants cover the whole application and keep both the bundle and the
 * accessibility surface small. Every button is a real `<button>` with a visible focus ring and a
 * hit target of at least 44 px (56 px in presentation mode) — the design system has no control
 * small enough to miss with a pen on a smart board.
 *
 * `loading` keeps the button's label on screen and adds a spinner plus `aria-busy`, so a lecturer
 * never loses the word they were about to read out; it also disables activation, because a second
 * press would queue a second computation.
 */
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cx } from "@/lib/cx";
import { Spinner } from "./Spinner";
import styles from "./Button.module.css";

/** Visual weight of the action. */
export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

/**
 * Control size.
 *
 * `md` and `lg` are the design-system sizes. `small` / `medium` / `large` are accepted aliases
 * from the first skeleton; `small` deliberately resolves to `md`, because a sub-44 px control has
 * no place on a touch board.
 */
export type ButtonSize = "md" | "lg" | "small" | "medium" | "large";

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type"> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Stretch to the container width (mobile forms). */
  block?: boolean;
  /** Show a spinner, keep the label, block activation. */
  loading?: boolean;
  /** Icon rendered before the label; decorative, never the only carrier of meaning. */
  icon?: ReactNode;
  /** `"button"` by default so it never submits a form by accident. */
  type?: "button" | "submit" | "reset";
  children?: ReactNode;
}

function resolveSize(size: ButtonSize): "md" | "lg" {
  return size === "lg" || size === "large" ? "lg" : "md";
}

/** A styled `<button>`. */
export function Button({
  variant = "secondary",
  size = "md",
  block = false,
  loading = false,
  icon,
  className,
  type = "button",
  disabled = false,
  children,
  ...rest
}: ButtonProps): JSX.Element {
  const resolved = resolveSize(size);
  return (
    <button
      type={type}
      className={cx(
        styles.button,
        styles[variant],
        resolved === "lg" && styles.lg,
        block && styles.block,
        loading && styles.loading,
        className,
      )}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <Spinner size="sm" inline /> : icon}
      <span className={styles.label}>{children}</span>
    </button>
  );
}
