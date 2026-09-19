/**
 * Form primitives: {@link Field}, {@link Input}, {@link Select}, {@link Textarea}.
 *
 * `Field` owns the wiring that is easy to get wrong and impossible to see: one `<label>` bound to
 * the control by id, a hint and an error message bound with `aria-describedby`, `aria-invalid`
 * when the field is wrong, and an error that is announced (`role="alert"`) rather than silently
 * appearing. The controls render children through a callback so the ids stay in one place.
 */
import {
  useId,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import styles from "./Field.module.css";

/** Ids and ARIA attributes a `Field` hands to its control. */
export interface FieldControlProps {
  id: string;
  "aria-describedby"?: string;
  "aria-invalid"?: true;
  required?: boolean;
}

export interface FieldProps {
  /** Visible label; always rendered, never replaced by a placeholder. */
  label: string;
  /** Quiet explanation under the label. */
  hint?: string;
  /** Turkish error message; its presence marks the control invalid. */
  error?: string;
  /** Mark the field required (adds the Turkish "(zorunlu)" cue for assistive tech). */
  required?: boolean;
  className?: string;
  children: (props: FieldControlProps) => ReactNode;
}

/** A labelled form row. */
export function Field({
  label,
  hint,
  error,
  required = false,
  className,
  children,
}: FieldProps): JSX.Element {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = cx(hint === undefined ? "" : hintId, error === undefined ? "" : errorId);
  const control: FieldControlProps = {
    id,
    ...(describedBy === "" ? {} : { "aria-describedby": describedBy }),
    ...(error === undefined ? {} : { "aria-invalid": true as const }),
    ...(required ? { required: true } : {}),
  };
  return (
    <div className={cx(styles.field, className)}>
      <label className={styles.label} htmlFor={id}>
        {label}
        {required ? (
          <span className={styles.required}>
            <span aria-hidden="true"> *</span>
            <span className="visually-hidden"> ({t("common.requiredShort")})</span>
          </span>
        ) : null}
      </label>
      {hint === undefined ? null : (
        <p id={hintId} className={styles.hint}>
          {hint}
        </p>
      )}
      {children(control)}
      {error === undefined ? null : (
        <p id={errorId} className={styles.error} role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export type InputProps = InputHTMLAttributes<HTMLInputElement>;

/** A text input sized for a pen or a finger. */
export function Input({ className, ...rest }: InputProps): JSX.Element {
  return <input className={cx(styles.control, className)} {...rest} />;
}

export type SelectProps = SelectHTMLAttributes<HTMLSelectElement>;

/** A native `<select>` — no custom listbox, because the platform ones are the accessible ones. */
export function Select({ className, children, ...rest }: SelectProps): JSX.Element {
  return (
    <select className={cx(styles.control, styles.select, className)} {...rest}>
      {children}
    </select>
  );
}

export type TextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement>;

/** A multi-line input (feedback, consent notes). */
export function Textarea({ className, ...rest }: TextareaProps): JSX.Element {
  return <textarea className={cx(styles.control, styles.textarea, className)} {...rest} />;
}
