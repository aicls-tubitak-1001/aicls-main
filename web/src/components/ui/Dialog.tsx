/**
 * Modal dialog primitive.
 *
 * A dialog is only ever opened by a deliberate action of the person driving the board — never by
 * an arriving error, a socket event or a finished request; those use `Toast` and the inline
 * error/empty states, because nothing may interrupt a lecture. When it *is* opened it behaves
 * properly: focus moves in, Tab is trapped inside, Escape closes, and focus returns to whatever
 * opened it.
 *
 * Implemented with plain elements rather than `<dialog>` so the behaviour is identical in the
 * three browsers the proposal names (Chrome, Firefox, Safari) and so the focus trap can be tested
 * in jsdom, where `showModal()` does not exist.
 */
import { useCallback, useEffect, useId, useRef, type ReactNode } from "react";

import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import styles from "./Dialog.module.css";

/** Selector of everything that can hold focus inside the panel. */
const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export interface DialogProps {
  open: boolean;
  title: string;
  /** Called on Escape, backdrop click and the close button. */
  onClose: () => void;
  /** Supporting sentence rendered under the title and wired to `aria-describedby`. */
  description?: string;
  /** Action row rendered at the bottom. */
  footer?: ReactNode;
  /** Wider panel for a book excerpt or a table. */
  wide?: boolean;
  /** Close when the backdrop is clicked (default `true`). */
  closeOnBackdrop?: boolean;
  className?: string;
  children: ReactNode;
}

/**
 * The focusable elements of the panel, in document order.
 *
 * Visibility is judged by attributes (`hidden`, `aria-hidden`) rather than by layout, because
 * jsdom performs no layout and every element would otherwise look invisible in the unit tests.
 */
function focusableIn(panel: HTMLElement): HTMLElement[] {
  return [...panel.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
    (element) => !element.hasAttribute("hidden") && element.getAttribute("aria-hidden") !== "true",
  );
}

/** An accessible modal dialog. */
export function Dialog({
  open,
  title,
  onClose,
  description,
  footer,
  wide = false,
  closeOnBackdrop = true,
  className,
  children,
}: DialogProps): JSX.Element | null {
  const titleId = useId();
  const descriptionId = useId();
  const panelRef = useRef<HTMLDivElement | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  const trapTab = useCallback((event: KeyboardEvent): void => {
    const panel = panelRef.current;
    if (panel === null) {
      return;
    }
    const items = focusableIn(panel);
    const first = items[0] ?? panel;
    const last = items[items.length - 1] ?? panel;
    const active = document.activeElement;
    if (event.shiftKey) {
      if (active === first || active === panel || !panel.contains(active)) {
        event.preventDefault();
        last.focus();
      }
      return;
    }
    if (active === last || !panel.contains(active)) {
      event.preventDefault();
      first.focus();
    }
  }, []);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    const previous = document.activeElement;
    returnFocusRef.current = previous instanceof HTMLElement ? previous : null;
    // Focus the panel itself rather than its first control: a screen reader then reads the title
    // and the description before the first button, and Tab still lands on that button next.
    panelRef.current?.focus();

    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key === "Tab") {
        trapTab(event);
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      returnFocusRef.current?.focus();
    };
  }, [open, onClose, trapTab]);

  if (!open) {
    return null;
  }
  return (
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (closeOnBackdrop && event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        ref={panelRef}
        className={cx(styles.dialog, wide && styles.wide, className)}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        {...(description === undefined ? {} : { "aria-describedby": descriptionId })}
        tabIndex={-1}
      >
        <div className={styles.header}>
          <h2 id={titleId} className={styles.title}>
            {title}
          </h2>
          <button
            type="button"
            className={styles.close}
            onClick={onClose}
            aria-label={t("common.close")}
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>
        {description === undefined ? null : (
          <p id={descriptionId} className={styles.description}>
            {description}
          </p>
        )}
        {children}
        {footer === undefined ? null : <div className={styles.footer}>{footer}</div>}
      </div>
    </div>
  );
}
