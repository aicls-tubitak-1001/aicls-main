/**
 * Card primitive — the surface every panel in the workspace sits on.
 *
 * With a `title` it renders a labelled `<section>`, so a screen reader announces the panel by
 * name and the document keeps exactly one `<h1>` per view (a card's heading is an `<h2>`).
 * Without a title it is a plain container.
 */
import { useId, type ReactNode } from "react";

import { cx } from "@/lib/cx";
import styles from "./Card.module.css";

export interface CardProps {
  /** Heading text; also becomes the accessible name of the section. */
  title?: string;
  /** Controls rendered on the right of the header. */
  actions?: ReactNode;
  /** Quieter surface for secondary panels. */
  muted?: boolean;
  /** Emphasised surface for the result area (bigger padding, stronger contrast). */
  emphasis?: boolean;
  /** Row rendered under the body, separated by a hairline. */
  footer?: ReactNode;
  className?: string;
  children: ReactNode;
}

/** A titled surface. */
export function Card({
  title,
  actions,
  muted = false,
  emphasis = false,
  footer,
  className,
  children,
}: CardProps): JSX.Element {
  const headingId = useId();
  const classes = cx(styles.card, muted && styles.muted, emphasis && styles.emphasis, className);
  if (title === undefined) {
    return (
      <div className={classes}>
        {children}
        {footer === undefined ? null : <div className={styles.footer}>{footer}</div>}
      </div>
    );
  }
  return (
    <section className={classes} aria-labelledby={headingId}>
      <div className={styles.header}>
        <h2 id={headingId} className={styles.title}>
          {title}
        </h2>
        {actions === undefined ? null : <div className={styles.actions}>{actions}</div>}
      </div>
      <div className={styles.body}>{children}</div>
      {footer === undefined ? null : <div className={styles.footer}>{footer}</div>}
    </section>
  );
}
