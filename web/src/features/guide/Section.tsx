/**
 * One addressable section of a guide page.
 *
 * Every section is a real `<section>` with an `id`, labelled by its own `<h2>`, and carries a
 * permalink. The permalink matters more here than in most documentation: a lecturer preparing a
 * class pastes `…/kilavuz/teknik#neden-cebir-motoru` into a lesson plan, and a reviewer quotes an
 * anchor rather than a page number. The link is a 44 px target like every other control, so it
 * can be hit with a pen on the board.
 *
 * Heading level is fixed at `<h2>`: each guide page owns exactly one `<h1>` (its title), and
 * nothing inside a section may skip a level.
 */
import type { ReactNode } from "react";

import { t, type MessageKey } from "@/i18n";
import styles from "./Section.module.css";

export interface SectionProps {
  /** DOM id and URL fragment; must match the entry in `sections.ts`. */
  readonly id: string;
  /** i18n key of the heading. */
  readonly titleKey: MessageKey;
  readonly children: ReactNode;
}

/** A titled, deep-linkable block of prose. */
export function Section({ id, titleKey, children }: SectionProps): JSX.Element {
  const title = t(titleKey);
  const headingId = `${id}-baslik`;
  return (
    <section id={id} className={styles.section} aria-labelledby={headingId}>
      <h2 id={headingId} className={styles.heading}>
        <span className={styles.headingText}>{title}</span>
        <a
          className={styles.anchor}
          href={`#${id}`}
          aria-label={t("guide.anchor", { section: title })}
        >
          <span aria-hidden="true">§</span>
        </a>
      </h2>
      <div className={styles.body}>{children}</div>
    </section>
  );
}
