/**
 * The table of contents, tracking the reader's position.
 *
 * It is an ordered list of in-page links inside a labelled `<nav>` — plain anchors, so it works
 * without JavaScript, prints as a real outline and can be tabbed through. The section under the
 * top of the viewport carries `aria-current="location"`, which is the attribute a screen reader
 * announces as "current"; colour alone never carries that state.
 *
 * On a narrow screen the list moves above the article and stops sticking, because a sidebar that
 * follows the scroll on a phone eats the reading area.
 */
import { t } from "@/i18n";
import type { GuideSection } from "./sections";
import { useActiveSection } from "./useActiveSection";
import styles from "./GuideToc.module.css";

export interface GuideTocProps {
  readonly sections: readonly GuideSection[];
}

/** In-page navigation for one guide page. */
export function GuideToc({ sections }: GuideTocProps): JSX.Element {
  const ids = sections.map((section) => section.id);
  const active = useActiveSection(ids);

  return (
    <nav className={styles.toc} aria-label={t("guide.toc.label")} data-testid="guide-toc">
      <h2 className={styles.title}>{t("guide.toc.title")}</h2>
      <ol className={styles.list}>
        {sections.map((section) => {
          const isActive = section.id === active;
          return (
            <li key={section.id}>
              <a
                className={isActive ? `${styles.link} ${styles.linkActive}` : styles.link}
                href={`#${section.id}`}
                {...(isActive ? { "aria-current": "location" as const } : {})}
              >
                {t(section.titleKey)}
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
