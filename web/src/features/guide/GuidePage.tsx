/**
 * The guide shell: one `<h1>`, the three-way sub-navigation, the tracking table of contents and
 * the article.
 *
 * Two audiences share the section but never the page: "Kullanım" is written for the lecturer and
 * the student, "Teknik" for the developer and the reviewer, "Durum" for whoever needs to know what
 * has actually been measured. Mixing them would make both worse, so they are three routes with
 * three `<h1>`s and three tables of contents.
 *
 * Deep links are first class. `/kilavuz/teknik#boru-hatti` opens the technical page scrolled to
 * that section; the scroll respects `prefers-reduced-motion`, and a fragment that names no section
 * is ignored rather than throwing. Nothing here steals focus: the page never moves the caret away
 * from where the reader put it.
 */
import { useEffect } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { Button } from "@/components/ui/Button";
import { t, type MessageKey } from "@/i18n";
import { GuideToc } from "./GuideToc";
import { GUIDE_ROUTES } from "./routes";
import { GUIDE_SECTIONS, type GuidePageId } from "./sections";
import { StatusGuide } from "./StatusGuide";
import { TechGuide } from "./TechGuide";
import { UsageGuide } from "./UsageGuide";
import styles from "./GuidePage.module.css";

interface TabDefinition {
  readonly page: GuidePageId;
  readonly to: string;
  readonly labelKey: MessageKey;
  readonly hintKey: MessageKey;
}

const TABS: readonly TabDefinition[] = [
  {
    page: "usage",
    to: GUIDE_ROUTES.usage,
    labelKey: "guide.nav.usage",
    hintKey: "guide.nav.usageHint",
  },
  {
    page: "tech",
    to: GUIDE_ROUTES.tech,
    labelKey: "guide.nav.tech",
    hintKey: "guide.nav.techHint",
  },
  {
    page: "status",
    to: GUIDE_ROUTES.status,
    labelKey: "guide.nav.status",
    hintKey: "guide.nav.statusHint",
  },
];

const PAGE_TITLE_KEY: Readonly<Record<GuidePageId, MessageKey>> = {
  usage: "guide.usage.title",
  tech: "guide.tech.title",
  status: "guide.status.title",
};

function prefersReducedMotion(): boolean {
  return globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
}

/** Open the browser's print dialog, if this environment has one. */
function printPage(): void {
  try {
    globalThis.print?.();
  } catch {
    // jsdom and locked-down kiosk browsers have no print dialog; the page still prints from the
    // browser menu, so there is nothing useful to tell the user here.
  }
}

export interface GuidePageProps {
  /** Which of the three guide pages to render. */
  readonly page: GuidePageId;
}

/** One page of the in-site documentation. */
export function GuidePage({ page }: GuidePageProps): JSX.Element {
  const location = useLocation();
  const sections = GUIDE_SECTIONS[page];
  const hash = location.hash;

  useEffect(() => {
    if (hash.length < 2) {
      return;
    }
    const element = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (element === null || typeof element.scrollIntoView !== "function") {
      return;
    }
    element.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "start",
    });
  }, [hash, page]);

  const body =
    page === "usage" ? <UsageGuide /> : page === "tech" ? <TechGuide /> : <StatusGuide />;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <p className={styles.kicker}>{t("guide.title")}</p>
        <h1 id="kilavuz-baslik" className={styles.title}>
          {t(PAGE_TITLE_KEY[page])}
        </h1>
        <p className={`${styles.subtitle} chrome-secondary`}>{t("guide.subtitle")}</p>

        <div className={styles.bar}>
          <nav className={styles.tabs} aria-label={t("guide.nav.label")}>
            {TABS.map((tab) => (
              <NavLink
                key={tab.page}
                to={tab.to}
                className={({ isActive }) =>
                  isActive ? `${styles.tab} ${styles.tabActive}` : styles.tab
                }
              >
                <span className={styles.tabLabel}>{t(tab.labelKey)}</span>
                <span className={`${styles.tabHint} chrome-secondary`}>{t(tab.hintKey)}</span>
              </NavLink>
            ))}
          </nav>

          <Button
            className={`${styles.print} chrome-secondary`}
            variant="ghost"
            onClick={printPage}
            title={t("guide.printHint")}
          >
            {t("guide.print")}
          </Button>
        </div>
      </header>

      <div className={styles.layout}>
        <GuideToc sections={sections} />
        <article className={styles.article} aria-labelledby="kilavuz-baslik">
          {body}
          <p className={styles.source}>{t("guide.sourceNote")}</p>
        </article>
      </div>
    </div>
  );
}
