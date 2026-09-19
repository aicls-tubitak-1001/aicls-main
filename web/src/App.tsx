/**
 * Application shell: skip link, header navigation, appearance switch and the routed main area.
 *
 * Everything visible here is Turkish and comes from `i18n/tr.json`; the shell itself holds no
 * business logic so that a route can be tested in isolation.
 */
import { NavLink } from "react-router-dom";

import { useDensity } from "@/app/density";
import { useTheme, type ThemePreference } from "@/app/theme";
import { Button } from "@/components/ui/Button";
import { AccountNav } from "@/features/auth/AccountNav";
import { SessionNotice } from "@/features/auth/SessionNotice";
import { t, type MessageKey } from "@/i18n";
import { AppRoutes, ROUTES } from "@/routes";
import styles from "./App.module.css";

const NAV_ITEMS: { to: string; label: MessageKey }[] = [
  { to: ROUTES.workspace, label: "nav.workspace" },
  { to: ROUTES.book, label: "nav.book" },
  { to: ROUTES.contribute, label: "nav.contribute" },
  { to: ROUTES.feedback, label: "nav.feedback" },
  { to: ROUTES.guide, label: "nav.guide" },
];

const THEME_OPTIONS: { value: ThemePreference; label: MessageKey }[] = [
  { value: "system", label: "theme.system" },
  { value: "light", label: "theme.light" },
  { value: "dark", label: "theme.dark" },
];

/** The shell around every route. */
export function App(): JSX.Element {
  const { preference, setPreference } = useTheme();
  const { isPresentation, toggleDensity } = useDensity();

  return (
    <div className={styles.shell}>
      <a className="skip-link" href="#main">
        {t("app.skipToContent")}
      </a>

      <header className={styles.header}>
        <div className={styles.headerInner}>
          <NavLink to={ROUTES.workspace} className={styles.brand ?? ""}>
            {t("app.title")}
            <span className={styles.brandSub}>{t("app.subtitle")}</span>
          </NavLink>

          <nav className={styles.nav} aria-label={t("nav.menu")}>
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  isActive ? `${styles.navLink} ${styles.navLinkActive}` : styles.navLink
                }
              >
                {t(item.label)}
              </NavLink>
            ))}
          </nav>

          <AccountNav />

          <div className={`${styles.appearance} ${styles.spacer}`}>
            {/*
              Presentation mode is written on <html> and therefore outlives the route that turned
              it on. Its switch lives here, in the shell, so a lecturer who walked from the board
              to the guide can always turn it off again — the workspace keeps its own copy of the
              button and the Alt + S shortcut, and `useDensity` keeps the two in step.
            */}
            <Button variant="ghost" onClick={toggleDensity} aria-pressed={isPresentation}>
              {isPresentation ? t("density.disable") : t("density.enable")}
            </Button>

            <div className={styles.themeSwitch}>
              <label htmlFor="theme-select">{t("theme.label")}</label>
              <select
                id="theme-select"
                className={styles.select}
                value={preference}
                onChange={(event) => {
                  setPreference(event.target.value as ThemePreference);
                }}
              >
                {THEME_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {t(option.label)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </header>

      <main id="main" className={styles.main}>
        {/* An expired session is stated, never acted upon: the board keeps what is on it. */}
        <SessionNotice />
        <AppRoutes />
      </main>

      <footer className={styles.footer}>{t("app.footer")}</footer>
    </div>
  );
}
