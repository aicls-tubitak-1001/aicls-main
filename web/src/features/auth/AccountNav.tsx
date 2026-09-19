/**
 * The identity corner of the shell: who is signed in, and the two links that depend on it.
 *
 * Kept as its own component so the shell needs one element rather than a block of session logic,
 * and so the administration link appears **only** for the roles that can actually use it — an
 * instructor or an administrator (`aicls.api.deps.StaffUser`). A visitor who is not signed in gets
 * a plain "Giriş yap" link; nothing here ever redirects on its own.
 */
import { NavLink } from "react-router-dom";

import { ROLE_LABEL_KEYS, useAuth } from "@/app/auth";
import { t } from "@/i18n";
import { ROUTES } from "@/routes";
import { AUTH_ROUTES } from "./paths";
import styles from "./AccountNav.module.css";

/** Account and administration links for the header. */
export function AccountNav(): JSX.Element {
  const { isAuthenticated, user, role, can } = useAuth();

  return (
    <div className={styles.nav}>
      {can("instructor", "admin") ? (
        <NavLink
          to={ROUTES.admin}
          className={({ isActive }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)}
        >
          {t("nav.admin")}
        </NavLink>
      ) : null}

      {isAuthenticated ? (
        <NavLink
          to={AUTH_ROUTES.account}
          className={({ isActive }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)}
        >
          <span className={styles.name}>{user?.display_name ?? t("auth.accountTitle")}</span>
          {role === null ? null : <span className={styles.role}>{t(ROLE_LABEL_KEYS[role])}</span>}
        </NavLink>
      ) : (
        <NavLink
          to={AUTH_ROUTES.login}
          className={({ isActive }) => (isActive ? `${styles.link} ${styles.active}` : styles.link)}
        >
          {t("auth.login")}
        </NavLink>
      )}
    </div>
  );
}
