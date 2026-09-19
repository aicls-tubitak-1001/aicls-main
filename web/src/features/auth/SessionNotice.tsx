/**
 * "Oturum sona erdi" — the one thing an expired session is allowed to do to a lecture.
 *
 * The client already refreshes silently on a 401 (`api/client.ts`); this banner appears only when
 * the refresh cookie itself is gone, i.e. when the session store flips to `anonymous` *carrying a
 * problem*. Even then it does nothing but state the fact and offer a link: no modal, no redirect,
 * no clearing of the board — the last result stays exactly where the lecturer left it.
 *
 * Mount it once, in the shell, above the routed area.
 */
import { Link } from "react-router-dom";

import { useSession } from "@/app/auth";
import { t } from "@/i18n";
import { AUTH_ROUTES } from "./paths";
import styles from "./SessionNotice.module.css";

/** The session-expiry banner; renders nothing while the session is healthy. */
export function SessionNotice(): JSX.Element | null {
  const expired = useSession(
    (session) => session.status === "anonymous" && session.problem !== null,
  );
  const detail = useSession((session) => session.problem?.detail_tr ?? null);

  if (!expired) {
    return null;
  }

  return (
    <div className={styles.notice} role="status" aria-live="polite">
      <p className={styles.title}>{t("auth.sessionExpiredTitle")}</p>
      {detail === null ? null : <p className={styles.body}>{detail}</p>}
      <p className={styles.body}>{t("auth.sessionExpiredBody")}</p>
      <Link className={styles.link} to={AUTH_ROUTES.login}>
        {t("auth.login")}
      </Link>
    </div>
  );
}
