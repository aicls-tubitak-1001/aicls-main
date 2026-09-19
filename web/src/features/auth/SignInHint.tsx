/**
 * "You need to be signed in for this" — with the way to fix it.
 *
 * Used under a form whose submission was refused for want of a session. It is a plain anchor
 * rather than a router link so the component can be dropped into any form without dragging a
 * `Router` into that form's tests, and a full page load is the honest thing to do here anyway: the
 * session is gone.
 */
import { t } from "@/i18n";
import { AUTH_ROUTES } from "./paths";
import styles from "./SessionNotice.module.css";

export interface SignInHintProps {
  /** The Turkish sentence explaining what needs a session. */
  message: string;
}

/** A sentence plus a sign-in link. */
export function SignInHint({ message }: SignInHintProps): JSX.Element {
  return (
    <p className={styles.notice} role="status">
      <span className={styles.body}>{message}</span>
      <a className={styles.link} href={AUTH_ROUTES.login}>
        {t("auth.login")}
      </a>
    </p>
  );
}
