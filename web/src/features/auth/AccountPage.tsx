/**
 * The account screen: who is signed in, with what role, and the way out.
 *
 * It is also where a session ends. `POST /auth/logout` revokes the whole refresh family, so the
 * button says what it does and the screen afterwards states plainly that the session is over
 * rather than silently turning into the sign-in form.
 */
import { useState } from "react";
import { Link } from "react-router-dom";

import { ROLE_LABEL_KEYS, useAuth } from "@/app/auth";
import { Button, Card, Chip, EmptyState, Spinner } from "@/components/ui";
import { t } from "@/i18n";
import { AUTH_ROUTES } from "./paths";
import styles from "./AuthPage.module.css";

/** The `/hesap` route. */
export function AccountPage(): JSX.Element {
  const { status, isAuthenticated, user, role, logout } = useAuth();
  const [signedOut, setSignedOut] = useState(false);
  const [busy, setBusy] = useState(false);

  if (status === "unknown") {
    return (
      <section className={styles.page}>
        <h1 className={styles.title}>{t("auth.accountTitle")}</h1>
        <Spinner size="lg" label={t("auth.restoring")} />
      </section>
    );
  }

  if (!isAuthenticated) {
    return (
      <section className={styles.page}>
        <h1 className={styles.title}>{t("auth.accountTitle")}</h1>
        <Card>
          <EmptyState
            title={signedOut ? t("auth.logoutDone") : t("auth.requiredTitle")}
            description={t("auth.requiredDescription")}
            action={
              <Link className={styles.link} to={AUTH_ROUTES.login}>
                {t("auth.login")}
              </Link>
            }
          />
        </Card>
      </section>
    );
  }

  return (
    <section className={styles.page}>
      <h1 className={styles.title}>{t("auth.accountTitle")}</h1>

      <Card
        title={user?.display_name ?? t("auth.anonymous")}
        actions={role === null ? null : <Chip tone="accent">{t(ROLE_LABEL_KEYS[role])}</Chip>}
      >
        <dl className={styles.details}>
          <div className={styles.detail}>
            <dt>{t("auth.accountName")}</dt>
            <dd>{user?.display_name ?? t("common.unknown")}</dd>
          </div>
          <div className={styles.detail}>
            <dt>{t("auth.accountEmail")}</dt>
            <dd>{user?.email ?? t("common.unknown")}</dd>
          </div>
          <div className={styles.detail}>
            <dt>{t("auth.accountRole")}</dt>
            <dd>{role === null ? t("common.unknown") : t(ROLE_LABEL_KEYS[role])}</dd>
          </div>
          <div className={styles.detail}>
            <dt>{t("auth.accountCreated")}</dt>
            <dd>{user?.created_at ?? t("common.unknown")}</dd>
          </div>
          <div className={styles.detail}>
            <dt>{t("auth.accountId")}</dt>
            <dd className={styles.mono}>{user?.id ?? t("common.unknown")}</dd>
          </div>
        </dl>

        <div className={styles.actions}>
          <Button
            variant="secondary"
            loading={busy}
            onClick={() => {
              setBusy(true);
              void logout().finally(() => {
                setBusy(false);
                setSignedOut(true);
              });
            }}
          >
            {t("auth.logout")}
          </Button>
        </div>
      </Card>
    </section>
  );
}
