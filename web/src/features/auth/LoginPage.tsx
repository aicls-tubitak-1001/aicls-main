/**
 * Sign-in screen.
 *
 * The backend answers a refused login with one Turkish sentence that is identical for an unknown
 * address, a wrong password and a locked account (`aicls.api.routers.auth.login`) — deliberately,
 * so accounts cannot be enumerated. This screen therefore never guesses *why* a sign-in failed: it
 * shows `detail_tr` above the form, and shows per-field messages only when the API attributed them
 * to a field. The single client-side rule is "this box is empty", which saves a round trip.
 *
 * Navigation happens only as the direct result of pressing the button, and only after the session
 * exists; a lecturer is never bounced somewhere by a background refresh.
 */
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";

import type { AiclsProblem } from "@/api/problem";
import { useAuth } from "@/app/auth";
import { Button, Card, ErrorState, Field, Input } from "@/components/ui";
import { t } from "@/i18n";
import { fieldErrorsOf, isFormLevel } from "./fieldErrors";
import { AFTER_LOGIN, AUTH_ROUTES } from "./paths";
import styles from "./AuthPage.module.css";

/** The `/giris` route. */
export function LoginPage(): JSX.Element {
  const { isAuthenticated, login } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [reveal, setReveal] = useState(false);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [missing, setMissing] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const empty: Record<string, string> = {};
    if (email.trim() === "") {
      empty.email = t("auth.emailRequired");
    }
    if (password === "") {
      empty.password = t("auth.passwordRequired");
    }
    setMissing(empty);
    if (Object.keys(empty).length > 0) {
      return;
    }
    setBusy(true);
    setProblem(null);
    const failure = await login(email.trim(), password);
    setBusy(false);
    if (failure !== null) {
      setProblem(failure);
      return;
    }
    navigate(AFTER_LOGIN, { replace: true });
  };

  if (isAuthenticated) {
    return (
      <section className={styles.page}>
        <h1 className={styles.title}>{t("auth.signedInTitle")}</h1>
        <Card>
          <p className={styles.lead}>{t("auth.signedInBody")}</p>
          <Link className={styles.link} to={AUTH_ROUTES.account}>
            {t("auth.accountTitle")}
          </Link>
        </Card>
      </section>
    );
  }

  const fields = fieldErrorsOf(problem);
  const emailError = missing.email ?? fields.email;
  const passwordError = missing.password ?? fields.password;

  return (
    <section className={styles.page}>
      <h1 className={styles.title}>{t("auth.loginTitle")}</h1>

      <Card>
        <form
          className={styles.form}
          onSubmit={(event) => {
            void submit(event);
          }}
        >
          <p className={styles.lead}>{t("auth.loginIntro")}</p>

          {problem !== null && isFormLevel(problem) ? <ErrorState problem={problem} /> : null}

          <Field
            label={t("auth.email")}
            required
            {...(emailError === undefined ? {} : { error: emailError })}
          >
            {(control) => (
              <Input
                {...control}
                required={false}
                aria-required="true"
                type="email"
                autoComplete="username"
                inputMode="email"
                maxLength={255}
                value={email}
                onChange={(event) => {
                  setEmail(event.target.value);
                }}
              />
            )}
          </Field>

          <Field
            label={t("auth.password")}
            required
            {...(passwordError === undefined ? {} : { error: passwordError })}
          >
            {(control) => (
              <Input
                {...control}
                required={false}
                aria-required="true"
                type={reveal ? "text" : "password"}
                autoComplete="current-password"
                maxLength={128}
                value={password}
                onChange={(event) => {
                  setPassword(event.target.value);
                }}
              />
            )}
          </Field>

          <div className={styles.actions}>
            <Button type="submit" variant="primary" size="lg" loading={busy}>
              {t("auth.login")}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setReveal(!reveal);
              }}
              aria-pressed={reveal}
            >
              {reveal ? t("auth.hidePassword") : t("auth.showPassword")}
            </Button>
          </div>

          <p className={styles.switch}>
            {t("auth.noAccount")}{" "}
            <Link className={styles.link} to={AUTH_ROUTES.register}>
              {t("auth.goRegister")}
            </Link>
          </p>
        </form>
      </Card>
    </section>
  );
}
