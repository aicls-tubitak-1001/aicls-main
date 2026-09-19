/**
 * Sign-up screen.
 *
 * `POST /auth/register` creates a `student` by default and an `instructor` when a valid invite
 * code is supplied, then `app/auth.register` immediately exchanges the credentials for a session —
 * so filling this form once is enough. The password rule (at least ten characters) is stated
 * before it is broken; everything else that can be wrong comes back from the API as a field
 * message, including "this address is already registered".
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

/** Shortest password the API accepts (`RegisterIn.password`). */
const MIN_PASSWORD = 10;

/** The `/kayit` route. */
export function RegisterPage(): JSX.Element {
  const { isAuthenticated, register } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [invite, setInvite] = useState("");
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
    if (displayName.trim() === "") {
      empty.display_name = t("auth.displayNameRequired");
    }
    if (password === "") {
      empty.password = t("auth.passwordRequired");
    } else if (password.length < MIN_PASSWORD) {
      empty.password = t("auth.passwordTooShort");
    }
    setMissing(empty);
    if (Object.keys(empty).length > 0) {
      return;
    }
    setBusy(true);
    setProblem(null);
    const failure = await register(
      email.trim(),
      password,
      displayName.trim(),
      invite.trim() === "" ? undefined : invite.trim(),
    );
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
  const nameError = missing.display_name ?? fields.display_name;
  const inviteError = fields.invite_code;

  return (
    <section className={styles.page}>
      <h1 className={styles.title}>{t("auth.registerTitle")}</h1>

      <Card>
        <form
          className={styles.form}
          onSubmit={(event) => {
            void submit(event);
          }}
        >
          <p className={styles.lead}>{t("auth.registerIntro")}</p>

          {problem !== null && isFormLevel(problem) ? <ErrorState problem={problem} /> : null}

          <Field
            label={t("auth.displayName")}
            required
            {...(nameError === undefined ? {} : { error: nameError })}
          >
            {(control) => (
              <Input
                {...control}
                required={false}
                aria-required="true"
                type="text"
                autoComplete="name"
                maxLength={120}
                value={displayName}
                onChange={(event) => {
                  setDisplayName(event.target.value);
                }}
              />
            )}
          </Field>

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
            hint={t("auth.passwordHint")}
            required
            {...(passwordError === undefined ? {} : { error: passwordError })}
          >
            {(control) => (
              <Input
                {...control}
                required={false}
                aria-required="true"
                type={reveal ? "text" : "password"}
                autoComplete="new-password"
                minLength={MIN_PASSWORD}
                maxLength={128}
                value={password}
                onChange={(event) => {
                  setPassword(event.target.value);
                }}
              />
            )}
          </Field>

          <Field
            label={t("auth.inviteCode")}
            hint={t("auth.inviteHint")}
            {...(inviteError === undefined ? {} : { error: inviteError })}
          >
            {(control) => (
              <Input
                {...control}
                type="text"
                maxLength={64}
                value={invite}
                onChange={(event) => {
                  setInvite(event.target.value);
                }}
              />
            )}
          </Field>

          <div className={styles.actions}>
            <Button type="submit" variant="primary" size="lg" loading={busy}>
              {t("auth.register")}
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
            {t("auth.haveAccount")}{" "}
            <Link className={styles.link} to={AUTH_ROUTES.login}>
              {t("auth.goLogin")}
            </Link>
          </p>
        </form>
      </Card>
    </section>
  );
}
