/**
 * Free-text feedback (`POST /feedback`, İP-9).
 *
 * Three fields, one of which fills itself: the category (the API's own three values), the text,
 * and the page the reporter was on, pre-filled from the current route so a bug report can be
 * located without asking the reporter to remember where they were. A sent report clears the box
 * and says so; a failed one keeps every character.
 */
import { useState, type FormEvent } from "react";

import type { AiclsProblem } from "@/api/problem";
import type { FeedbackIn } from "@/api/types";
import { Button, Card, Field, Input, Select, Textarea, ErrorState } from "@/components/ui";
import { t, type MessageKey } from "@/i18n";
import { isAuthProblem } from "@/features/auth/fieldErrors";
import { SignInHint } from "@/features/auth/SignInHint";
import { submitFeedback } from "./api";
import styles from "./FeedbackForm.module.css";

/** Longest text the API accepts (`FeedbackIn.text`). */
const MAX_TEXT = 4000;

/** Longest page label the API accepts (`FeedbackIn.page`). */
const MAX_PAGE = 120;

/** The three categories of `FeedbackIn.category`. */
const CATEGORIES: { value: FeedbackIn["category"]; label: MessageKey }[] = [
  { value: "critical", label: "feedback.category.critical" },
  { value: "usability", label: "feedback.category.usability" },
  { value: "suggestion", label: "feedback.category.suggestion" },
];

export interface FeedbackFormProps {
  /** Page label pre-filled into the form (usually the current path). */
  page?: string;
}

/** The free-text feedback form. */
export function FeedbackForm({ page = "" }: FeedbackFormProps): JSX.Element {
  const [category, setCategory] = useState<FeedbackIn["category"]>("usability");
  const [text, setText] = useState("");
  const [pageLabel, setPageLabel] = useState(page);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [missing, setMissing] = useState(false);
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  const send = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault();
    const trimmed = text.trim();
    if (trimmed === "") {
      setMissing(true);
      return;
    }
    setBusy(true);
    setProblem(null);
    const { data, problem: failure } = await submitFeedback({
      category,
      text: trimmed,
      page: pageLabel.slice(0, MAX_PAGE),
    });
    setBusy(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    setText("");
    setMissing(false);
    setSent(true);
  };

  return (
    <Card title={t("feedback.formTitle")}>
      <form
        className={styles.form}
        onSubmit={(event) => {
          void send(event);
        }}
      >
        <p className={styles.intro}>{t("feedback.intro")}</p>

        <div className={styles.row}>
          <Field label={t("feedback.category")}>
            {(control) => (
              <Select
                {...control}
                value={category}
                onChange={(event) => {
                  setCategory(event.target.value as FeedbackIn["category"]);
                }}
              >
                {CATEGORIES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {t(option.label)}
                  </option>
                ))}
              </Select>
            )}
          </Field>

          <Field label={t("feedback.pageLabel")}>
            {(control) => (
              <Input
                {...control}
                type="text"
                maxLength={MAX_PAGE}
                value={pageLabel}
                onChange={(event) => {
                  setPageLabel(event.target.value);
                }}
              />
            )}
          </Field>
        </div>

        <Field
          label={t("feedback.text")}
          hint={t("feedback.textHint")}
          required
          {...(missing ? { error: t("feedback.textRequired") } : {})}
        >
          {(control) => (
            <Textarea
              {...control}
              // The Turkish message under the field replaces the browser's own untranslatable
              // validation bubble, so the control is required for assistive technology but not
              // for the HTML constraint API.
              required={false}
              aria-required="true"
              rows={6}
              maxLength={MAX_TEXT}
              value={text}
              onChange={(event) => {
                setMissing(false);
                setSent(false);
                setText(event.target.value);
              }}
            />
          )}
        </Field>

        <p className={styles.count}>{t("feedback.charCount", { count: text.length })}</p>

        {problem === null ? null : (
          <>
            <ErrorState problem={problem} />
            {isAuthProblem(problem) ? <SignInHint message={t("feedback.authRequired")} /> : null}
          </>
        )}

        {sent ? (
          <p className={styles.sent} role="status">
            {t("feedback.sent")}
          </p>
        ) : null}

        <div className={styles.actions}>
          <Button type="submit" variant="primary" loading={busy}>
            {t("feedback.send")}
          </Button>
        </div>
      </form>
    </Card>
  );
}
