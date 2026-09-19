/**
 * The System Usability Scale questionnaire (İP-9; the contract's "SUS > 75" figure).
 *
 * This form produces a **contractual metric**, which makes two things non-negotiable:
 *
 * 1. **It must be finishable in under two minutes.** Ten statements, one screen, a five-point row
 *    per statement with the whole cell as the hit target, an answered-count that tells the
 *    participant how far they are, and no scrolling back and forth: the submit button and the
 *    counter travel with the form.
 * 2. **The score is not computed here.** `POST /feedback/sus` runs the frozen definition
 *    (`aicls.db.repositories.registry.sus_score`, ADR-0015), so the number a participant sees and
 *    the number in the progress report are produced by the same code. The browser only collects
 *    ten integers in 1..5.
 *
 * Each statement is a `<fieldset>` with its text as the `<legend>`, so a screen reader announces
 * the statement before the five options, and the options are plain radios — arrow keys work
 * because they are the platform's own.
 */
import { useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import { Button, Card, Chip, ErrorState } from "@/components/ui";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import { isAuthProblem } from "@/features/auth/fieldErrors";
import { SignInHint } from "@/features/auth/SignInHint";
import { submitSus } from "./api";
import styles from "./SusForm.module.css";

/** The ten SUS statements, in the canonical order (odd = positive, even = negative). */
const QUESTION_KEYS: MessageKey[] = [
  "sus.q1",
  "sus.q2",
  "sus.q3",
  "sus.q4",
  "sus.q5",
  "sus.q6",
  "sus.q7",
  "sus.q8",
  "sus.q9",
  "sus.q10",
];

/** The five points of the Likert scale. */
const SCALE = [1, 2, 3, 4, 5] as const;

/** Contract target the pilot has to beat (PROJECT_BRIEF §2). */
const TARGET_SCORE = 75;

export interface SusFormProps {
  /** Cohort label recorded with the answers (e.g. `"pilot-2027-guz"`). */
  cohort: string;
  /** ISO week the answers were collected in, when the study assigns one. */
  collectedWeek?: number;
  /** Called with the score the server computed. */
  onScored?: (score: number) => void;
}

/** The questionnaire. */
export function SusForm({ cohort, collectedWeek, onScored }: SusFormProps): JSX.Element {
  const [answers, setAnswers] = useState<(number | null)[]>(() => QUESTION_KEYS.map(() => null));
  const [score, setScore] = useState<number | null>(null);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [incomplete, setIncomplete] = useState(false);
  const [busy, setBusy] = useState(false);

  const answered = answers.filter((answer): answer is number => answer !== null);
  const complete = answered.length === QUESTION_KEYS.length;

  const submit = async (): Promise<void> => {
    if (!complete) {
      setIncomplete(true);
      return;
    }
    setBusy(true);
    setProblem(null);
    const { data, problem: failure } = await submitSus(answered, cohort, collectedWeek);
    setBusy(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    setIncomplete(false);
    setScore(data.score);
    onScored?.(data.score);
  };

  const reset = (): void => {
    setAnswers(QUESTION_KEYS.map(() => null));
    setScore(null);
    setProblem(null);
    setIncomplete(false);
  };

  return (
    <Card
      title={t("sus.title")}
      actions={
        <Chip tone={complete ? "ok" : "neutral"}>
          {t("sus.progress", { done: answered.length })}
        </Chip>
      }
    >
      <div className={styles.form}>
        <p className={styles.intro}>{t("sus.intro")}</p>
        <p className={styles.hint}>{t("sus.timeHint")}</p>

        {QUESTION_KEYS.map((key, index) => (
          <fieldset key={key} className={styles.question}>
            <legend className={styles.text}>
              <span className={styles.number} aria-hidden="true">
                {index + 1}.
              </span>{" "}
              {t(key)}
            </legend>
            <div className={styles.scale}>
              <span className={styles.anchor}>{t("sus.scaleMin")}</span>
              <div className={styles.options}>
                {SCALE.map((value) => (
                  <label
                    key={value}
                    className={cx(styles.option, answers[index] === value && styles.optionChecked)}
                  >
                    <input
                      type="radio"
                      name={`sus-${String(index + 1)}`}
                      value={value}
                      checked={answers[index] === value}
                      aria-label={t("sus.answerLabel", { question: t(key), value })}
                      onChange={() => {
                        setIncomplete(false);
                        setAnswers((previous) =>
                          previous.map((answer, position) => (position === index ? value : answer)),
                        );
                      }}
                    />
                    <span aria-hidden="true">{value}</span>
                  </label>
                ))}
              </div>
              <span className={styles.anchor}>{t("sus.scaleMax")}</span>
            </div>
          </fieldset>
        ))}

        {incomplete ? (
          <p className={styles.error} role="alert">
            {t("sus.incomplete")}
          </p>
        ) : null}

        {problem === null ? null : (
          <>
            <ErrorState problem={problem} />
            {isAuthProblem(problem) ? <SignInHint message={t("sus.authRequired")} /> : null}
          </>
        )}

        {score === null ? null : (
          <div className={styles.result} role="status">
            <p className={styles.score}>{t("sus.score", { score: Math.round(score * 10) / 10 })}</p>
            <p className={styles.scoreMeta}>
              <Chip tone={score > TARGET_SCORE ? "ok" : "warn"}>{t("sus.scoreTarget")}</Chip>{" "}
              {t("sus.scoreHint")}
            </p>
            <p className={styles.scoreMeta}>{t("sus.submitted")}</p>
          </div>
        )}

        <div className={styles.actions}>
          <Button
            variant="primary"
            size="lg"
            disabled={busy || !complete}
            loading={busy}
            onClick={() => {
              void submit();
            }}
          >
            {t("sus.submit")}
          </Button>
          {score === null ? null : (
            <Button variant="ghost" onClick={reset}>
              {t("sus.again")}
            </Button>
          )}
          <span className={styles.cohort}>{t("sus.cohort", { cohort })}</span>
        </div>
      </div>
    </Card>
  );
}
