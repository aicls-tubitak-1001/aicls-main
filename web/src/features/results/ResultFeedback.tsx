/**
 * "Bu sonuç yanlış" — the report button that turns a bad answer into İP-9 data.
 *
 * Error reports from a real classroom are the only way the evaluation work package learns what the
 * test set does not contain, so the affordance is always within reach of the answer and costs one
 * tap. It expands inline (never a modal), pre-fills what the interface already knows — the command,
 * the answer and where it happened — and posts to `POST /feedback` with the category the lecturer
 * picked (`FeedbackIn.category`: `critical` | `usability` | `suggestion`).
 *
 * Handwriting corrections are *not* routed here: a wrong recognition has its own endpoint
 * (`POST /ocr/feedback`, with the recognition id) and is offered next to the ink preview, where the
 * corrected LaTeX is the natural thing to type.
 */
import { useState } from "react";

import { api, call } from "@/api/client";
import type { ComputeOut, FeedbackIn } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Field, Select, Textarea } from "@/components/ui/Field";
import { t, type MessageKey } from "@/i18n";
import styles from "./ResultFeedback.module.css";

/** The three categories of `FeedbackIn` (api.md §3.9). */
const CATEGORIES: { value: FeedbackIn["category"]; label: MessageKey }[] = [
  { value: "critical", label: "feedback.category.critical" },
  { value: "usability", label: "feedback.category.usability" },
  { value: "suggestion", label: "feedback.category.suggestion" },
];

/** Longest `page` value the API accepts (`FeedbackIn.page`, maxLength 120). */
const MAX_PAGE = 120;

export interface ResultFeedbackProps {
  /** The outcome being reported on; its command and answer seed the description. */
  outcome: ComputeOut | null;
}

/** Compose the pre-filled description so the lecturer only has to add what went wrong. */
function seedText(outcome: ComputeOut | null): string {
  if (outcome === null) {
    return "";
  }
  // The labels are read (and edited) by the lecturer inside the textarea, so they are UI text and
  // come from the catalogue like everything else on screen.
  const parts: string[] = [];
  if (typeof outcome.transcript === "string" && outcome.transcript !== "") {
    parts.push(`${t("results.command")}: ${outcome.transcript}`);
  }
  if (typeof outcome.input_latex === "string" && outcome.input_latex !== "") {
    parts.push(`${t("results.input")}: ${outcome.input_latex}`);
  }
  if (outcome.result !== null && outcome.result !== undefined) {
    parts.push(`${t("results.result")}: ${outcome.result.plain}`);
  }
  return parts.length === 0 ? "" : `${parts.join("\n")}\n`;
}

/** The "this answer is wrong" affordance and its inline form. */
export function ResultFeedback({ outcome }: ResultFeedbackProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const [category, setCategory] = useState<FeedbackIn["category"]>("critical");
  const [text, setText] = useState("");
  const [sent, setSent] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  const start = (): void => {
    setText(seedText(outcome));
    setSent(false);
    setErrorText(null);
    setOpen(true);
  };

  const send = async (): Promise<void> => {
    const trimmed = text.trim();
    if (trimmed === "") {
      return;
    }
    setSending(true);
    const page = globalThis.location?.pathname ?? "";
    const { data, problem } = await call(
      api.POST("/feedback", {
        body: {
          category,
          text: trimmed,
          page: page.slice(0, MAX_PAGE),
        },
      }),
      { fallbackDetailTr: t("errors.network") },
    );
    setSending(false);
    // `POST /feedback` answers 201 with the stored record, so a missing body means the report was
    // refused; the Turkish detail of the problem is what the lecturer sees.
    if (data === null) {
      setErrorText(problem?.detail_tr ?? t("errors.unknown"));
      return;
    }
    setSent(true);
    setOpen(false);
  };

  if (sent) {
    return (
      <p className={styles.sent} role="status">
        {t("feedback.sent")}
      </p>
    );
  }

  if (!open) {
    return (
      <div className={styles.trigger}>
        <Button variant="ghost" onClick={start} disabled={outcome === null}>
          {t("results.feedback")}
        </Button>
      </div>
    );
  }

  return (
    <section className={styles.form} aria-label={t("results.feedbackTitle")}>
      <h3 className={styles.title}>{t("results.feedbackTitle")}</h3>
      <Field label={t("feedback.category")}>
        {(control) => (
          <Select
            {...control}
            value={category}
            onChange={(event) => {
              setCategory(event.target.value as FeedbackIn["category"]);
            }}
          >
            {CATEGORIES.map((item) => (
              <option key={item.value} value={item.value}>
                {t(item.label)}
              </option>
            ))}
          </Select>
        )}
      </Field>
      <Field label={t("feedback.text")} {...(errorText === null ? {} : { error: errorText })}>
        {(control) => (
          <Textarea
            {...control}
            rows={4}
            value={text}
            placeholder={t("results.feedbackPlaceholder")}
            onChange={(event) => {
              setText(event.target.value);
            }}
          />
        )}
      </Field>
      <div className={styles.actions}>
        <Button
          variant="primary"
          loading={sending}
          disabled={text.trim() === ""}
          onClick={() => {
            void send();
          }}
        >
          {t("feedback.send")}
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            setOpen(false);
          }}
        >
          {t("common.cancel")}
        </Button>
      </div>
    </section>
  );
}
