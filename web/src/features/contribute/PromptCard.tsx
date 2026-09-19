/**
 * The sentence to read aloud.
 *
 * This is the largest thing on the collection screen for the same reason the result is the largest
 * thing in the workspace: the volunteer is reading it out loud, often at arm's length from a
 * tablet, and a mis-read sentence is a wasted recording. The scenario tag and the position in the
 * batch sit beside it as quiet chips, never competing with the text.
 */
import { useEffect, useRef } from "react";

import type { PromptOut } from "@/api/types";
import { Card, Chip } from "@/components/ui";
import { t } from "@/i18n";
import styles from "./PromptCard.module.css";

export interface PromptCardProps {
  /** The sentence, as `GET /contribute/prompts/next` returned it. */
  prompt: PromptOut;
  /** 1-based position in the current batch. */
  index: number;
  /** Size of the current batch. */
  total: number;
}

/** One prompt, rendered for reading aloud. */
export function PromptCard({ prompt, index, total }: PromptCardProps): JSX.Element {
  const sentenceRef = useRef<HTMLParagraphElement | null>(null);

  // Moving to the next sentence must move the reader with it: a screen reader announces the new
  // text and a keyboard user's next Tab lands on the record button rather than back at the top of
  // the page. The sentence is not in the tab order itself (`tabIndex={-1}`).
  useEffect(() => {
    sentenceRef.current?.focus();
  }, [prompt.id]);

  return (
    <Card
      title={t("contribute.promptTitle")}
      actions={<Chip tone="neutral">{t("contribute.promptCounter", { index, total })}</Chip>}
    >
      <p
        className={styles.sentence}
        data-testid="prompt-sentence"
        lang="tr"
        ref={sentenceRef}
        tabIndex={-1}
      >
        {prompt.text_tr}
      </p>
      <div className={styles.tags}>
        <Chip tone="accent">{t("contribute.promptScenario", { tag: prompt.scenario_tag })}</Chip>
        <Chip tone="neutral">{t("contribute.promptDifficulty", { level: prompt.difficulty })}</Chip>
      </div>
    </Card>
  );
}
