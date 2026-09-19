/**
 * A clarification is a **200**, not an error (api.md §2, "Hata *olmayan* durumlar").
 *
 * When the NLU understands the shape of a command but not one of its slots — "türevini al" without
 * a variable — the backend answers with a Turkish question and the candidates it would run. In a
 * lecture that must feel like the platform asking politely, not like something breaking: no red, no
 * alert role, no modal. The question is large, the candidates are big tappable buttons with the
 * expression each one would compute, and choosing one re-submits the *same* sentence with
 * `confirm_intent`, so SymPy still produces the answer — nothing is guessed on the client.
 */
import type { ClarificationCandidateOut, ClarificationOut } from "@/api/types";
import { Chip } from "@/components/ui/Chip";
import { Button } from "@/components/ui/Button";
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import { MathView } from "./MathView";
import { useRenderedLatency } from "./useRenderedLatency";
import styles from "./ClarificationCard.module.css";

export interface ClarificationCardProps {
  /** The question and its candidates, straight from `ComputeOut.clarification`. */
  clarification: ClarificationOut;
  /** Called with the candidate the lecturer picked; the caller re-submits with `confirm_intent`. */
  onChoose: (candidate: ClarificationCandidateOut) => void;
  /** Called when the lecturer would rather rewrite the command. */
  onDismiss?: () => void;
  /** What the ASR heard, when the question came from a spoken command. */
  transcript?: string | null;
  /** Provisional latency key of the request that produced the question. */
  latencyKey?: string | null;
  /** Server operation id, when the question was recorded as an operation. */
  operationId?: string | null;
}

/** Label of one candidate: the backend's Turkish label, falling back to the intent id. */
function candidateLabel(candidate: ClarificationCandidateOut): string {
  const label = candidate.label_tr ?? "";
  return label === "" ? candidate.intent : label;
}

/** The question, its candidates and a way out. */
export function ClarificationCard({
  clarification,
  onChoose,
  onDismiss,
  transcript = null,
  latencyKey = null,
  operationId = null,
}: ClarificationCardProps): JSX.Element {
  const candidates = clarification.candidates ?? [];
  // The question is what the system produced for this command: the measurement ends here, before
  // the lecturer's own decision time, so the contractual average stays a machine measurement.
  useRenderedLatency(latencyKey, operationId);

  return (
    <section className={styles.card} aria-labelledby="clarification-question" role="group">
      <p className={styles.eyebrow}>{t("clarification.title")}</p>
      <h2 className={styles.question} id="clarification-question" data-testid="clarification">
        {clarification.question_tr}
      </h2>

      {transcript === null || transcript === "" ? null : (
        <p className={styles.transcript}>“{transcript}”</p>
      )}

      {clarification.missing_slot === null || clarification.missing_slot === undefined ? null : (
        <p className={styles.slot}>
          {t("clarification.missingSlot", { slot: clarification.missing_slot })}
        </p>
      )}

      {candidates.length === 0 ? (
        <p className={styles.empty}>{t("clarification.empty")}</p>
      ) : (
        <>
          <p className={styles.hint}>{t("clarification.hint")}</p>
          <ul className={styles.options}>
            {candidates.map((candidate) => (
              <li key={`${candidate.intent}-${candidate.latex_preview ?? ""}`}>
                <button
                  type="button"
                  className={styles.option}
                  data-testid="clarification-candidate"
                  onClick={() => {
                    onChoose(candidate);
                  }}
                >
                  <span className={styles.optionLabel}>{candidateLabel(candidate)}</span>
                  {candidate.latex_preview === undefined ||
                  candidate.latex_preview === "" ? null : (
                    <span className={styles.optionMath}>
                      <MathView latex={candidate.latex_preview} />
                    </span>
                  )}
                  <Chip tone="neutral" className={cx(styles.optionScore)}>
                    {t("clarification.score", {
                      percent: Math.round((candidate.score ?? 0) * 100),
                    })}
                  </Chip>
                </button>
              </li>
            ))}
          </ul>
        </>
      )}

      {onDismiss === undefined ? null : (
        <div className={styles.actions}>
          <Button variant="ghost" onClick={onDismiss}>
            {t("clarification.dismiss")}
          </Button>
        </div>
      )}
    </section>
  );
}
