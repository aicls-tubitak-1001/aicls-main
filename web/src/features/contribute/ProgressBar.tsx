/**
 * "How much more?" — the session goal and the running total.
 *
 * A volunteer who cannot see the end of the task stops in the middle of it, so the primary figure
 * here is a **session** goal ({@link SESSION_GOAL} accepted sentences, a few minutes' work) and not
 * the campaign's 50 hours. The lifetime total from `GET /contribute/me/progress` is shown beneath
 * it as context, and the moment the session goal is reached the panel says, in Turkish, that
 * stopping now is a complete contribution.
 */
import type { ContributeProgressOut } from "@/api/types";
import { Chip } from "@/components/ui";
import { t } from "@/i18n";
import { SESSION_GOAL } from "./api";
import styles from "./ProgressBar.module.css";

export interface ProgressBarProps {
  /** Accepted takes in this sitting. */
  sessionAccepted: number;
  /** Lifetime progress from the API, or `null` before it has loaded. */
  progress: ContributeProgressOut | null;
  /** Accepted takes that count as a full session; defaults to {@link SESSION_GOAL}. */
  goal?: number;
}

/** Session goal plus lifetime total. */
export function ProgressBar({
  sessionAccepted,
  progress,
  goal = SESSION_GOAL,
}: ProgressBarProps): JSX.Element {
  const done = Math.min(sessionAccepted, goal);
  const percent = goal === 0 ? 0 : Math.round((done / goal) * 100);
  const complete = sessionAccepted >= goal;
  const minutes = progress === null ? 0 : Math.round((progress.seconds_accepted / 60) * 10) / 10;

  return (
    <div className={styles.block}>
      <div
        className={styles.track}
        role="progressbar"
        aria-label={t("contribute.progressTitle")}
        aria-valuemin={0}
        aria-valuemax={goal}
        aria-valuenow={done}
        aria-valuetext={t("contribute.sessionGoal", { done: sessionAccepted, goal })}
      >
        <span className={styles.fill} style={{ width: `${String(percent)}%` }} />
      </div>

      <p className={styles.line}>
        <strong>{t("contribute.sessionGoal", { done: sessionAccepted, goal })}</strong>{" "}
        {complete ? null : <span className={styles.hint}>{t("contribute.sessionGoalHint")}</span>}
      </p>

      {complete ? (
        <p className={styles.done} role="status">
          <Chip tone="ok" large>
            {t("contribute.sessionGoalDone")}
          </Chip>
        </p>
      ) : null}

      {progress === null ? null : (
        <dl className={styles.totals}>
          <div className={styles.total}>
            <dt>{t("contribute.totalProgressTitle")}</dt>
            <dd>{t("contribute.progressAccepted", { count: progress.n_accepted })}</dd>
          </div>
          <div className={styles.total}>
            <dt className="visually-hidden">{t("contribute.totalProgressTitle")}</dt>
            <dd>{t("contribute.progressMinutes", { minutes })}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}
