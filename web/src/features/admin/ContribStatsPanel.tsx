/**
 * The collection dashboard (`GET /admin/contributions/stats`).
 *
 * The four headline numbers are the ones İP-3 is judged on — accepted hours against the 50-hour
 * target, utterances, distinct speakers and the automatic pass rate — and the three breakdowns
 * below them are what tells a coordinator whether the corpus is balanced enough to be usable
 * (a speaker-disjoint test set needs speakers from more than one university).
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import { Button, Card, ErrorState, Spinner } from "@/components/ui";
import { t } from "@/i18n";
import { fetchContributionStats, type ContribStats } from "./api";
import styles from "./Admin.module.css";

/** Contract target of the collection campaign (PROJECT_BRIEF §2). */
const TARGET_HOURS = 50;

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/** One `label: value` breakdown block. */
function Breakdown({
  title,
  entries,
}: {
  title: string;
  entries: Readonly<Record<string, number>>;
}): JSX.Element {
  const rows = Object.entries(entries);
  return (
    <div className={styles.breakdown}>
      <h3 className={styles.subheading}>{title}</h3>
      {rows.length === 0 ? (
        <p className={styles.muted}>{t("common.none")}</p>
      ) : (
        <dl className={styles.meta}>
          {rows.map(([key, value]) => (
            <div className={styles.metaRow} key={key}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

/** The contribution dashboard. */
export function ContribStatsPanel(): JSX.Element {
  const [stats, setStats] = useState<ContribStats | null>(null);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (): Promise<void> => {
    setLoading(true);
    const { data, problem: failure } = await fetchContributionStats();
    setLoading(false);
    setProblem(failure);
    setStats(data);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <Card
      title={t("admin.statsTitle")}
      actions={
        <Button
          variant="ghost"
          onClick={() => {
            void load();
          }}
        >
          {t("admin.refresh")}
        </Button>
      }
    >
      {problem === null ? null : (
        <ErrorState
          problem={problem}
          onRetry={() => {
            void load();
          }}
        />
      )}
      {loading && stats === null ? <Spinner label={t("common.loading")} /> : null}

      {stats === null ? null : (
        <>
          <ul className={styles.metrics}>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.statsHours")}</span>
              <span className={styles.metricValue}>{round(stats.hours_accepted, 2)}</span>
              <span className={styles.metricHint}>{t("admin.statsTargetHours")}</span>
            </li>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.statsUtterances")}</span>
              <span className={styles.metricValue}>{stats.utterances}</span>
            </li>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.statsSpeakers")}</span>
              <span className={styles.metricValue}>{stats.speakers}</span>
            </li>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.statsPassRate")}</span>
              <span className={styles.metricValue}>
                {t("common.percent", { value: round(stats.qc_pass_rate * 100, 1) })}
              </span>
            </li>
          </ul>

          <div
            className={styles.progressTrack}
            role="progressbar"
            aria-label={t("admin.statsHours")}
            aria-valuemin={0}
            aria-valuemax={TARGET_HOURS}
            aria-valuenow={round(stats.hours_accepted, 2)}
          >
            <span
              className={styles.progressFill}
              style={{
                width: `${String(Math.min(100, (stats.hours_accepted / TARGET_HOURS) * 100))}%`,
              }}
            />
          </div>

          <div className={styles.breakdowns}>
            <Breakdown title={t("admin.statsByUniversity")} entries={stats.by_university ?? {}} />
            <Breakdown title={t("admin.statsByGender")} entries={stats.by_gender ?? {}} />
            <Breakdown title={t("admin.statsByAgeBand")} entries={stats.by_age_band ?? {}} />
          </div>
        </>
      )}
    </Card>
  );
}
