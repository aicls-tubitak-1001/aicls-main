/**
 * The evaluation history (`GET /admin/eval-runs`).
 *
 * Every number here is a candidate for a TÜBİTAK progress report, so the panel treats provenance
 * as the primary information rather than a footnote: a run is either marked **alıntılanabilir**
 * (`is_reportable`) or it carries a warning that says, in plain Turkish, not to quote it. The
 * dataset fingerprint and the model version are shown next to the metrics for the same reason —
 * a CER without the checkpoint and the data it was measured on is not a result, it is a rumour.
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import { Button, Card, Chip, EmptyState, ErrorState, Spinner } from "@/components/ui";
import { t } from "@/i18n";
import { fetchEvalRuns, type EvalRun } from "./api";
import { metricEntries } from "./format";
import styles from "./Admin.module.css";

/** The evaluation history. */
export function EvalRunsPanel(): JSX.Element {
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (): Promise<void> => {
    setLoading(true);
    const { data, problem: failure } = await fetchEvalRuns();
    setLoading(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    setProblem(null);
    setRuns(data);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <Card
      title={t("admin.evalsTitle")}
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
      <p className={styles.lead}>{t("admin.evalsIntro")}</p>

      {problem === null ? null : (
        <ErrorState
          problem={problem}
          onRetry={() => {
            void load();
          }}
        />
      )}
      {loading && runs.length === 0 ? <Spinner label={t("common.loading")} /> : null}
      {!loading && runs.length === 0 ? <EmptyState title={t("admin.evalsEmpty")} /> : null}

      <ul className={styles.rows}>
        {runs.map((run) => (
          <li key={run.id} className={styles.row} data-testid="eval-run">
            <div className={styles.rowHead}>
              <span className={styles.rowId}>{run.name === "" ? run.id : run.name}</span>
              <Chip tone="neutral">{run.kind}</Chip>
              <Chip tone={run.is_reportable ? "ok" : "warn"}>
                {run.is_reportable ? t("admin.evalsCitable") : t("admin.evalsNotCitable")}
              </Chip>
            </div>

            {run.is_reportable ? null : (
              <p className={styles.warningBox}>{t("admin.evalsProvenanceWarning")}</p>
            )}

            <dl className={styles.meta}>
              <div className={styles.metaRow}>
                <dt>{t("admin.evalsN")}</dt>
                <dd>{run.n}</dd>
              </div>
              <div className={styles.metaRow}>
                <dt>{t("admin.evalsModel")}</dt>
                <dd className={styles.mono}>{run.model_version ?? t("common.none")}</dd>
              </div>
              <div className={styles.metaRow}>
                <dt>{t("admin.evalsFingerprint")}</dt>
                <dd className={styles.mono}>
                  {run.dataset_fingerprint === ""
                    ? t("admin.evalsNoFingerprint")
                    : run.dataset_fingerprint}
                </dd>
              </div>
              <div className={styles.metaRow}>
                <dt>{t("admin.created")}</dt>
                <dd>{run.created_at === "" ? t("common.none") : run.created_at}</dd>
              </div>
            </dl>

            <h3 className={styles.subheading}>{t("admin.evalsMetrics")}</h3>
            <dl className={styles.meta}>
              {metricEntries(run.metrics ?? {}).map((entry) => (
                <div className={styles.metaRow} key={entry.name}>
                  <dt className={styles.mono}>{entry.name}</dt>
                  <dd className={styles.metricValue}>{entry.value}</dd>
                </div>
              ))}
            </dl>
          </li>
        ))}
      </ul>
    </Card>
  );
}
