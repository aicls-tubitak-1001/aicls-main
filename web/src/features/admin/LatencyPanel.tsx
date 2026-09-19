/**
 * Measured end-to-end latency (`GET /admin/metrics/latency`).
 *
 * This is the contractual "ortalama < 3 s" figure, so the panel states the budget next to the mean
 * and says whether it is met — with the sample count beside it, because a mean over eleven samples
 * is not a result. The per-modality breakdown is what tells a coordinator whether it is the
 * handwriting path or the voice path that is slow.
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import {
  Button,
  Card,
  Chip,
  EmptyState,
  ErrorState,
  LATENCY_BUDGET_MS,
  Spinner,
} from "@/components/ui";
import { t, type MessageKey } from "@/i18n";
import { fetchLatencyStats, type LatencyStats } from "./api";
import styles from "./Admin.module.css";

/** Turkish label of each latency modality (the four `input_kind` values). */
const MODALITY_LABELS: Record<string, MessageKey> = {
  text: "history.inputKind.text",
  latex: "history.inputKind.latex",
  ink: "history.inputKind.ink",
  audio: "history.inputKind.audio",
};

function modalityLabel(name: string): string {
  const key = MODALITY_LABELS[name];
  return key === undefined ? name : t(key);
}

function ms(value: number): string {
  return `${String(Math.round(value))} ${t("common.milliseconds")}`;
}

/** The latency dashboard. */
export function LatencyPanel(): JSX.Element {
  const [stats, setStats] = useState<LatencyStats | null>(null);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (): Promise<void> => {
    setLoading(true);
    const { data, problem: failure } = await fetchLatencyStats();
    setLoading(false);
    setProblem(failure);
    setStats(data);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const modalities = stats === null ? [] : Object.entries(stats.by_modality ?? {});
  const withinBudget = stats !== null && stats.mean_ms > 0 && stats.mean_ms <= LATENCY_BUDGET_MS;

  return (
    <Card
      title={t("admin.latencyTitle")}
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
      <p className={styles.lead}>{t("admin.latencyIntro")}</p>

      {problem === null ? null : (
        <ErrorState
          problem={problem}
          onRetry={() => {
            void load();
          }}
        />
      )}
      {loading && stats === null ? <Spinner label={t("common.loading")} /> : null}

      {stats === null ? null : stats.n === 0 ? (
        <EmptyState title={t("admin.latencyEmpty")} />
      ) : (
        <>
          <ul className={styles.metrics}>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.latencyMean")}</span>
              <span className={styles.metricValue}>{ms(stats.mean_ms)}</span>
              <Chip tone={withinBudget ? "ok" : "warn"}>
                {withinBudget ? t("admin.latencyWithinBudget") : t("admin.latencyOverBudget")}
              </Chip>
            </li>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.latencyP50")}</span>
              <span className={styles.metricValue}>{ms(stats.p50_ms)}</span>
            </li>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.latencyP95")}</span>
              <span className={styles.metricValue}>{ms(stats.p95_ms)}</span>
            </li>
            <li className={styles.metric}>
              <span className={styles.metricLabel}>{t("admin.latencyN")}</span>
              <span className={styles.metricValue}>{stats.n}</span>
            </li>
          </ul>

          {modalities.length === 0 ? null : (
            <>
              <h3 className={styles.subheading}>{t("admin.latencyByModality")}</h3>
              <dl className={styles.meta}>
                {modalities.map(([name, value]) => (
                  <div className={styles.metaRow} key={name}>
                    <dt>{modalityLabel(name)}</dt>
                    <dd>{ms(value)}</dd>
                  </div>
                ))}
              </dl>
            </>
          )}
        </>
      )}
    </Card>
  );
}
