/**
 * Audience C: what has actually been measured.
 *
 * This is the page that decides whether AI-CLS reads as a research platform or as a demo. Its rule
 * is simple and absolute: every number is shown with the thing that limits it. A value produced by
 * an untrained stub carries the sentence saying so; a value measured on a corpus the project wrote
 * itself carries the sentence saying so; a run produced on a dirty tree is marked as unquotable.
 * Nothing is rounded in a favourable direction and no failing target is omitted — the ASR row
 * reports a word error rate above one, in the same table and the same type size as the rows that
 * pass.
 *
 * The data is a prop with a default, so the unit test renders a fixture and asserts on the values
 * rather than on the committed snapshot.
 */
import { Chip, type ChipTone } from "@/components/ui/Chip";
import { t } from "@/i18n";
import { Callout } from "./Callout";
import {
  CONTRACT_EVIDENCE_KEY,
  CONTRACT_TARGETS,
  EVIDENCE_HINT_KEY,
  EVIDENCE_LABEL_KEY,
  METRICS_SNAPSHOT,
  formatInterval,
  formatMetricValue,
  shortSha,
  type ContractEvidence,
  type ContractTarget,
  type EvidenceLevel,
  type GuideRun,
  type MetricsSnapshot,
} from "./metrics";
import { Section } from "./Section";
import { STATUS } from "./sections";
import styles from "./GuideArticle.module.css";
import status from "./StatusGuide.module.css";
import tables from "./tables.module.css";

const EVIDENCE_TONE: Readonly<Record<EvidenceLevel, ChipTone>> = {
  evidence: "ok",
  regression: "warn",
  stub: "critical",
};

const CONTRACT_TONE: Readonly<Record<ContractEvidence, ChipTone>> = {
  none: "neutral",
  stub: "critical",
  regression: "warn",
  partial: "info",
};

const EVIDENCE_ORDER: readonly EvidenceLevel[] = ["evidence", "regression", "stub"];

const NEXT_STEPS = [
  "guide.status.next.ocr",
  "guide.status.next.asr",
  "guide.status.next.corpus",
  "guide.status.next.latency",
  "guide.status.next.sus",
  "guide.status.next.citable",
] as const;

/** One evaluation run: provenance first, then its numbers. */
function RunCard({ run }: { run: GuideRun }): JSX.Element {
  const title = t(run.titleKey);
  const headingId = `kosum-${run.id}`;
  return (
    <article className={status.run} aria-labelledby={headingId} data-testid={`guide-run-${run.id}`}>
      <div className={status.runHeader}>
        <h3 id={headingId} className={status.runTitle}>
          {title}
        </h3>
        <Chip tone={EVIDENCE_TONE[run.evidence]} title={t(EVIDENCE_HINT_KEY[run.evidence])}>
          {t(EVIDENCE_LABEL_KEY[run.evidence])}
        </Chip>
      </div>

      <p className={status.meta}>
        {t("guide.status.meta", {
          dataset: run.datasetVersion,
          fingerprint: run.datasetFingerprint.slice(0, 19),
          sha: shortSha(run.gitSha),
          at: run.generatedAt,
        })}
        {" · "}
        {t("guide.status.col.n")} = {run.n}
        {run.modelVersion === null
          ? null
          : ` · ${t("guide.status.modelVersion", { version: run.modelVersion })}`}
      </p>

      {run.citable ? null : (
        <Callout tone="warn" label={t("guide.status.notCitableLabel")}>
          <p>{t("guide.status.notCitable")}</p>
        </Callout>
      )}

      {run.noticeKeys.map((key) => (
        <Callout key={key} tone="critical" label={t(EVIDENCE_LABEL_KEY[run.evidence])}>
          <p>{t(key)}</p>
        </Callout>
      ))}

      <div className={`${tables.wrap} scroll-x`}>
        <table className={tables.table}>
          <caption className={tables.caption}>{title}</caption>
          <thead>
            <tr>
              <th scope="col">{t("guide.status.col.metric")}</th>
              <th scope="col">{t("guide.status.col.value")}</th>
              <th scope="col">{t("guide.status.col.ci")}</th>
              <th scope="col">{t("guide.status.col.target")}</th>
              <th scope="col">{t("guide.status.col.verdict")}</th>
            </tr>
          </thead>
          <tbody>
            {run.metrics.map((metric) => (
              <tr key={metric.id}>
                <th scope="row">{t(metric.labelKey)}</th>
                <td className={tables.numeric} data-testid={`metric-${metric.id}`}>
                  {formatMetricValue(metric.value, metric.unit)}
                </td>
                <td className={tables.numeric}>
                  {metric.ci95 === null ? "—" : formatInterval(metric.ci95, metric.unit)}
                </td>
                <td className={tables.numeric}>{metric.target?.text ?? "—"}</td>
                <td>
                  {metric.target === null ? (
                    "—"
                  ) : (
                    <Chip tone={metric.target.met ? "ok" : "critical"}>
                      {metric.target.met
                        ? t("guide.status.verdict.met")
                        : t("guide.status.verdict.notMet")}
                    </Chip>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

export interface StatusGuideProps {
  /** Measured runs; defaults to the committed snapshot. */
  readonly snapshot?: MetricsSnapshot;
  /** Contractual targets; defaults to the proposal's seven. */
  readonly contract?: readonly ContractTarget[];
}

/** The status page's article body. */
export function StatusGuide({
  snapshot = METRICS_SNAPSHOT,
  contract = CONTRACT_TARGETS,
}: StatusGuideProps = {}): JSX.Element {
  return (
    <>
      <p className={styles.lead}>{t("guide.status.lead")}</p>

      <Section {...STATUS["nasil-okunmali"]}>
        <p>{t("guide.status.how.p1")}</p>
        <p>{t("guide.status.how.p2")}</p>
        <Callout tone="warn" label={t("guide.status.notCitableLabel")}>
          <p>{t("guide.status.how.p3")}</p>
        </Callout>
        <dl className={styles.terms} data-testid="guide-evidence-legend">
          {EVIDENCE_ORDER.map((level) => (
            <div key={level} className={styles.termRow}>
              <dt className={styles.term}>
                <Chip tone={EVIDENCE_TONE[level]}>{t(EVIDENCE_LABEL_KEY[level])}</Chip>
              </dt>
              <dd className={styles.definition}>{t(EVIDENCE_HINT_KEY[level])}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section {...STATUS["olculen-degerler"]}>
        {snapshot.runs.length === 0 ? (
          <p>{t("guide.status.empty")}</p>
        ) : (
          snapshot.runs.map((run) => <RunCard key={run.id} run={run} />)
        )}
        <p className={styles.footnote}>
          {t("guide.sourceLabel")}:{" "}
          <code className={styles.inlineCode}>{snapshot.definitionsRef}</code>
        </p>
      </Section>

      <Section {...STATUS["sozlesme-olcutleri"]}>
        <p>{t("guide.status.contract.intro")}</p>
        <div className={`${tables.wrap} scroll-x`}>
          <table className={tables.table} data-testid="guide-contract-table">
            <caption className={tables.caption}>{t("guide.status.contract.heading")}</caption>
            <thead>
              <tr>
                <th scope="col">{t("guide.status.contract.colTarget")}</th>
                <th scope="col">{t("guide.status.contract.colToday")}</th>
                <th scope="col">{t("guide.status.contract.colEvidence")}</th>
              </tr>
            </thead>
            <tbody>
              {contract.map((row) => (
                <tr key={row.id}>
                  <th scope="row">{t(row.labelKey)}</th>
                  <td>{t(row.todayKey)}</td>
                  <td>
                    <Chip tone={CONTRACT_TONE[row.evidence]}>
                      {t(CONTRACT_EVIDENCE_KEY[row.evidence])}
                    </Chip>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section {...STATUS["kanita-giden-yol"]}>
        <p>{t("guide.status.next.p1")}</p>
        <ul className={styles.list}>
          {NEXT_STEPS.map((key) => (
            <li key={key}>{t(key)}</li>
          ))}
        </ul>
      </Section>
    </>
  );
}
