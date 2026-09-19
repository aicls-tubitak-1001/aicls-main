/**
 * The quality-control review queue (`GET /admin/qc/queue`, `POST /admin/qc/{id}/review`).
 *
 * Each row is one recording the automatic checks have already judged; the human decision is the
 * one the training manifest will use. The three verdicts the API accepts are three buttons —
 * accept, reject, and "fix the transcript and accept" — with the corrected text in a field beside
 * them, because a mis-read sentence is far more often a transcript problem than a bad recording.
 *
 * A decided row leaves the list immediately (it is no longer pending) and the count is restated,
 * so a reviewer working through a queue always knows how much is left.
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import {
  Button,
  Card,
  Chip,
  EmptyState,
  ErrorState,
  Field,
  Input,
  Select,
  Spinner,
} from "@/components/ui";
import { t, type MessageKey } from "@/i18n";
import {
  fetchQcQueue,
  reviewRecording,
  REVIEW_STATUSES,
  type QcQueueItem,
  type QcVerdictValue,
} from "./api";
import styles from "./Admin.module.css";

/** Turkish label of each review status. */
const STATUS_LABELS: Record<string, MessageKey> = {
  pending: "admin.qcStatus.pending",
  accepted: "admin.qcStatus.accepted",
  rejected: "admin.qcStatus.rejected",
  fixed: "admin.qcStatus.fixed",
};

function statusLabel(status: string): string {
  const key = STATUS_LABELS[status];
  return key === undefined ? status : t(key);
}

/** The review queue. */
export function QcQueuePanel(): JSX.Element {
  const [status, setStatus] = useState<string>("pending");
  const [items, setItems] = useState<QcQueueItem[]>([]);
  const [fixes, setFixes] = useState<Record<string, string>>({});
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setLoading(true);
    const { data, problem: failure } = await fetchQcQueue(status);
    setLoading(false);
    if (data === null) {
      setProblem(failure);
      return;
    }
    setProblem(null);
    setItems(data);
  }, [status]);

  useEffect(() => {
    void load();
  }, [load]);

  const decide = async (item: QcQueueItem, verdict: QcVerdictValue): Promise<void> => {
    setBusyId(item.id);
    setProblem(null);
    const { problem: failure } = await reviewRecording(item.id, verdict, fixes[item.id]);
    setBusyId(null);
    if (failure !== null) {
      setProblem(failure);
      return;
    }
    setItems((current) => current.filter((row) => row.id !== item.id));
    setNotice(t("admin.qcReviewed"));
  };

  return (
    <Card
      title={t("admin.qcTitle")}
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
      <p className={styles.lead}>{t("admin.qcIntro")}</p>

      <div className={styles.filters}>
        <Field label={t("admin.qcStatusFilter")}>
          {(control) => (
            <Select
              {...control}
              value={status}
              onChange={(event) => {
                setStatus(event.target.value);
              }}
            >
              {REVIEW_STATUSES.map((value) => (
                <option key={value} value={value}>
                  {statusLabel(value)}
                </option>
              ))}
            </Select>
          )}
        </Field>
      </div>

      {problem === null ? null : (
        <ErrorState
          problem={problem}
          onRetry={() => {
            void load();
          }}
        />
      )}

      {notice === null ? null : (
        <p className={styles.notice} role="status">
          {notice}
        </p>
      )}

      {loading ? <Spinner label={t("common.loading")} /> : null}

      {!loading && items.length === 0 ? <EmptyState title={t("admin.qcEmpty")} /> : null}

      <ul className={styles.rows}>
        {items.map((item) => (
          <li key={item.id} className={styles.row}>
            <div className={styles.rowHead}>
              <span className={styles.rowId}>{item.id}</span>
              <Chip tone={item.qc_passed ? "ok" : "warn"}>
                {t("admin.qcAuto")}:{" "}
                {item.qc_passed ? t("contribute.checkPassed") : t("contribute.checkFailed")}
              </Chip>
              <Chip tone="neutral">{statusLabel(item.review_status)}</Chip>
            </div>

            <dl className={styles.meta}>
              <div className={styles.metaRow}>
                <dt>{t("admin.qcSpeaker")}</dt>
                <dd>{item.speaker_id}</dd>
              </div>
              <div className={styles.metaRow}>
                <dt>{t("admin.qcPrompt")}</dt>
                <dd>{item.prompt_id}</dd>
              </div>
              <div className={styles.metaRow}>
                <dt>{t("admin.qcDuration")}</dt>
                <dd>
                  {item.duration_s.toFixed(2)} {t("common.seconds")}
                </dd>
              </div>
              <div className={styles.metaRow}>
                <dt>{t("admin.qcTranscript")}</dt>
                <dd>{item.transcript_final === "" ? t("common.none") : item.transcript_final}</dd>
              </div>
            </dl>

            <Field label={t("admin.qcTranscriptFix")}>
              {(control) => (
                <Input
                  {...control}
                  type="text"
                  maxLength={500}
                  value={fixes[item.id] ?? ""}
                  onChange={(event) => {
                    const next = event.target.value;
                    setFixes((current) => ({ ...current, [item.id]: next }));
                  }}
                />
              )}
            </Field>

            <div className={styles.rowActions}>
              <Button
                variant="primary"
                loading={busyId === item.id}
                onClick={() => {
                  void decide(item, "accepted");
                }}
              >
                {t("admin.qcAccept")}
              </Button>
              <Button
                variant="secondary"
                disabled={busyId === item.id || (fixes[item.id] ?? "") === ""}
                onClick={() => {
                  void decide(item, "fixed");
                }}
              >
                {t("admin.qcFix")}
              </Button>
              <Button
                variant="danger"
                disabled={busyId === item.id}
                onClick={() => {
                  void decide(item, "rejected");
                }}
              >
                {t("admin.qcReject")}
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}
