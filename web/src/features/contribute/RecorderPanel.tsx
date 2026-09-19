/**
 * Record → check → upload → verdict, for one prompt.
 *
 * The panel is built around three promises made to the volunteer:
 *
 * * **Nothing is recorded without permission, and permission is asked for with an explanation.**
 *   The microphone is opened by a button whose label says so; the level meter then runs live, so
 *   the first take already has a usable gain.
 * * **A doubtful take is not uploaded silently.** Clipping, a too-quiet peak and a take longer
 *   than the 20 s the automatic control accepts are caught in the browser and shown *before* the
 *   upload, with the choice to re-record or to send anyway. A clean take goes straight up, because
 *   an extra confirmation click on every sentence is what makes a corpus campaign die.
 * * **No recording is lost.** A captured take stays in state until the server has answered, an
 *   upload failure keeps the bytes so the button can simply be pressed again, and leaving the page
 *   with an unsent take asks for confirmation (`beforeunload`).
 *
 * A rejection is a 200 with reasons, so it is rendered by {@link QcVerdict} and the same sentence
 * can be re-read immediately (api.md §2).
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import type { PromptOut, RecordingOut } from "@/api/types";
import { Button, Card, Chip, ErrorState, Spinner } from "@/components/ui";
import { t } from "@/i18n";
import { deleteRecording, uploadRecording } from "./api";
import { MicMeter } from "./MicMeter";
import { QcVerdict } from "./QcVerdict";
import {
  CLIPPING_LIMIT,
  QC_MAX_DURATION_S,
  QUIET_PEAK,
  useMicRecorder,
  type Take,
} from "./useMicRecorder";
import styles from "./RecorderPanel.module.css";

/** Why a finished take is being held back instead of uploaded straight away. */
type Concern = "clipping" | "quiet" | "tooLong";

/** Everything the browser can tell about a take before the server sees it. */
function concernsOf(take: Take): Concern[] {
  const concerns: Concern[] = [];
  if (take.clippingRatio > CLIPPING_LIMIT) {
    concerns.push("clipping");
  }
  if (take.peak < QUIET_PEAK) {
    concerns.push("quiet");
  }
  if (take.seconds > QC_MAX_DURATION_S) {
    concerns.push("tooLong");
  }
  return concerns;
}

const CONCERN_TEXT: Record<Concern, () => string> = {
  clipping: () => t("contribute.clippingWarning"),
  quiet: () => t("contribute.quietWarning"),
  tooLong: () => t("contribute.tooLongWarning"),
};

export interface RecorderPanelProps {
  /** The sentence being recorded. */
  prompt: PromptOut;
  /** Participant token for `X-Participant-Token`, or `null` for a signed-in contributor. */
  token: string | null;
  /** Called after every accepted upload, so the page can refresh the progress figures. */
  onAccepted?: (recording: RecordingOut) => void;
  /** Called after an accepted recording was deleted again, so the same figures stay honest. */
  onDeleted?: (recordingId: string) => void;
  /** Move to the next sentence. */
  onNext: () => void;
  /** Skip this sentence without recording it. */
  onSkip: () => void;
}

/** The recording loop for one prompt. */
export function RecorderPanel({
  prompt,
  token,
  onAccepted,
  onDeleted,
  onNext,
  onSkip,
}: RecorderPanelProps): JSX.Element {
  const mic = useMicRecorder();
  const [take, setTake] = useState<Take | null>(null);
  const [concerns, setConcerns] = useState<Concern[]>([]);
  const [uploading, setUploading] = useState(false);
  const [verdict, setVerdict] = useState<RecordingOut | null>(null);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const pending = take !== null || uploading;

  // A take that has not reached the server yet only exists in this tab: warn before it is lost.
  useEffect(() => {
    if (!pending) {
      return undefined;
    }
    const warn = (event: BeforeUnloadEvent): void => {
      event.preventDefault();
      event.returnValue = t("contribute.unsavedWarning");
    };
    globalThis.addEventListener("beforeunload", warn);
    return () => {
      globalThis.removeEventListener("beforeunload", warn);
    };
  }, [pending]);

  const send = useCallback(
    async (wav: Blob): Promise<void> => {
      setUploading(true);
      setProblem(null);
      const { data, problem: failure } = await uploadRecording(prompt.id, wav, token);
      setUploading(false);
      if (data === null) {
        // Keep the bytes: the volunteer presses the same button again instead of re-reading.
        setProblem(failure);
        return;
      }
      setTake(null);
      setConcerns([]);
      setVerdict(data);
      if (data.accepted) {
        onAccepted?.(data);
      }
    },
    [onAccepted, prompt.id, token],
  );

  const startTake = useCallback(async (): Promise<void> => {
    setVerdict(null);
    setProblem(null);
    setNotice(null);
    setTake(null);
    setConcerns([]);
    await mic.start();
  }, [mic]);

  const stopTake = useCallback(async (): Promise<void> => {
    const finished = await mic.stop();
    if (finished === null) {
      return;
    }
    const found = concernsOf(finished);
    if (found.length === 0) {
      await send(finished.wav);
      return;
    }
    setTake(finished);
    setConcerns(found);
  }, [mic, send]);

  const discard = useCallback((): void => {
    setTake(null);
    setConcerns([]);
    setProblem(null);
    void startTake();
  }, [startTake]);

  const removeRecording = useCallback(async (): Promise<void> => {
    const id = verdict?.id;
    if (id === undefined || id === null) {
      return;
    }
    const { problem: failure } = await deleteRecording(id, token);
    if (failure !== null) {
      setProblem(failure);
      return;
    }
    setVerdict(null);
    setNotice(t("contribute.deleted"));
    onDeleted?.(id);
  }, [onDeleted, token, verdict]);

  const recording = mic.phase === "recording";
  const seconds = Math.round(mic.seconds * 10) / 10;

  return (
    <Card
      title={t("contribute.micTitle")}
      actions={
        recording ? (
          <Chip tone="critical" dot large>
            {t("voice.recording")}
          </Chip>
        ) : null
      }
    >
      <div className={styles.panel}>
        {mic.permission === "granted" ? null : (
          <p className={styles.explain}>{t("contribute.micExplain")}</p>
        )}

        {mic.permission === "granted" ? (
          <p className={styles.notice}>{t("contribute.micReady")}</p>
        ) : (
          <div className={styles.permission}>
            <Button
              variant="secondary"
              onClick={() => {
                void mic.open();
              }}
              loading={mic.phase === "opening"}
            >
              {mic.permission === "denied" ? t("contribute.micRetry") : t("contribute.micAllow")}
            </Button>
            {mic.phase === "opening" ? (
              <span className={styles.duration}>{t("contribute.micRequesting")}</span>
            ) : null}
            {mic.error === null ? null : <ErrorState problem={null} message={mic.error} inline />}
          </div>
        )}

        <MicMeter
          level={mic.level}
          recording={recording}
          clippingRatio={recording && mic.clippingRatio > CLIPPING_LIMIT ? mic.clippingRatio : 0}
        />

        <div className={styles.controls}>
          {recording ? (
            <Button
              variant="danger"
              size="lg"
              onClick={() => {
                void stopTake();
              }}
            >
              {t("voice.stop")}
            </Button>
          ) : (
            <Button
              variant="primary"
              size="lg"
              disabled={uploading || take !== null}
              onClick={() => {
                void startTake();
              }}
            >
              {t("voice.start")}
            </Button>
          )}
          <Button variant="ghost" disabled={recording || uploading} onClick={onSkip}>
            {t("contribute.skip")}
          </Button>
          <span className={styles.duration}>
            {recording || seconds > 0
              ? t("voice.duration", { seconds })
              : t("contribute.recordHint")}
          </span>
          {mic.phase === "encoding" || uploading ? (
            <Spinner
              size="sm"
              label={uploading ? t("contribute.uploading") : t("voice.processing")}
            />
          ) : null}
        </div>

        {take === null ? null : (
          <div className={styles.pending} role="status">
            <p className={styles.pendingTitle}>{t("contribute.pendingNotice")}</p>
            <ul className={styles.concerns}>
              {concerns.map((concern) => (
                <li key={concern}>{CONCERN_TEXT[concern]()}</li>
              ))}
            </ul>
            <div className={styles.controls}>
              <Button
                variant="primary"
                loading={uploading}
                onClick={() => {
                  void send(take.wav);
                }}
              >
                {t("common.send")}
              </Button>
              <Button variant="secondary" disabled={uploading} onClick={discard}>
                {t("contribute.retake")}
              </Button>
            </div>
          </div>
        )}

        {problem === null ? null : take === null ? (
          <ErrorState problem={problem} />
        ) : (
          <ErrorState
            problem={problem}
            onRetry={() => {
              void send(take.wav);
            }}
          />
        )}

        {notice === null ? null : (
          <p className={styles.notice} role="status">
            {notice}
          </p>
        )}

        {verdict === null ? null : (
          <>
            <QcVerdict
              recording={verdict}
              onDelete={() => {
                void removeRecording();
              }}
            />
            <div className={styles.controls}>
              {verdict.accepted ? (
                <Button variant="primary" size="lg" onClick={onNext}>
                  {t("contribute.next")}
                </Button>
              ) : (
                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => {
                    void startTake();
                  }}
                >
                  {t("contribute.retake")}
                </Button>
              )}
              {verdict.accepted ? null : (
                <Button variant="ghost" onClick={onSkip}>
                  {t("contribute.skip")}
                </Button>
              )}
            </div>
          </>
        )}
      </div>
    </Card>
  );
}
