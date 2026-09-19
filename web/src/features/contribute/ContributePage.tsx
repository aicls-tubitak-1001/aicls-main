/**
 * "Model Gelişimine Katkıda Bulun" — the voice data-collection tab (İP-3, api.md §3.8).
 *
 * The whole 50-hour corpus depends on volunteers finishing a sitting here, so the page is built
 * as a short, honest path: read why this is being collected, read and accept the consent text,
 * then a loop of *one* sentence at a time with the end of the session visible at all times.
 *
 * Three states are first-class rather than error screens:
 *
 * * **the ethics gate** — every collection endpoint answers `503 GATE_BLOCKED` until an approval
 *   reference is configured, so the page says "henüz açık değil" calmly and still offers
 *   withdrawal, which the backend deliberately keeps outside the gate;
 * * **consent not given yet** — nothing recordable is rendered at all, which mirrors the
 *   server-side gate instead of trusting the UI;
 * * **a rejected take** — a 200 with Turkish reasons, rendered as a verdict the volunteer can act
 *   on immediately.
 */
import { useCallback, useEffect, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import type { ConsentTextOut, ContributeProgressOut, PromptOut } from "@/api/types";
import { useAuth } from "@/app/auth";
import { Button, Card, Chip, EmptyState, ErrorState, Spinner } from "@/components/ui";
import { t } from "@/i18n";
import { fetchConsentText, fetchNextPrompts, fetchProgress, isGateBlocked } from "./api";
import { CommunityCounter } from "./CommunityCounter";
import { ConsentForm } from "./ConsentForm";
import { ProgressBar } from "./ProgressBar";
import { PromptCard } from "./PromptCard";
import { RecorderPanel } from "./RecorderPanel";
import { useParticipant } from "./participant";
import { WithdrawDialog } from "./WithdrawDialog";
import styles from "./ContributePage.module.css";

/** The collection tab. */
export function ContributePage(): JSX.Element {
  const participant = useParticipant();
  const { can } = useAuth();

  const [consent, setConsent] = useState<ConsentTextOut | null>(null);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);
  const [gateBlocked, setGateBlocked] = useState(false);
  const [loading, setLoading] = useState(false);

  const [prompts, setPrompts] = useState<PromptOut[]>([]);
  const [index, setIndex] = useState(0);
  const [progress, setProgress] = useState<ContributeProgressOut | null>(null);
  const [sessionAccepted, setSessionAccepted] = useState(0);

  const [withdrawOpen, setWithdrawOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const loadConsent = useCallback(async (): Promise<void> => {
    setLoading(true);
    const { data, problem: failure } = await fetchConsentText();
    setLoading(false);
    if (data === null) {
      setGateBlocked(isGateBlocked(failure));
      setProblem(isGateBlocked(failure) ? null : failure);
      return;
    }
    setGateBlocked(false);
    setProblem(null);
    setConsent(data);
  }, []);

  const loadPrompts = useCallback(async (token: string | null): Promise<void> => {
    const { data, problem: failure } = await fetchNextPrompts(token);
    if (data === null) {
      setGateBlocked(isGateBlocked(failure));
      setProblem(isGateBlocked(failure) ? null : failure);
      return;
    }
    setPrompts(data);
    setIndex(0);
  }, []);

  const loadProgress = useCallback(async (token: string | null): Promise<void> => {
    const { data } = await fetchProgress(token);
    setProgress(data);
  }, []);

  useEffect(() => {
    if (participant !== null) {
      return;
    }
    void loadConsent();
  }, [participant, loadConsent]);

  useEffect(() => {
    if (participant === null) {
      return;
    }
    void loadPrompts(participant.token);
    void loadProgress(participant.token);
  }, [participant, loadPrompts, loadProgress]);

  // The participant itself arrives through the store (`useParticipant`); the callback only resets
  // what belongs to a sitting.
  const onConsented = useCallback((): void => {
    setNotice(null);
    setSessionAccepted(0);
  }, []);

  const advance = useCallback(
    (token: string | null): void => {
      if (index + 1 >= prompts.length) {
        void loadPrompts(token);
        return;
      }
      setIndex(index + 1);
    },
    [index, loadPrompts, prompts.length],
  );

  const header = (
    <header className={styles.intro}>
      <h1 className={styles.title}>{t("contribute.title")}</h1>
      <p className={styles.subtitle}>{t("contribute.subtitle")}</p>
      <p className={styles.lead}>{t("contribute.intro")}</p>
    </header>
  );

  // A withdrawal takes the page back to the consent step, so the confirmation has to survive that
  // switch: it is rendered by every branch, not only by the one that raised it.
  const noticeLine =
    notice === null ? null : (
      <p className={styles.notice} role="status">
        {notice}
      </p>
    );

  /* ------------------------------------------------------------------ the gate is closed --- */
  if (gateBlocked) {
    return (
      <section className={styles.page} aria-labelledby="contribute-title">
        <header className={styles.intro}>
          <h1 className={styles.title} id="contribute-title">
            {t("contribute.title")}
          </h1>
        </header>
        <Card>
          <EmptyState
            title={t("contribute.gateTitle")}
            description={t("contribute.gateBody")}
            icon="⏳"
          />
          <p className={styles.gateNote}>{t("contribute.gateWithdrawNote")}</p>
          {participant === null ? null : (
            <Button
              variant="secondary"
              onClick={() => {
                setWithdrawOpen(true);
              }}
            >
              {t("contribute.withdraw")}
            </Button>
          )}
        </Card>
        <WithdrawDialog
          open={withdrawOpen}
          token={participant?.token ?? null}
          onClose={() => {
            setWithdrawOpen(false);
          }}
          onWithdrawn={() => {
            setNotice(t("contribute.withdrawDone"));
          }}
        />
        {noticeLine}
      </section>
    );
  }

  /* ------------------------------------------------------------------------ consent step --- */
  if (participant === null) {
    return (
      <section className={styles.page}>
        {header}
        {noticeLine}
        <Card title={t("contribute.whyTitle")} muted>
          <p className={styles.lead}>{t("contribute.whyBody")}</p>
        </Card>
        {problem === null ? null : (
          <ErrorState
            problem={problem}
            onRetry={() => {
              void loadConsent();
            }}
          />
        )}
        {consent === null ? (
          loading ? (
            <Spinner size="lg" label={t("common.loading")} />
          ) : null
        ) : (
          <ConsentForm consent={consent} onConsented={onConsented} />
        )}
      </section>
    );
  }

  /* --------------------------------------------------------------------- recording loop --- */
  const prompt = prompts[index];
  return (
    <section className={styles.page}>
      {header}

      <Card
        title={t("contribute.progressTitle")}
        actions={
          <Chip tone="accent">{t("contribute.pseudonym", { name: participant.pseudonym })}</Chip>
        }
        footer={<p className={styles.hint}>{t("contribute.pseudonymHint")}</p>}
      >
        <ProgressBar sessionAccepted={sessionAccepted} progress={progress} />
      </Card>

      {problem === null ? null : (
        <ErrorState
          problem={problem}
          onRetry={() => {
            void loadPrompts(participant.token);
          }}
        />
      )}

      {prompt === undefined ? (
        <Card>
          <EmptyState
            title={t("contribute.promptsEmpty")}
            action={
              <Button
                variant="secondary"
                onClick={() => {
                  void loadPrompts(participant.token);
                }}
              >
                {t("contribute.promptsReload")}
              </Button>
            }
          />
        </Card>
      ) : (
        <>
          <PromptCard prompt={prompt} index={index + 1} total={prompts.length} />
          <RecorderPanel
            key={prompt.id}
            prompt={prompt}
            token={participant.token}
            onAccepted={() => {
              setSessionAccepted((count) => count + 1);
              void loadProgress(participant.token);
            }}
            onDeleted={() => {
              setSessionAccepted((count) => Math.max(0, count - 1));
              void loadProgress(participant.token);
            }}
            onNext={() => {
              advance(participant.token);
            }}
            onSkip={() => {
              advance(participant.token);
            }}
          />
        </>
      )}

      {can("instructor", "admin") ? <CommunityCounter /> : null}

      <Card title={t("contribute.withdrawTitle")} muted>
        <p className={styles.lead}>{t("contribute.withdrawExplain")}</p>
        <Button
          variant="secondary"
          onClick={() => {
            setWithdrawOpen(true);
          }}
        >
          {t("contribute.withdraw")}
        </Button>
      </Card>

      {noticeLine}

      <WithdrawDialog
        open={withdrawOpen}
        token={participant.token}
        onClose={() => {
          setWithdrawOpen(false);
        }}
        onWithdrawn={() => {
          setPrompts([]);
          setProgress(null);
          setSessionAccepted(0);
          setNotice(t("contribute.withdrawDone"));
        }}
      />
    </section>
  );
}
