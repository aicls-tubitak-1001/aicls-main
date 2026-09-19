/**
 * The screen a lecturer actually teaches from.
 *
 * One full-height surface: a status strip, a collapsible rail with the session's history, the board
 * (the result, and the pen panel beside or over it), the book rail, and the command bar docked at
 * the bottom where a hand can always reach it. Presentation mode collapses both rails and scales
 * the type through `data-density`, leaving the command bar and the answer alone on the wall.
 *
 * The rules this file exists to enforce:
 *
 * - **Nothing interrupts.** The only modal is the shortcut sheet, and only because the lecturer
 *   asked for it. A clarification, a confirmation and a failure are all inline panels above the
 *   result; the previous answer stays visible behind every one of them.
 * - **Every modality lands in the same pipeline.** Typing, speaking and writing all end in
 *   `useWorkspace`, which owns the request, the outcome and the latency marks.
 * - **`kind: "ui"` is an instruction, not an answer.** "Tuvali temizle" clears the canvas, "son
 *   komutu yinele" re-runs the previous *computational* command; neither replaces the result.
 * - **The keyboard is complete.** `/` focuses the bar, Space is push-to-talk, `Ctrl + Enter`
 *   submits, Escape backs out, `Alt + …` covers the rest, and `?` shows the whole table.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { AiclsProblem } from "@/api/problem";
import type { ClarificationCandidateOut, ComputeOut, ContentUnitOut, HitOut } from "@/api/types";
import { useSession } from "@/app/auth";
import { useDensity } from "@/app/density";
import { useOnline } from "@/app/online";
import { isTypingTarget, useShortcuts, type ShortcutHandlers } from "@/app/shortcuts";
import { Button } from "@/components/ui/Button";
import { Chip } from "@/components/ui/Chip";
import { t } from "@/i18n";
import { cx } from "@/lib/cx";
import { CommandBar } from "@/features/command/CommandBar";
import { CommandPreview } from "@/features/command/CommandPreview";
import { FirstRunCard } from "@/features/guide/FirstRunCard";
import { useVoiceCommand } from "@/features/command/useVoiceCommand";
import { needsConfirmation, useVoiceConfirmPreference } from "@/features/command/voicePreference";
import { BookPanel } from "@/features/book/BookPanel";
import { BookUnitView } from "@/features/book/BookUnitView";
import { HistoryPanel } from "@/features/history/HistoryPanel";
import { ClarificationCard } from "@/features/results/ClarificationCard";
import { ProblemNotice } from "@/features/results/ProblemNotice";
import { ResultCard } from "@/features/results/ResultCard";
import { InkPanel } from "./InkPanel";
import { ShortcutSheet } from "./ShortcutSheet";
import { useWorkspace } from "./useWorkspace";
import styles from "./WorkspacePage.module.css";

/** A voice outcome waiting for the lecturer's confirmation (ADR-0014, api.md §3.6). */
interface VoicePreview {
  outcome: ComputeOut;
  latencyKey: string;
}

/** The main working surface. */
export function WorkspacePage(): JSX.Element {
  const ws = useWorkspace();
  const { isPresentation, toggleDensity } = useDensity();
  const online = useOnline();
  const authenticated = useSession((session) => session.accessToken !== null);
  const { always: alwaysAsk, setAlways } = useVoiceConfirmPreference();

  // A board that is already in presentation mode when the page loads starts with a clear wall.
  const [railOpen, setRailOpen] = useState(() => !isPresentation);
  const [penOpen, setPenOpen] = useState(true);
  const [penOverlay, setPenOverlay] = useState(false);
  const [clearToken, setClearToken] = useState(0);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [voicePreview, setVoicePreview] = useState<VoicePreview | null>(null);
  const [pinned, setPinned] = useState<ContentUnitOut | null>(null);
  const commandRef = useRef<HTMLInputElement | null>(null);

  const {
    outcome,
    problem,
    stages,
    busy,
    latencyKey,
    workspaceId,
    inputLatex,
    commandText,
    run,
    applyOutcome,
    repeatLast,
    setCommandText,
    setInputLatex,
    setProblem,
    clearBoard,
    bookOpen,
    setBookOpen,
    historyToken,
    uiAction,
    uiActionId,
    lastCommandText,
  } = ws;

  const focusCommand = useCallback((): void => {
    const field = commandRef.current;
    field?.focus();
    field?.select();
  }, []);

  /* -- voice ----------------------------------------------------------------------------- */

  const onVoiceOutcome = useCallback(
    (next: ComputeOut, key: string): void => {
      const heard = next.transcript ?? "";
      if (heard !== "") {
        // Keep the transcript in the bar: a clarification answer re-submits this same sentence,
        // and "düzelt" must land on editable text rather than on an empty field.
        setCommandText(heard);
      }
      if (next.kind === "clarification" || !needsConfirmation(next.confidence, alwaysAsk)) {
        applyOutcome(next, { latencyKey: key });
        return;
      }
      setVoicePreview({ outcome: next, latencyKey: key });
    },
    [alwaysAsk, applyOutcome, setCommandText],
  );

  const voice = useVoiceCommand({
    workspaceId,
    lastLatex: inputLatex,
    onOutcome: onVoiceOutcome,
    onProblem: (failure: AiclsProblem) => {
      setProblem(failure);
    },
  });
  const voiceRef = useRef(voice);
  voiceRef.current = voice;

  const confirmVoice = useCallback((): void => {
    const preview = voicePreview;
    setVoicePreview(null);
    if (preview !== null) {
      // The answer already exists — the backend computed it before it asked. Committing it is what
      // makes confirmation cheap enough to be the default.
      applyOutcome(preview.outcome);
    }
  }, [applyOutcome, voicePreview]);

  const editVoice = useCallback((): void => {
    setVoicePreview(null);
    focusCommand();
  }, [focusCommand]);

  /* -- client-side actions (`kind: "ui"`) ------------------------------------------------- */

  useEffect(() => {
    if (uiAction === null || uiActionId === 0) {
      return;
    }
    switch (uiAction) {
      case "clear":
        setClearToken((token) => token + 1);
        setInputLatex("");
        break;
      case "undo":
        clearBoard();
        break;
      case "repeat_last":
        repeatLast();
        break;
      case "export":
        // Exporting is a two-parameter action (workspace + operation) and the buttons that carry
        // both live on the result card; the spoken form focuses them instead of guessing.
        focusCommand();
        break;
      default:
        break;
    }
    // `uiActionId` changes on every `ui` outcome, so the same action twice is acted on twice.
  }, [uiActionId, uiAction, clearBoard, focusCommand, repeatLast, setInputLatex]);

  /* -- presentation mode ------------------------------------------------------------------ */

  /*
   * Entering presentation mode clears the wall: both rails close so the answer and the command bar
   * are alone on the board. It closes them rather than forcing `display: none`, because every
   * control must keep telling the truth — the "Kitap" button and Alt + B still open the textbook
   * on the wall mid-lecture, and a `kind: "book"` answer ("türev tanımını getir") still shows the
   * unit it fetched instead of silently claiming a panel opened. Leaving the mode restores the
   * desk layout's history rail.
   */
  const wasPresentation = useRef(isPresentation);
  useEffect(() => {
    if (wasPresentation.current === isPresentation) {
      return;
    }
    wasPresentation.current = isPresentation;
    if (isPresentation) {
      setRailOpen(false);
      setBookOpen(false);
      return;
    }
    setRailOpen(true);
  }, [isPresentation, setBookOpen]);

  /* -- keyboard -------------------------------------------------------------------------- */

  const handlers = useMemo<ShortcutHandlers>(
    () => ({
      focusCommand,
      toggleMic: () => {
        voiceRef.current.toggle();
      },
      clearCanvas: () => {
        setClearToken((token) => token + 1);
      },
      togglePresentation: toggleDensity,
      toggleBook: () => {
        setBookOpen(!bookOpen);
      },
      repeatLast,
      showHelp: () => {
        setShortcutsOpen(true);
      },
    }),
    [bookOpen, focusCommand, repeatLast, setBookOpen, toggleDensity],
  );

  useShortcuts(handlers, { enabled: !shortcutsOpen });

  // Space is push-to-talk while it is held, but only outside text fields and off buttons, where
  // Space is the platform's own "activate" key.
  const heldRef = useRef(false);
  useEffect(() => {
    const isActivationTarget = (target: EventTarget | null): boolean =>
      target instanceof HTMLElement &&
      ["button", "a", "summary", "input", "select", "textarea"].includes(
        target.tagName.toLowerCase(),
      );

    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape" && !isTypingTarget(event.target)) {
        setVoicePreview(null);
        return;
      }
      if (event.code !== "Space" || event.repeat) {
        return;
      }
      if (event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }
      if (isTypingTarget(event.target) || isActivationTarget(event.target)) {
        return;
      }
      if (!voiceRef.current.supported || voiceRef.current.active) {
        return;
      }
      event.preventDefault();
      heldRef.current = true;
      voiceRef.current.start();
    };

    const onKeyUp = (event: KeyboardEvent): void => {
      if (event.code !== "Space" || !heldRef.current) {
        return;
      }
      heldRef.current = false;
      event.preventDefault();
      voiceRef.current.stop();
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  /* -- derived ---------------------------------------------------------------------------- */

  const clarification = outcome?.kind === "clarification" ? (outcome.clarification ?? null) : null;

  const bookHits = useMemo<HitOut[]>(() => {
    if (outcome === null || outcome.kind !== "book") {
      return [];
    }
    return (outcome.book_units ?? []).map((unit) => ({
      unit,
      score: 1,
      stage: "command",
      matched_alias: null,
    }));
  }, [outcome]);

  const onChooseCandidate = useCallback(
    (candidate: ClarificationCandidateOut): void => {
      void run(commandText, { confirmIntent: candidate.intent });
    },
    [commandText, run],
  );

  const acceptRecognition = useCallback(
    (latex: string): void => {
      setInputLatex(latex);
      focusCommand();
    },
    [focusCommand, setInputLatex],
  );

  return (
    <div className={cx(styles.page, isPresentation && styles.presentation)}>
      <h1 className="visually-hidden">{t("workspace.heading")}</h1>

      <div className={styles.topbar}>
        <div className={styles.status}>
          <Chip tone={online ? "ok" : "warn"} dot>
            {online ? t("ws.connected") : t("errors.offline")}
          </Chip>
          {workspaceId === null && authenticated ? (
            <span className={styles.preparing}>{t("workspace.preparing")}</span>
          ) : null}
        </div>

        <div className={styles.topActions}>
          <Button
            variant="ghost"
            onClick={() => {
              setRailOpen(!railOpen);
            }}
            aria-pressed={railOpen}
          >
            {t("workspace.toggleRail")}
          </Button>
          <Button
            variant="ghost"
            onClick={() => {
              setBookOpen(!bookOpen);
            }}
            aria-pressed={bookOpen}
          >
            {t("nav.book")}
          </Button>
          <Button
            variant="ghost"
            onClick={() => {
              setPenOpen(!penOpen);
            }}
            aria-pressed={penOpen}
          >
            {t("workspace.togglePen")}
          </Button>
          {/* The presentation switch is in the application header, where it is reachable from
              every route — including the ones a lecturer walks to while the wall is still in
              presentation mode. Alt + S below drives the same shared state. */}
          <Button
            variant="ghost"
            onClick={() => {
              setShortcutsOpen(true);
            }}
          >
            {t("workspace.shortcutsButton")}
          </Button>
        </div>
      </div>

      {online ? null : (
        <p className={styles.offline} role="status">
          {t("errors.offline")}
        </p>
      )}

      <div className={styles.body}>
        <aside className={styles.rail} aria-label={t("workspace.railLabel")} hidden={!railOpen}>
          <HistoryPanel
            workspaceId={workspaceId}
            refreshToken={historyToken}
            onReuse={(operation) => {
              const text = (operation.input_summary ?? "").trim();
              const latex = operation.input_latex ?? "";
              if (latex !== "") {
                setInputLatex(latex);
              }
              if (text === "") {
                focusCommand();
                return;
              }
              void run(text, latex === "" ? {} : { lastLatex: latex });
            }}
          />
        </aside>

        <section className={styles.board} aria-label={t("workspace.boardLabel")}>
          {authenticated || !ws.ready ? null : (
            <p className={styles.signIn} role="status">
              {t("workspace.signInNeeded")}
            </p>
          )}

          {/* The guide's front door: a five-step card that hides itself once, undoably. */}
          <FirstRunCard />

          {voicePreview === null ? null : (
            <CommandPreview
              latex={voicePreview.outcome.input_latex ?? ""}
              command={voicePreview.outcome.command ?? {}}
              confidence={voicePreview.outcome.confidence ?? 0}
              transcript={voicePreview.outcome.transcript ?? null}
              intent={voicePreview.outcome.nlu?.intent ?? null}
              voice
              alwaysAsk={alwaysAsk}
              onAlwaysAskChange={setAlways}
              latencyKey={voicePreview.latencyKey}
              operationId={voicePreview.outcome.operation_id ?? null}
              onConfirm={confirmVoice}
              onEdit={editVoice}
            />
          )}

          {clarification === null ? null : (
            <ClarificationCard
              clarification={clarification}
              transcript={outcome?.transcript ?? null}
              latencyKey={latencyKey}
              operationId={outcome?.operation_id ?? null}
              onChoose={onChooseCandidate}
              onDismiss={() => {
                clearBoard();
                focusCommand();
              }}
            />
          )}

          {problem === null ? null : (
            <ProblemNotice
              problem={problem}
              {...(lastCommandText === null ? {} : { onRetry: repeatLast })}
              onDismiss={() => {
                setProblem(null);
              }}
            />
          )}

          <div className={styles.boardGrid}>
            <div className={styles.resultColumn}>
              <ResultCard
                outcome={outcome}
                stages={stages}
                latencyKey={latencyKey}
                busy={busy}
                workspaceId={workspaceId}
              />
              {pinned === null ? null : (
                <section aria-label={t("book.pinned")}>
                  <BookUnitView
                    unit={pinned}
                    onUnpin={() => {
                      setPinned(null);
                    }}
                  />
                </section>
              )}
            </div>

            <div
              className={cx(styles.penColumn, penOverlay && styles.penOverlay)}
              hidden={!penOpen}
            >
              <InkPanel
                streamEnabled={authenticated && penOpen}
                workspaceId={workspaceId}
                clearToken={clearToken}
                overlay={penOverlay}
                onAccept={acceptRecognition}
                onProblem={setProblem}
                onToggleOverlay={() => {
                  setPenOverlay(!penOverlay);
                }}
                onClose={() => {
                  setPenOpen(false);
                }}
              />
            </div>
          </div>
        </section>

        <aside className={styles.bookRail} aria-label={t("workspace.bookLabel")} hidden={!bookOpen}>
          <BookPanel
            hits={bookHits}
            onPin={setPinned}
            pinnedId={pinned?.id ?? null}
            onClose={() => {
              setBookOpen(false);
            }}
          />
        </aside>
      </div>

      <div className={styles.dock}>
        <CommandBar
          workspaceId={workspaceId}
          lastLatex={inputLatex === "" ? null : inputLatex}
          value={commandText}
          onValueChange={setCommandText}
          busy={busy}
          inputRef={commandRef}
          voice={voice}
          penOpen={penOpen}
          canRepeat={lastCommandText !== null}
          onRepeat={repeatLast}
          onTogglePen={() => {
            setPenOpen(!penOpen);
          }}
          onClearContext={() => {
            setInputLatex("");
          }}
          onSubmit={(text) => {
            void run(text);
          }}
        />
      </div>

      <ShortcutSheet
        open={shortcutsOpen}
        onClose={() => {
          setShortcutsOpen(false);
        }}
      />
    </div>
  );
}
