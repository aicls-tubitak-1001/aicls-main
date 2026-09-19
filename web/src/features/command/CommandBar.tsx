/**
 * The command bar — the one control that is always reachable during a lecture.
 *
 * It carries the three input modalities side by side: a Turkish sentence typed into the field, the
 * microphone (hold to talk, or tap to latch), and the pen panel toggle. Whatever the modality, the
 * sentence ends up in the same pipeline call (`POST /compute/text` or `POST /asr/command`), so the
 * lecturer never has to think about which "mode" the board is in.
 *
 * Interaction rules that come from the room rather than from the framework:
 *
 * - the field is the largest control on the bar and keeps focus by default, so `/` and typing just
 *   work; `Ctrl + Enter` submits from anywhere inside it and `Escape` empties it;
 * - the microphone is a press-and-hold *and* a tap-to-latch button, because a hand on a smart board
 *   holds, and a hand on a tablet taps. A quick tap latches, a long press records while held;
 * - nothing here is hover-only, every control is at least the token tap-target tall, and the hint
 *   and the example chips are secondary chrome that presentation mode hides.
 *
 * The bar can run a command by itself (`onOutcome`) or delegate to the workspace (`onSubmit`); both
 * end in {@link submitTextCommand}, so the request body is identical either way.
 */
import { useRef, useState, type FormEvent, type KeyboardEvent, type RefObject } from "react";

import type { ComputeOut } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { t, type MessageKey } from "@/i18n";
import { cx } from "@/lib/cx";
import { MathView } from "@/features/results/MathView";
import { submitTextCommand } from "./submitCommand";
import type { VoiceCommandApi } from "./useVoiceCommand";
import styles from "./CommandBar.module.css";

/** How long a press may last before it counts as "hold to talk" rather than "tap to latch". */
export const HOLD_TO_TALK_MS = 400;

export interface CommandBarProps {
  /** Active workspace id, if any (used by the built-in submit path). */
  workspaceId?: string | null;
  /** `context.last_latex` — the expression currently on the board. */
  lastLatex?: string | null;
  /** Called with every outcome of the built-in submit path. */
  onOutcome?: (outcome: ComputeOut) => void;
  /** Called with a Turkish message when the built-in submit path fails. */
  onError?: (message: string) => void;
  /** Take the submission over entirely (the workspace does this so history and latency line up). */
  onSubmit?: (text: string) => void;
  /** Controlled text; omit to let the bar own it. */
  value?: string;
  /** Change handler for the controlled text. */
  onValueChange?: (value: string) => void;
  /** Disable the input while another command is running. */
  busy?: boolean;
  /** Ref to the input so the page can focus it (the `/` shortcut). */
  inputRef?: RefObject<HTMLInputElement>;
  /** Microphone controller; the microphone button appears only when this is given. */
  voice?: VoiceCommandApi;
  /** Toggle the handwriting panel; the pen button appears only when this is given. */
  onTogglePen?: () => void;
  /** Whether the handwriting panel is open (drives `aria-pressed`). */
  penOpen?: boolean;
  /** Re-run the previous command. */
  onRepeat?: () => void;
  /** Whether there is a previous command to re-run. */
  canRepeat?: boolean;
  /** Remove the board expression shown as the context chip. */
  onClearContext?: () => void;
}

/** Example commands offered as one-tap chips (proposal A.1.2 wording). */
const EXAMPLE_KEYS: MessageKey[] = [
  "command.example.derivative",
  "command.example.integral",
  "command.example.limit",
  "command.example.book",
];

/** The command input, the microphone, the pen toggle and the example chips. */
export function CommandBar({
  workspaceId = null,
  lastLatex = null,
  onOutcome,
  onError,
  onSubmit,
  value,
  onValueChange,
  busy = false,
  inputRef,
  voice,
  onTogglePen,
  penOpen = false,
  onRepeat,
  canRepeat = false,
  onClearContext,
}: CommandBarProps): JSX.Element {
  const [internalText, setInternalText] = useState("");
  const [errorText, setErrorText] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const pressStartedAt = useRef<number | null>(null);
  const suppressClick = useRef(false);
  const fallbackRef = useRef<HTMLInputElement | null>(null);
  const field = inputRef ?? fallbackRef;

  const text = value ?? internalText;
  const setText = (next: string): void => {
    if (onValueChange === undefined) {
      setInternalText(next);
    } else {
      onValueChange(next);
    }
  };

  const disabled = busy || sending;

  const submit = async (raw: string): Promise<void> => {
    const trimmed = raw.trim();
    if (trimmed === "" || disabled) {
      return;
    }
    if (onSubmit !== undefined) {
      onSubmit(trimmed);
      return;
    }
    setSending(true);
    const { outcome, problem } = await submitTextCommand(trimmed, {
      workspaceId,
      lastLatex,
    });
    setSending(false);
    if (outcome === null) {
      const message = problem?.detail_tr ?? t("errors.unknown");
      setErrorText(message);
      onError?.(message);
      return;
    }
    setErrorText(null);
    onOutcome?.(outcome);
  };

  const onFormSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    void submit(text);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      void submit(text);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setText("");
      setErrorText(null);
    }
  };

  /* -- microphone: hold to talk, tap to latch -------------------------------------------- */

  const onMicPointerDown = (): void => {
    if (voice === undefined || disabled) {
      return;
    }
    pressStartedAt.current = Date.now();
    if (!voice.active) {
      voice.start();
    }
  };

  const onMicPointerUp = (): void => {
    if (voice === undefined) {
      return;
    }
    const startedAt = pressStartedAt.current;
    pressStartedAt.current = null;
    if (startedAt === null) {
      return;
    }
    suppressClick.current = true;
    // A long press is "hold to talk": lifting the finger ends the utterance. A quick tap latches,
    // and the same button then reads "Kaydı bitir".
    if (Date.now() - startedAt >= HOLD_TO_TALK_MS && voice.active) {
      voice.stop();
    }
  };

  const onMicClick = (): void => {
    if (voice === undefined) {
      return;
    }
    if (suppressClick.current) {
      suppressClick.current = false;
      return;
    }
    voice.toggle();
  };

  const recording = voice?.phase === "recording";
  const micLabel = recording ? t("command.voiceStop") : t("command.voiceToggle");

  return (
    <form className={styles.bar} onSubmit={onFormSubmit} aria-label={t("command.formLabel")}>
      {lastLatex === null || lastLatex === "" ? null : (
        <div className={styles.context}>
          <span className={styles.contextLabel}>{t("workspace.lastInput")}</span>
          <span className="visually-hidden">{t("workspace.lastInputHint")}</span>
          <MathView latex={lastLatex} />
          {onClearContext === undefined ? null : (
            <button
              type="button"
              className={styles.contextClear}
              onClick={onClearContext}
              aria-label={t("workspace.clearInput")}
            >
              <span aria-hidden="true">×</span>
            </button>
          )}
        </div>
      )}

      <div className={styles.row}>
        {voice === undefined ? null : (
          <button
            type="button"
            className={cx(styles.mic, recording && styles.micActive)}
            onPointerDown={onMicPointerDown}
            onPointerUp={onMicPointerUp}
            onPointerCancel={onMicPointerUp}
            onClick={onMicClick}
            disabled={disabled || !voice.supported}
            aria-pressed={recording}
            aria-label={micLabel}
            title={t("command.voiceHold")}
            data-testid="mic-button"
          >
            <span aria-hidden="true" className={styles.micGlyph}>
              ●
            </span>
          </button>
        )}

        <label className="visually-hidden" htmlFor="command-input">
          {t("command.label")}
        </label>
        <input
          id="command-input"
          ref={field}
          className={styles.input}
          type="text"
          value={text}
          placeholder={t("command.placeholder")}
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
          enterKeyHint="send"
          aria-describedby="command-hint"
          disabled={disabled}
          onChange={(event) => {
            setText(event.target.value);
          }}
          onKeyDown={onKeyDown}
        />

        <Button
          type="submit"
          variant="primary"
          size="lg"
          loading={disabled}
          disabled={text.trim() === ""}
        >
          {disabled ? t("command.busy") : t("command.submit")}
        </Button>
      </div>

      <div className={styles.tools}>
        {onTogglePen === undefined ? null : (
          <Button variant="ghost" onClick={onTogglePen} aria-pressed={penOpen}>
            {t("command.pen")}
          </Button>
        )}
        {onRepeat === undefined ? null : (
          <Button variant="ghost" onClick={onRepeat} disabled={!canRepeat}>
            {t("command.repeat")}
          </Button>
        )}
        <Button
          variant="ghost"
          disabled={text === ""}
          onClick={() => {
            setText("");
            setErrorText(null);
          }}
        >
          {t("command.clear")}
        </Button>
        {voice !== undefined && voice.active ? (
          <Button variant="ghost" onClick={voice.cancel}>
            {t("command.voiceCancel")}
          </Button>
        ) : null}
        {voice === undefined ? null : (
          <span className={styles.voiceStatus} role="status">
            {voice.phase === "recording"
              ? t("command.recording", { seconds: voice.seconds })
              : voice.phase === "encoding" || voice.phase === "uploading"
                ? t("command.voiceUploading")
                : ""}
          </span>
        )}
      </div>

      {/* The input points at this paragraph with `aria-describedby`, so presentation mode clips it
          rather than removing it: a `display: none` description is announced to nobody. */}
      <p id="command-hint" className={cx(styles.hint, "chrome-secondary-quiet")}>
        {t("command.hint")} {t("command.submitHint")}
        {voice === undefined ? "" : ` ${t("command.voiceHint")}`}
      </p>

      <div
        className={cx(styles.examples, "chrome-secondary")}
        role="group"
        aria-label={t("command.examplesTitle")}
      >
        {EXAMPLE_KEYS.map((key) => (
          <button
            key={key}
            type="button"
            className={styles.example}
            disabled={disabled}
            onClick={() => {
              setText(t(key));
              field.current?.focus();
            }}
          >
            {t(key)}
          </button>
        ))}
      </div>

      {errorText === null ? null : (
        <p className={styles.error} role="alert">
          {errorText}
        </p>
      )}
    </form>
  );
}
