/**
 * "Ask me before you compute what you heard" — remembered per browser.
 *
 * api.md §3.6 makes the confirmation step mandatory on the voice path, and ADR-0014 explains why:
 * a wrong command in front of a class costs more than a slow one. Some lecturers will nevertheless
 * want the fast path once they trust the microphone, so the step is skippable — but only for
 * confident recognitions. Below {@link CONFIRMATION_THRESHOLD} the interface always asks, whatever
 * the preference says, so the contract's floor is never traded away for convenience.
 *
 * The choice is a per-browser convenience in `localStorage`, never personal data; a browser that
 * blocks storage simply asks every time.
 */
import { useCallback, useState } from "react";

/** Confidence below which the confirmation step is compulsory (api.md §3.6). */
export const CONFIRMATION_THRESHOLD = 0.8;

const STORAGE_KEY = "aicls.voiceConfirm";

/** Whether the lecturer wants to confirm every spoken command. */
export function readVoiceConfirm(): boolean {
  try {
    return globalThis.localStorage?.getItem(STORAGE_KEY) !== "off";
  } catch {
    return true;
  }
}

/** Remember the choice (silently ignored when storage is unavailable). */
export function writeVoiceConfirm(always: boolean): void {
  try {
    globalThis.localStorage?.setItem(STORAGE_KEY, always ? "on" : "off");
  } catch {
    // Private mode or blocked storage: the choice does not survive a reload.
  }
}

/**
 * Decide whether a spoken command has to be confirmed before it counts.
 *
 * @param confidence - ASR confidence of the transcript (`ComputeOut.confidence`), if reported.
 * @param always - The remembered preference.
 * @returns `true` when the confirmation panel must be shown.
 */
export function needsConfirmation(confidence: number | null | undefined, always: boolean): boolean {
  if (typeof confidence !== "number" || Number.isNaN(confidence)) {
    return true;
  }
  return always || confidence < CONFIRMATION_THRESHOLD;
}

/** React binding for the preference switch shown inside the confirmation panel. */
export function useVoiceConfirmPreference(): {
  always: boolean;
  setAlways: (value: boolean) => void;
} {
  const [always, setAlwaysState] = useState<boolean>(readVoiceConfirm);
  const setAlways = useCallback((value: boolean): void => {
    writeVoiceConfirm(value);
    setAlwaysState(value);
  }, []);
  return { always, setAlways };
}
