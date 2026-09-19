/**
 * The participant handle of the voice-collection tab.
 *
 * `POST /contribute/consent` answers with a pseudonym (`P00012`) and a **participant token**. That
 * token is the only thing that ties the next upload — and, more importantly, a later withdrawal —
 * back to the same speaker row for a volunteer who is not signed in
 * (`aicls.api.routers.contribute.current_speaker`). It is therefore kept here, in one place, with
 * two deliberate properties:
 *
 * 1. **Session storage, not local storage.** The token is a credential; it lives for the tab that
 *    earned it and disappears when that tab closes. A shared classroom machine must not hand the
 *    next person the previous volunteer's speaker row. Everything is wrapped in `try`/`catch` so a
 *    browser with storage blocked simply keeps the token in memory for the page's lifetime.
 * 2. **Observable.** The page, the recorder and the withdrawal dialog all read the same value, so
 *    the store is the small `createStore` from `app/store` rather than a prop drilled four levels.
 *
 * A signed-in contributor does not depend on it at all: the backend resolves their speaker from
 * the account. The token is still stored for them, because it keeps a reload inside the same tab
 * working before `POST /auth/refresh` has answered.
 */
import { createStore, useStore } from "@/app/store";

/** Key of the per-tab storage slot. */
const STORAGE_KEY = "aicls.participant";

/** What the consent step produced. */
export interface Participant {
  /** Human-readable pseudonym shown to the volunteer (`speaker_pseudonym`). */
  readonly pseudonym: string;
  /** Opaque token sent as `X-Participant-Token`. */
  readonly token: string;
  /** Version of the consent text that was accepted. */
  readonly consentVersion: string;
}

/** Session state of the contribution tab. */
export interface ParticipantState {
  /** The current participant, or `null` before consent (and after withdrawal). */
  participant: Participant | null;
}

function isParticipant(value: unknown): value is Participant {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    typeof record.pseudonym === "string" &&
    typeof record.token === "string" &&
    typeof record.consentVersion === "string"
  );
}

/** Read the participant persisted for this tab, if any. */
export function readStoredParticipant(): Participant | null {
  try {
    const raw = globalThis.sessionStorage?.getItem(STORAGE_KEY) ?? null;
    if (raw === null) {
      return null;
    }
    const parsed: unknown = JSON.parse(raw);
    return isParticipant(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** Observable participant state (one per tab). */
export const participantStore = createStore<ParticipantState>({
  participant: readStoredParticipant(),
});

/** Remember a participant for this tab. */
export function setParticipant(participant: Participant): void {
  participantStore.setState({ participant });
  try {
    globalThis.sessionStorage?.setItem(STORAGE_KEY, JSON.stringify(participant));
  } catch {
    // Storage blocked: the token still lives in the store for this page view.
  }
}

/** Forget the participant (withdrawal, or an explicit "start over"). */
export function clearParticipant(): void {
  participantStore.setState({ participant: null });
  try {
    globalThis.sessionStorage?.removeItem(STORAGE_KEY);
  } catch {
    // Nothing to clean up when storage is unavailable.
  }
}

/** The token to send as `X-Participant-Token`, or `null` when there is none. */
export function participantToken(): string | null {
  return participantStore.getState().participant?.token ?? null;
}

/** Subscribe a component to the participant of this tab. */
export function useParticipant(): Participant | null {
  return useStore(participantStore, (state) => state.participant);
}
