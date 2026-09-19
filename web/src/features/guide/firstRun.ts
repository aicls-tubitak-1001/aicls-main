/**
 * Whether the workspace still shows the "İlk beş dakika" card.
 *
 * The card is the guide's front door: without it the documentation would be one more item in a
 * navigation bar that a lecturer never has time to open. It is therefore shown until it is
 * explicitly dismissed, and the dismissal is remembered per browser.
 *
 * `localStorage` is the right home for this and only this reason: it is a per-viewer convenience,
 * it is not personal data, and losing it (private window, cleared site data, a different machine)
 * costs the user exactly one dismissal. Every access is wrapped, because a browser configured to
 * block site data throws on the accessor itself rather than returning null.
 */

/** Key under which the dismissal is remembered. */
export const FIRST_RUN_STORAGE_KEY = "aicls.guide.firstRun";

/** Value written when the lecturer hides the card. */
const DISMISSED = "dismissed";

/** Whether the card should be rendered. */
export function isFirstRunVisible(): boolean {
  try {
    return globalThis.localStorage?.getItem(FIRST_RUN_STORAGE_KEY) !== DISMISSED;
  } catch {
    return true;
  }
}

/**
 * Remember (or forget) the dismissal.
 *
 * @param visible - `false` hides the card from now on; `true` brings it back, which is what the
 *   undo action in the toast calls.
 */
export function setFirstRunVisible(visible: boolean): void {
  try {
    if (visible) {
      globalThis.localStorage?.removeItem(FIRST_RUN_STORAGE_KEY);
    } else {
      globalThis.localStorage?.setItem(FIRST_RUN_STORAGE_KEY, DISMISSED);
    }
  } catch {
    // Storage is blocked: the choice simply does not survive a reload.
  }
}
