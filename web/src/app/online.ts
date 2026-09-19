/**
 * Network awareness.
 *
 * A lecture theatre's Wi-Fi drops. When it does, the platform must not look broken: the last
 * result stays on screen, the canvas and the book panel keep working from what is already loaded,
 * and one calm Turkish line says what is unavailable. This hook is the signal the shell and the
 * panels read to do that.
 *
 * `navigator.onLine` is only a hint (it reports link state, not reachability), so a request that
 * fails is still the authority; this is for the banner, never for blocking an action.
 */
import { useEffect, useState } from "react";

/** Whether the browser currently believes it has a network connection. */
export function isOnline(): boolean {
  const online = globalThis.navigator?.onLine;
  return online === undefined ? true : online;
}

/**
 * Subscribe to the browser's online/offline events.
 *
 * @returns `true` while the browser reports a connection.
 */
export function useOnline(): boolean {
  const [online, setOnline] = useState<boolean>(isOnline);

  useEffect(() => {
    const update = (): void => {
      setOnline(isOnline());
    };
    globalThis.addEventListener("online", update);
    globalThis.addEventListener("offline", update);
    update();
    return () => {
      globalThis.removeEventListener("online", update);
      globalThis.removeEventListener("offline", update);
    };
  }, []);

  return online;
}
