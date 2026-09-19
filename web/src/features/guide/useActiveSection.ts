/**
 * Which section the reader is currently looking at.
 *
 * The table of contents marks one entry as the current location while the page scrolls. It is
 * done with an `IntersectionObserver` and never with a scroll listener: a listener on a long
 * document runs on every frame, and this page may be open on the same machine that is driving a
 * lecture.
 *
 * The observation band is the top of the viewport (`rootMargin` pushes the bottom edge up), so the
 * "current" section is the one whose heading has just passed under the header — which is what a
 * reader perceives as *here* — rather than whichever section happens to cover the most pixels.
 *
 * Environments without an `IntersectionObserver` (jsdom, very old browsers) simply get the first
 * section: the table of contents still works as a list of links, it just does not follow along.
 */
import { useEffect, useMemo, useRef, useState } from "react";

/** Top of the band, below the sticky header; bottom of the band, 55 % up from the viewport foot. */
const ROOT_MARGIN = "-96px 0px -55% 0px";

/**
 * Track the section currently under the top of the viewport.
 *
 * @param ids - Section element ids, in document order.
 * @returns The active id, or `null` before anything has been observed.
 */
export function useActiveSection(ids: readonly string[]): string | null {
  const key = ids.join("|");
  const stableIds = useMemo(() => key.split("|").filter(Boolean), [key]);
  const [active, setActive] = useState<string | null>(stableIds[0] ?? null);
  const activeRef = useRef<string | null>(active);

  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  useEffect(() => {
    setActive(stableIds[0] ?? null);
    if (typeof IntersectionObserver === "undefined" || stableIds.length === 0) {
      return undefined;
    }

    const visible = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = entry.target.id;
          if (entry.isIntersecting) {
            visible.set(id, entry.boundingClientRect.top);
          } else {
            visible.delete(id);
          }
        }
        if (visible.size === 0) {
          // Nothing in the band (a long section, or the very bottom of the page): keep the last
          // answer rather than blanking the marker, which would read as a glitch.
          return;
        }
        let best: string | null = null;
        let bestTop = Number.POSITIVE_INFINITY;
        for (const id of stableIds) {
          const top = visible.get(id);
          if (top !== undefined && top < bestTop) {
            best = id;
            bestTop = top;
          }
        }
        if (best !== null && best !== activeRef.current) {
          setActive(best);
        }
      },
      { rootMargin: ROOT_MARGIN, threshold: 0 },
    );

    for (const id of stableIds) {
      const element = document.getElementById(id);
      if (element !== null) {
        observer.observe(element);
      }
    }
    return () => {
      observer.disconnect();
    };
  }, [stableIds]);

  return active;
}
