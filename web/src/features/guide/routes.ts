/**
 * The guide's own paths.
 *
 * They live in the feature rather than in `src/routes.tsx` so the guide's components can link to
 * each other without importing the route table that mounts them (which would be a cycle). The
 * application's `ROUTES` re-exports the root, so the header navigation and the Playwright specs
 * keep a single spelling.
 *
 * Paths are Turkish because the whole interface is; a deep link looks like
 * `/kilavuz/teknik#boru-hatti` and is meant to be pasted into a lesson plan or a review comment.
 */

/** Canonical guide paths. */
export const GUIDE_ROUTES = {
  root: "/kilavuz",
  usage: "/kilavuz/kullanim",
  tech: "/kilavuz/teknik",
  status: "/kilavuz/durum",
} as const;

/** Route segments below `/kilavuz`, used by the nested `<Route>` elements. */
export const GUIDE_SEGMENTS = {
  usage: "kullanim",
  tech: "teknik",
  status: "durum",
} as const;
