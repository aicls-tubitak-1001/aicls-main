/**
 * Paths of the identity screens.
 *
 * They live beside the screens rather than in `routes.tsx` so a page can link to its sibling
 * without importing the route table that imports it back. The route table re-exports them, so
 * `ROUTES` stays the single place a link is looked up from outside this feature.
 */

/** Turkish URLs of the sign-in, sign-up and account screens. */
export const AUTH_ROUTES = {
  login: "/giris",
  register: "/kayit",
  account: "/hesap",
} as const;

/** Where a screen sends someone after a successful sign-in (the root redirects to the workspace). */
export const AFTER_LOGIN = "/";
