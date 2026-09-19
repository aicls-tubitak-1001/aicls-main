/**
 * Session handling: the access token lives in memory, the refresh token in an httpOnly cookie.
 *
 * api.md §1 fixes the scheme — a 15-minute HS256 access token sent as `Authorization: Bearer`, and
 * a rotating `aicls_refresh` cookie JavaScript never sees. The access token is therefore a module
 * variable inside the store and **never** `localStorage`, which any injected script could read.
 * Losing it on reload is intended: `POST /auth/refresh` mints a new one from the cookie, which is
 * exactly what {@link restoreSession} does while the shell paints.
 *
 * A dropped session must never interrupt a lecture, so the client refreshes silently on a 401
 * (see `api/client.ts`) and only lands here — signed out — when the cookie itself is gone.
 *
 * WebSocket connections use short single-use tickets from `POST /auth/ws-ticket`; never a JWT in a
 * URL (api.md §1, ADR-0014).
 */
import { api, call, setSessionBridge, type AiclsProblem } from "@/api/client";
import { setDefaultTicketSource } from "@/api/sockets";
import type { Role, UserOut } from "@/api/types";
import { t } from "@/i18n";
import { createStore, useStore } from "./store";

/** The four roles of api.md §1, in the order the admin UI lists them. */
export const ROLES: readonly Role[] = ["student", "instructor", "contributor", "admin"];

/** Where the session stands; `unknown` until the first refresh attempt has answered. */
export type SessionStatus = "unknown" | "anonymous" | "authenticated";

/** The signed-in user as far as the UI is concerned. */
export interface Session {
  /** Lifecycle of the session, so a guard can wait instead of bouncing to the login screen. */
  status: SessionStatus;
  /** The bearer token; in memory only. */
  accessToken: string | null;
  /** The profile from `GET /users/me`, once loaded. */
  user: UserOut | null;
  /** Convenience mirror of `user.role` (available before the profile is fetched). */
  role: Role | null;
  /** Display name shown in the header. */
  displayName: string | null;
  /** Epoch milliseconds at which the access token expires. */
  expiresAt: number | null;
  /** Why the last sign-in or refresh failed, if it did. */
  problem: AiclsProblem | null;
}

const EMPTY_SESSION: Session = {
  status: "unknown",
  accessToken: null,
  user: null,
  role: null,
  displayName: null,
  expiresAt: null,
  problem: null,
};

/** Observable session state. */
export const sessionStore = createStore<Session>({ ...EMPTY_SESSION });

/** Read a slice of the session inside a component. */
export function useSession<S>(selector: (session: Session) => S): S {
  return useStore(sessionStore, selector);
}

/** True when an unexpired access token is held. */
export function isAuthenticated(): boolean {
  const { accessToken, expiresAt } = sessionStore.getState();
  return accessToken !== null && (expiresAt === null || expiresAt > Date.now());
}

/** Store a freshly minted access token. */
export function setAccessToken(token: string, expiresInSeconds: number): void {
  sessionStore.setState({
    status: "authenticated",
    accessToken: token,
    expiresAt: Date.now() + expiresInSeconds * 1000,
    problem: null,
  });
}

/** Record the profile returned by `GET /users/me`. */
export function setUser(user: UserOut): void {
  sessionStore.setState({ user, role: user.role, displayName: user.display_name });
}

/**
 * Forget the session (logout, or a refresh that failed).
 *
 * @param problem - The failure to show on the sign-in screen, if any.
 */
export function clearSession(problem: AiclsProblem | null = null): void {
  sessionStore.setState({ ...EMPTY_SESSION, status: "anonymous", problem });
}

setSessionBridge({
  getAccessToken: () => sessionStore.getState().accessToken,
  onTokenRefreshed: (token, expiresIn) => {
    setAccessToken(token, expiresIn);
  },
  onSignOut: (problem) => {
    clearSession(problem);
  },
});

/* ------------------------------------------------------------------------------- roles --- */

/** Whether `role` is one of `allowed` (an empty allow-list means "any signed-in user"). */
export function hasRole(role: Role | null, ...allowed: readonly Role[]): boolean {
  if (role === null) {
    return false;
  }
  return allowed.length === 0 || allowed.includes(role);
}

/** Whether the current session may act as `allowed`. */
export function sessionHasRole(...allowed: readonly Role[]): boolean {
  return isAuthenticated() && hasRole(sessionStore.getState().role, ...allowed);
}

/** Turkish label key for a role (`i18n/tr.json`). */
export const ROLE_LABEL_KEYS = {
  student: "role.student",
  instructor: "role.instructor",
  contributor: "role.contributor",
  admin: "role.admin",
} as const satisfies Record<Role, string>;

/* ---------------------------------------------------------------------------- requests --- */

/**
 * Exchange e-mail and password for an access token.
 *
 * @returns `null` on success, or the problem to render (its `detail_tr` is always set).
 */
export async function login(email: string, password: string): Promise<AiclsProblem | null> {
  const { data, problem } = await call(api.POST("/auth/login", { body: { email, password } }), {
    fallbackDetailTr: t("errors.network"),
  });
  if (data === null) {
    clearSession(problem);
    return problem;
  }
  setAccessToken(data.access_token, data.expires_in);
  await loadCurrentUser();
  return null;
}

/**
 * Create an account and sign in with it.
 *
 * `POST /auth/register` answers 201 with the new `UserOut` and does **not** mint a token, so the
 * credentials are exchanged for one immediately afterwards; the caller gets a signed-in session
 * either way.
 *
 * @param invite - Invitation code, required by the backend for privileged roles.
 */
export async function register(
  email: string,
  password: string,
  displayName: string,
  invite?: string,
): Promise<AiclsProblem | null> {
  const { data, problem } = await call(
    api.POST("/auth/register", {
      body: {
        email,
        password,
        display_name: displayName,
        ...(invite === undefined || invite === "" ? {} : { invite_code: invite }),
      },
    }),
    { fallbackDetailTr: t("errors.network") },
  );
  if (data === null) {
    clearSession(problem);
    return problem;
  }
  return login(email, password);
}

/** End the session on the server (refresh-family revocation) and locally. */
export async function logout(): Promise<void> {
  try {
    await call(api.POST("/auth/logout", {}), { fallbackDetailTr: t("errors.network") });
  } finally {
    clearSession();
  }
}

/**
 * Mint a new access token from the refresh cookie.
 *
 * @returns `true` when the session is (still) live.
 */
export async function refreshSession(): Promise<boolean> {
  const { data, problem } = await call(api.POST("/auth/refresh", {}), {
    fallbackDetailTr: t("errors.network"),
  });
  if (data === null) {
    clearSession(problem);
    return false;
  }
  setAccessToken(data.access_token, data.expires_in);
  return true;
}

/** Load `GET /users/me` into the session; silent on failure (the shell stays usable). */
export async function loadCurrentUser(): Promise<UserOut | null> {
  const { data } = await call(api.GET("/users/me", {}), {
    fallbackDetailTr: t("errors.network"),
  });
  if (data !== null) {
    setUser(data);
  }
  return data;
}

/**
 * Restore a session on page load.
 *
 * Called once from `main.tsx`. A failure is not an error state: the visitor is simply anonymous.
 */
export async function restoreSession(): Promise<boolean> {
  const restored = await refreshSession();
  if (restored) {
    await loadCurrentUser();
  }
  return restored;
}

/* -------------------------------------------------------------------------- react binding --- */

/** Everything a screen needs from the session, in one hook. */
export interface AuthApi {
  /** The whole session record. */
  readonly session: Session;
  /** The loaded profile, or `null`. */
  readonly user: UserOut | null;
  /** The caller's role, or `null` when signed out. */
  readonly role: Role | null;
  /** `unknown` until the first refresh answered — a guard should wait, not redirect. */
  readonly status: SessionStatus;
  /** True while an unexpired access token is held. */
  readonly isAuthenticated: boolean;
  /** Whether the current role is one of `allowed` (no argument: any signed-in user). */
  readonly can: (...allowed: readonly Role[]) => boolean;
  readonly login: typeof login;
  readonly register: typeof register;
  readonly logout: typeof logout;
  readonly refresh: typeof refreshSession;
}

/**
 * Subscribe a component to the session.
 *
 * @example
 * ```tsx
 * const { isAuthenticated, can, logout } = useAuth();
 * if (can("instructor", "admin")) { … }
 * ```
 */
export function useAuth(): AuthApi {
  const session = useStore(sessionStore, (state) => state);
  const authenticated =
    session.accessToken !== null && (session.expiresAt === null || session.expiresAt > Date.now());
  return {
    session,
    user: session.user,
    role: session.role,
    status: session.status,
    isAuthenticated: authenticated,
    can: (...allowed: readonly Role[]) => authenticated && hasRole(session.role, ...allowed),
    login,
    register,
    logout,
    refresh: refreshSession,
  };
}

/**
 * Fetch a single-use WebSocket ticket.
 *
 * Passed to `TypedSocket` both at construction and, as `ticketProvider`, on every reconnect —
 * tickets are consumed by the first connection and expire after 60 s.
 *
 * @throws {Error} When the API refuses; the caller shows a disconnected badge.
 */
export async function fetchWsTicket(): Promise<string> {
  const { data, problem } = await call(api.POST("/auth/ws-ticket", {}), {
    fallbackDetailTr: t("ws.ticketFailed"),
  });
  if (data === null) {
    throw new Error(problem?.detail_tr ?? t("ws.ticketFailed"));
  }
  return data.ticket;
}

// The socket hooks fetch their own tickets; registering the source here (rather than importing
// `app/auth` from `api/sockets`) keeps the dependency pointing one way: app -> api.
setDefaultTicketSource(fetchWsTicket);
