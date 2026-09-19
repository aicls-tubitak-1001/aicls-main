/**
 * The session store, the role helper and the route guards.
 *
 * Two properties matter most here and are asserted directly: the access token never reaches
 * `localStorage` (an injected script must not be able to read it), and a guard never navigates on
 * its own — it waits while the session is being restored and explains itself when it refuses.
 */
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearSession,
  fetchWsTicket,
  hasRole,
  isAuthenticated,
  login,
  logout,
  refreshSession,
  ROLE_LABEL_KEYS,
  ROLES,
  sessionHasRole,
  sessionStore,
  setAccessToken,
  useAuth,
} from "@/app/auth";
import { RequireAuth, RequireRole } from "@/app/guards";
import { installApiMock, jsonResponse, problemResponse } from "./apiMock";

const USER = {
  id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  email: "hoca@alparslan.edu.tr",
  display_name: "Öğretim Elemanı",
  role: "instructor" as const,
  created_at: "2026-09-01T08:00:00Z",
};

beforeEach(() => {
  clearSession();
  globalThis.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  clearSession();
});

describe("session store", () => {
  it("keeps the access token in memory only", async () => {
    installApiMock({
      "POST /auth/login": { access_token: "tok", expires_in: 900 },
      "GET /users/me": USER,
    });

    await login("hoca@alparslan.edu.tr", "parola-parola");

    expect(sessionStore.getState().accessToken).toBe("tok");
    expect(isAuthenticated()).toBe(true);
    const storage = JSON.stringify({ ...globalThis.localStorage });
    expect(storage).not.toContain("tok");
  });

  it("loads the profile and mirrors the role", async () => {
    installApiMock({
      "POST /auth/login": { access_token: "tok", expires_in: 900 },
      "GET /users/me": USER,
    });

    await login("hoca@alparslan.edu.tr", "parola-parola");

    await waitFor(() => {
      expect(sessionStore.getState().role).toBe("instructor");
    });
    expect(sessionStore.getState().displayName).toBe("Öğretim Elemanı");
  });

  it("reports a refused sign-in with the server's Turkish sentence", async () => {
    installApiMock({
      "POST /auth/login": () => problemResponse("AUTH_INVALID", 401, "E-posta veya parola hatalı."),
    });

    const problem = await login("hoca@alparslan.edu.tr", "yanlış");

    expect(problem?.detail_tr).toBe("E-posta veya parola hatalı.");
    expect(sessionStore.getState().status).toBe("anonymous");
    expect(isAuthenticated()).toBe(false);
  });

  it("treats an expired token as signed out", () => {
    setAccessToken("tok", -1);

    expect(isAuthenticated()).toBe(false);
  });

  it("clears the session on logout even when the server call fails", async () => {
    setAccessToken("tok", 900);
    installApiMock({
      "POST /auth/logout": () => problemResponse("INTERNAL", 500, "Sunucu hatası."),
    });

    await logout();

    expect(sessionStore.getState().accessToken).toBeNull();
    expect(sessionStore.getState().status).toBe("anonymous");
  });

  it("mints a token from the refresh cookie", async () => {
    installApiMock({ "POST /auth/refresh": { access_token: "fresh", expires_in: 900 } });

    await expect(refreshSession()).resolves.toBe(true);
    expect(sessionStore.getState().accessToken).toBe("fresh");
  });

  it("stays anonymous when there is no cookie to refresh from", async () => {
    installApiMock({
      "POST /auth/refresh": () => problemResponse("AUTH_INVALID", 401, "Oturum bulunamadı."),
    });

    await expect(refreshSession()).resolves.toBe(false);
    expect(sessionStore.getState().status).toBe("anonymous");
    expect(sessionStore.getState().problem?.detail_tr).toBe("Oturum bulunamadı.");
  });

  it("fetches a WebSocket ticket, and throws a Turkish message when refused", async () => {
    installApiMock({ "POST /auth/ws-ticket": { ticket: "tk-1", expires_in: 60 } });
    await expect(fetchWsTicket()).resolves.toBe("tk-1");

    installApiMock({
      "POST /auth/ws-ticket": () => problemResponse("AUTH_INVALID", 401, "Bilet verilemedi."),
    });
    await expect(fetchWsTicket()).rejects.toThrow("Bilet verilemedi.");
  });
});

describe("roles", () => {
  it("knows the four roles of api.md §1", () => {
    expect(ROLES).toEqual(["student", "instructor", "contributor", "admin"]);
    expect(Object.keys(ROLE_LABEL_KEYS)).toHaveLength(4);
  });

  it("matches a role against an allow-list, and an empty list means any signed-in user", () => {
    expect(hasRole("instructor", "instructor", "admin")).toBe(true);
    expect(hasRole("student", "instructor", "admin")).toBe(false);
    expect(hasRole("student")).toBe(true);
    expect(hasRole(null)).toBe(false);
  });

  it("answers for the live session", () => {
    expect(sessionHasRole("admin")).toBe(false);

    act(() => {
      setAccessToken("tok", 900);
      sessionStore.setState({ role: "admin" });
    });

    expect(sessionHasRole("admin")).toBe(true);
    expect(sessionHasRole("student")).toBe(false);
  });
});

describe("useAuth", () => {
  it("exposes the session and a role predicate", () => {
    function Probe(): JSX.Element {
      const { isAuthenticated: signedIn, can, role } = useAuth();
      return (
        <p>
          {signedIn ? "içeride" : "dışarıda"} · {role ?? "-"} ·{" "}
          {can("instructor", "admin") ? "yetkili" : "yetkisiz"}
        </p>
      );
    }

    render(<Probe />);
    expect(screen.getByText(/dışarıda/)).toBeInTheDocument();

    act(() => {
      setAccessToken("tok", 900);
      sessionStore.setState({ role: "instructor" });
    });

    expect(screen.getByText(/içeride/)).toBeInTheDocument();
    expect(screen.getByText(/yetkili/)).toBeInTheDocument();
  });
});

describe("route guards", () => {
  it("waits instead of deciding while the session is being restored", () => {
    act(() => {
      sessionStore.setState({ status: "unknown", accessToken: null, role: null });
    });

    render(
      <RequireAuth>
        <p>gizli</p>
      </RequireAuth>,
    );

    expect(screen.queryByText("gizli")).toBeNull();
    expect(screen.getByRole("status")).toHaveTextContent("Oturum geri yükleniyor…");
  });

  it("explains itself instead of navigating when the visitor is anonymous", () => {
    act(() => {
      clearSession();
    });

    render(
      <RequireAuth>
        <p>gizli</p>
      </RequireAuth>,
    );

    expect(screen.queryByText("gizli")).toBeNull();
    expect(screen.getByText("Bu bölüm için giriş gerekiyor")).toBeInTheDocument();
  });

  it("renders the children for an allowed role and refuses another", () => {
    act(() => {
      setAccessToken("tok", 900);
      sessionStore.setState({ role: "student" });
    });

    const { rerender } = render(
      <RequireRole roles={["instructor", "admin"]}>
        <p>yönetim</p>
      </RequireRole>,
    );
    expect(screen.queryByText("yönetim")).toBeNull();
    expect(screen.getByText("Bu bölüme erişim yetkiniz yok")).toBeInTheDocument();

    act(() => {
      sessionStore.setState({ role: "admin" });
    });
    rerender(
      <RequireRole roles={["instructor", "admin"]}>
        <p>yönetim</p>
      </RequireRole>,
    );
    expect(screen.getByText("yönetim")).toBeInTheDocument();
  });
});

describe("apiMock sanity", () => {
  it("answers an unknown route with a problem document", async () => {
    const fetchMock = installApiMock({});
    const response = await globalThis.fetch(new Request("http://localhost/api/v1/unknown"));

    expect(response.status).toBe(404);
    expect(fetchMock).toHaveBeenCalled();
    expect(jsonResponse({ ok: true }).status).toBe(200);
  });
});
