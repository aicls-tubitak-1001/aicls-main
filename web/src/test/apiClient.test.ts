/**
 * The API client: silent re-authentication, problem parsing and latency capture.
 *
 * The 401 path is the one a lecture depends on. A 15-minute access token *will* expire in the
 * middle of a class, and when it does the client must refresh from the cookie and replay the
 * request without the lecturer noticing — and, if the cookie is gone too, sign out once and
 * report a Turkish sentence rather than a stack trace.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  api,
  call,
  lastRequestId,
  resetSessionBridge,
  setSessionBridge,
  type AiclsProblem,
} from "@/api/client";

const USER = {
  id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  email: "hoca@alparslan.edu.tr",
  display_name: "Öğretim Elemanı",
  role: "instructor" as const,
  created_at: "2026-09-01T08:00:00Z",
};

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

function problemResponse(
  code: string,
  status: number,
  detailTr: string,
  headers: Record<string, string> = {},
): Response {
  return new Response(
    JSON.stringify({
      type: `https://ai-cls.example/errors/${code}`,
      title: "Hata",
      status,
      code,
      detail_tr: detailTr,
    }),
    { status, headers: { "content-type": "application/problem+json", ...headers } },
  );
}

/** Requests recorded by the stubbed `fetch`, in order. */
let recorded: Request[] = [];

function installFetch(handler: (request: Request, index: number) => Response): void {
  recorded = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((input: Request | string) => {
      const request = typeof input === "string" ? new Request(input) : input;
      const index = recorded.length;
      recorded.push(request);
      return Promise.resolve(handler(request, index));
    }),
  );
}

beforeEach(() => {
  resetSessionBridge();
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetSessionBridge();
});

describe("401 -> refresh -> retry", () => {
  it("refreshes once and replays the original request with the new token", async () => {
    let token = "expired-token";
    const refreshed = vi.fn((next: string) => {
      token = next;
    });
    const signedOut = vi.fn();
    setSessionBridge({
      getAccessToken: () => token,
      onTokenRefreshed: (next) => {
        refreshed(next);
      },
      onSignOut: signedOut,
    });

    installFetch((request) => {
      const path = new URL(request.url).pathname;
      if (path.endsWith("/auth/refresh")) {
        return jsonResponse({ access_token: "fresh-token", expires_in: 900 });
      }
      if (request.headers.get("Authorization") === "Bearer fresh-token") {
        return jsonResponse(USER);
      }
      return problemResponse("AUTH_INVALID", 401, "Oturumunuzun süresi doldu.");
    });

    const { data, problem } = await call(api.GET("/users/me", {}), {
      fallbackDetailTr: "ağ hatası",
    });

    expect(problem).toBeNull();
    expect(data?.email).toBe("hoca@alparslan.edu.tr");
    expect(refreshed).toHaveBeenCalledWith("fresh-token");
    expect(signedOut).not.toHaveBeenCalled();
    // original 401, refresh, replay
    expect(recorded).toHaveLength(3);
    expect(new URL(recorded[1]?.url ?? "").pathname).toContain("/auth/refresh");
    expect(recorded[2]?.headers.get("Authorization")).toBe("Bearer fresh-token");
    expect(recorded[2]?.headers.get("X-Aicls-Retry")).toBe("1");
  });

  it("replays a POST with its body intact", async () => {
    let token = "expired-token";
    setSessionBridge({
      getAccessToken: () => token,
      onTokenRefreshed: (next) => {
        token = next;
      },
    });
    installFetch((request) => {
      const path = new URL(request.url).pathname;
      if (path.endsWith("/auth/refresh")) {
        return jsonResponse({ access_token: "fresh-token", expires_in: 900 });
      }
      if (request.headers.get("Authorization") === "Bearer fresh-token") {
        return jsonResponse({ kind: "result" });
      }
      return problemResponse("AUTH_INVALID", 401, "Oturumunuzun süresi doldu.");
    });

    const { data } = await call(
      api.POST("/compute/text", { body: { text: "x karenin türevini al" } }),
      { fallbackDetailTr: "ağ hatası" },
    );

    expect(data?.kind).toBe("result");
    const replayed = recorded[2];
    expect(replayed).toBeDefined();
    expect(await replayed?.clone().json()).toEqual({ text: "x karenin türevini al" });
  });

  it("signs out when the refresh cookie is gone, and does not loop", async () => {
    const signedOut = vi.fn();
    setSessionBridge({
      getAccessToken: () => "expired-token",
      onSignOut: signedOut,
    });
    installFetch((request) => {
      const path = new URL(request.url).pathname;
      if (path.endsWith("/auth/refresh")) {
        return problemResponse("AUTH_INVALID", 401, "Oturum bulunamadı.");
      }
      return problemResponse("AUTH_INVALID", 401, "Oturumunuzun süresi doldu.");
    });

    const { data, problem } = await call(api.GET("/users/me", {}), {
      fallbackDetailTr: "ağ hatası",
    });

    expect(data).toBeNull();
    expect(problem?.code).toBe("AUTH_INVALID");
    expect(problem?.detail_tr).toBe("Oturumunuzun süresi doldu.");
    expect(signedOut).toHaveBeenCalledTimes(1);
    const problemArgument = signedOut.mock.calls[0]?.[0] as AiclsProblem | null;
    expect(problemArgument?.detail_tr).toBe("Oturum bulunamadı.");
    expect(recorded).toHaveLength(2); // request + one refresh attempt, no replay
  });

  it("shares one refresh between concurrent 401s", async () => {
    let token = "expired-token";
    setSessionBridge({
      getAccessToken: () => token,
      onTokenRefreshed: (next) => {
        token = next;
      },
    });
    installFetch((request) => {
      const path = new URL(request.url).pathname;
      if (path.endsWith("/auth/refresh")) {
        return jsonResponse({ access_token: "fresh-token", expires_in: 900 });
      }
      if (request.headers.get("Authorization") === "Bearer fresh-token") {
        return jsonResponse(USER);
      }
      return problemResponse("AUTH_INVALID", 401, "Oturumunuzun süresi doldu.");
    });

    const [first, second] = await Promise.all([
      call(api.GET("/users/me", {}), { fallbackDetailTr: "ağ hatası" }),
      call(api.GET("/users/me", {}), { fallbackDetailTr: "ağ hatası" }),
    ]);

    expect(first.problem).toBeNull();
    expect(second.problem).toBeNull();
    const refreshCalls = recorded.filter((request) =>
      new URL(request.url).pathname.endsWith("/auth/refresh"),
    );
    expect(refreshCalls).toHaveLength(1);
  });

  it("never refreshes (or signs out) for the login endpoint itself", async () => {
    const signedOut = vi.fn();
    setSessionBridge({ getAccessToken: () => null, onSignOut: signedOut });
    installFetch(() => problemResponse("AUTH_INVALID", 401, "E-posta veya parola hatalı."));

    const { problem } = await call(
      api.POST("/auth/login", { body: { email: "a@b.co", password: "x" } }),
      { fallbackDetailTr: "ağ hatası" },
    );

    // A refused login is the caller's business: it must not drop a session that may still be live
    // in another tab, and it must certainly not trigger a refresh loop.
    expect(problem?.detail_tr).toBe("E-posta veya parola hatalı.");
    expect(recorded).toHaveLength(1);
    expect(signedOut).not.toHaveBeenCalled();
  });
});

describe("call()", () => {
  it("parses problem+json into a typed problem with the request id", async () => {
    installFetch(() =>
      problemResponse(
        "UNSUPPORTED_PROBLEM",
        422,
        "Bu problem tipi şu an için desteklenmemektedir.",
        {
          "X-Request-Id": "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
        },
      ),
    );

    const { data, problem, requestId } = await call(
      api.POST("/compute/text", { body: { text: "üçlü integral" } }),
      { fallbackDetailTr: "ağ hatası" },
    );

    expect(data).toBeNull();
    expect(problem?.code).toBe("UNSUPPORTED_PROBLEM");
    expect(problem?.status).toBe(422);
    expect(problem?.detail_tr).toBe("Bu problem tipi şu an için desteklenmemektedir.");
    expect(problem?.instance).toBe("01J8R2ZQ7G4V0X9K3M2N5P6Q7R");
    expect(requestId).toBe("01J8R2ZQ7G4V0X9K3M2N5P6Q7R");
    expect(lastRequestId()).toBe("01J8R2ZQ7G4V0X9K3M2N5P6Q7R");
  });

  it("reads the Server-Timing stages of a successful call", async () => {
    installFetch(() =>
      jsonResponse({ kind: "result" }, 200, {
        "Server-Timing": "nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, app;dur=58.2",
      }),
    );

    const { stages } = await call(api.POST("/compute/text", { body: { text: "türev al" } }), {
      fallbackDetailTr: "ağ hatası",
    });

    expect(stages.map((stage) => stage.name)).toEqual(["nlu", "parse", "cas", "app"]);
    expect(stages[3]?.ms).toBeCloseTo(58.2, 5);
  });

  it("turns a transport failure into a Turkish problem instead of throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new Error("network down"))),
    );

    const { data, problem } = await call(api.GET("/users/me", {}), {
      fallbackDetailTr: "Sunucuya ulaşılamadı.",
    });

    expect(data).toBeNull();
    expect(problem?.status).toBe(0);
    expect(problem?.code).toBe("CLIENT_TRANSPORT");
    expect(problem?.detail_tr).toBe("Sunucuya ulaşılamadı.");
  });

  it("treats a 204 as success (logout answers no content)", async () => {
    installFetch(() => new Response(null, { status: 204 }));

    const { data, problem } = await call(api.POST("/auth/logout", {}), {
      fallbackDetailTr: "ağ hatası",
    });

    expect(problem).toBeNull();
    expect(data).toBeNull();
  });
});
