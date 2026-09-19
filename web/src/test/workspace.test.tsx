/**
 * The classroom workspace, driven the way a lecturer drives it.
 *
 * Every test here is a promise the interface makes in front of a class: a typed command produces an
 * answer and reports how long it took to *display* it; an ambiguous command asks a question instead
 * of guessing; a problem the CAS cannot solve is a calm sentence, not a crash; the history re-runs a
 * command with one tap; and the whole bar is reachable from the keyboard alone.
 *
 * The `fetch` router in `apiMock.ts` stands in for the backend, so the assertions are about the two
 * things the front end owns: the request it builds and the Turkish surface it renders.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";

import type { ComputeOut, OperationOut, WorkspaceOut } from "@/api/types";
import { setAccessToken, sessionStore } from "@/app/auth";
import { resetAppState } from "@/app/store";
import { resetLatencyMarks } from "@/lib/latency";
import { WorkspacePage } from "@/features/workspace/WorkspacePage";
import { installApiMock, problemResponse, type RouteHandler } from "./apiMock";

const WORKSPACE: WorkspaceOut = {
  id: "ws-1",
  title: "Ders oturumu",
  created_at: "2027-03-01T09:00:00Z",
  updated_at: "2027-03-01T09:00:00Z",
  n_operations: 1,
};

const RESULT: ComputeOut = {
  kind: "result",
  operation_id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  input_latex: "x^{2}",
  command: { op: "diff", expr: "x^{2}", var: "x", order: 1 },
  result: {
    kind: "expr",
    latex: "2 x",
    plain: "2*x",
    srepr: "Mul(Integer(2), Symbol('x'))",
    steps: [],
    plot_url: null,
    cache_hit: false,
  },
  clarification: null,
  book_units: [],
  nlu: { intent: "differentiate", score: 0.93, trace: [], source: "rules" },
  timings: { nlu: 12.3, parse: 1.1, cas: 41, render: 0.4, total: 58.2 },
  model_versions: { ocr: "stub-0.1.0", asr: null },
};

const CLARIFICATION: ComputeOut = {
  kind: "clarification",
  operation_id: null,
  input_latex: "x^{2}",
  command: null,
  result: null,
  clarification: {
    question_tr: "Hangi değişkene göre türev alınsın?",
    missing_slot: "var",
    candidates: [
      {
        intent: "differentiate",
        command: { op: "diff", expr: "x^{2}", var: "x", order: 1 },
        latex_preview: "\\frac{d}{dx} x^{2}",
        score: 0.61,
        label_tr: "x'e göre türev",
      },
      {
        intent: "differentiate_y",
        command: { op: "diff", expr: "x^{2}", var: "y", order: 1 },
        latex_preview: "\\frac{d}{dy} x^{2}",
        score: 0.24,
        label_tr: "y'ye göre türev",
      },
    ],
  },
  book_units: [],
  nlu: { intent: "differentiate", score: 0.61, trace: [], source: "rules" },
  timings: { nlu: 14.2, total: 15 },
  model_versions: {},
};

const OPERATION: OperationOut = {
  id: "01J8R2ZQ7G4V0X9K3M2N5P6Q80",
  input_kind: "text",
  input_summary: "x karenin türevini al",
  input_latex: "x^{2}",
  command: { op: "diff", expr: "x^{2}" },
  result_latex: "2 x",
  result_kind: "expr",
  plot_url: null,
  error_code: null,
  timings: { total: 40 },
  created_at: "2027-03-01T09:15:00Z",
};

/** Server-Timing exactly as `RequestContextMiddleware` writes it. */
const SERVER_TIMING = "nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4, app;dur=58.2";

function computeResponse(body: ComputeOut): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      "content-type": "application/json",
      "Server-Timing": SERVER_TIMING,
      "X-Request-Id": "01J8REQUEST",
    },
  });
}

/** Mount the workspace the way the shell does: inside a router. */
function renderWorkspace(): void {
  render(
    <MemoryRouter>
      <WorkspacePage />
    </MemoryRouter>,
  );
}

/** Base routes every workspace test needs. */
function baseRoutes(): Record<string, RouteHandler> {
  return {
    "GET /workspaces": { items: [WORKSPACE], next_cursor: null },
    "GET /workspaces/ws-1/operations": { items: [], next_cursor: null },
    "POST /telemetry/latency": () => new Response(null, { status: 204 }),
  };
}

/** Find the JSON body of the first request matching `method` and `path`. */
async function bodyOf(mock: Mock, method: string, path: string): Promise<unknown> {
  for (const call of mock.mock.calls) {
    const request = call[0] as Request;
    if (request.method === method && new URL(request.url).pathname === `/api/v1${path}`) {
      return await request.clone().json();
    }
  }
  throw new Error(`istek bulunamadı: ${method} ${path}`);
}

/** How many requests matched `method` and `path`. */
function countOf(mock: Mock, method: string, path: string): number {
  return mock.mock.calls.filter((call) => {
    const request = call[0] as Request;
    return request.method === method && new URL(request.url).pathname === `/api/v1${path}`;
  }).length;
}

beforeEach(() => {
  setAccessToken("access-token", 900);
});

afterEach(() => {
  vi.unstubAllGlobals();
  resetAppState();
  resetLatencyMarks();
  sessionStore.setState({ accessToken: null, status: "anonymous", user: null, role: null });
  globalThis.localStorage.clear();
  document.documentElement.removeAttribute("data-density");
});

describe("workspace — typed command", () => {
  it("sends the sentence with the workspace and renders the answer", async () => {
    const mock = installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => computeResponse(RESULT),
    });
    renderWorkspace();

    const input = await screen.findByLabelText("Komut");
    await waitFor(() => {
      expect(countOf(mock, "GET", "/workspaces")).toBe(1);
    });

    fireEvent.change(input, { target: { value: "x karenin türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();
    expect(screen.getByTestId("result-plain")).toHaveTextContent("2*x");
    expect(await bodyOf(mock, "POST", "/compute/text")).toEqual({
      text: "x karenin türevini al",
      workspace_id: "ws-1",
    });
  });

  it("exports the operation the lecturer is looking at", async () => {
    const mock = installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => computeResponse(RESULT),
      [`POST /workspaces/ws-1/operations/${RESULT.operation_id as string}/export`]: () =>
        new Response("\\documentclass{article}", {
          status: 200,
          headers: { "content-type": "text/x-tex; charset=utf-8" },
        }),
    });
    // jsdom has no object URLs and no real downloads; stub both so the component's success path
    // is exercised instead of its "this browser cannot save" branch.
    Object.assign(globalThis.URL, {
      createObjectURL: () => "blob:aicls",
      revokeObjectURL: () => undefined,
    });
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));
    await screen.findByTestId("result-latex");

    fireEvent.click(screen.getByRole("button", { name: "TeX olarak kaydet" }));

    await waitFor(() => {
      expect(
        countOf(
          mock,
          "POST",
          `/workspaces/ws-1/operations/${RESULT.operation_id as string}/export`,
        ),
      ).toBe(1);
    });
    const request = mock.mock.calls
      .map((entry) => entry[0] as Request)
      .find((entry) => entry.url.includes("/export"));
    expect(new URL(request?.url ?? "http://x/").searchParams.get("format")).toBe("tex");
    expect(screen.queryByText("Dışa aktarma yapılamadı.")).toBeNull();
  });

  it("shows the stage breakdown and the stub-model warning", async () => {
    installApiMock({ ...baseRoutes(), "POST /compute/text": () => computeResponse(RESULT) });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    expect(await screen.findByText("Toplam: 58 ms")).toBeInTheDocument();
    expect(screen.getByText("anlama")).toBeInTheDocument();

    // Stated once, and above the answer: a result with steps and a plot is taller than the board,
    // and a warning under all of it is a warning the room scrolls past. `compareDocumentPosition`
    // is the honest test in jsdom, which performs no layout.
    const warning = screen.getByTestId("model-stub-warning");
    const answer = screen.getByTestId("result-latex");
    expect(screen.getAllByTestId("model-stub-warning")).toHaveLength(1);
    expect(warning.compareDocumentPosition(answer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("says nothing about models when every one of them is a promoted checkpoint", async () => {
    // The warning must cost a normal answer no space at all, or it becomes chrome people stop
    // reading — and then it stops working on the day it matters.
    const trained: ComputeOut = { ...RESULT, model_versions: { ocr: "ocr-2027.03" } };
    installApiMock({ ...baseRoutes(), "POST /compute/text": () => computeResponse(trained) });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    await screen.findByTestId("result-latex");
    expect(screen.queryByTestId("model-stub-warning")).toBeNull();
  });

  it("reports the render latency with the operation id the server assigned", async () => {
    const mock = installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => computeResponse(RESULT),
    });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));
    await screen.findByTestId("result-latex");

    await waitFor(() => {
      expect(countOf(mock, "POST", "/telemetry/latency")).toBe(1);
    });
    const sample = (await bodyOf(mock, "POST", "/telemetry/latency")) as Record<string, unknown>;
    expect(sample.operation_id).toBe(RESULT.operation_id);
    expect(sample.modality).toBe("text");
    expect(Number(sample.render_ms)).toBeGreaterThanOrEqual(0);
    expect(Number(sample.client_render_ts)).toBeGreaterThanOrEqual(Number(sample.client_submit_ts));
  });

  it("keeps the previous answer on screen while the next command runs", async () => {
    let call = 0;
    installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => {
        call += 1;
        return call === 1
          ? computeResponse(RESULT)
          : // Never resolves: this is exactly the moment a lecture must not go blank.
            new Promise<Response>(() => undefined);
      },
    });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));
    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Komut"), { target: { value: "integralini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    expect(await screen.findByText("Yeni sonuç hazırlanıyor…")).toBeInTheDocument();
    expect(screen.getByTestId("result-latex")).toBeInTheDocument();
    expect(screen.getByTestId("result-plain")).toHaveTextContent("2*x");
  });
});

describe("workspace — clarification", () => {
  it("asks the question with tappable candidates and re-submits with confirm_intent", async () => {
    let call = 0;
    const mock = installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => {
        call += 1;
        return computeResponse(call === 1 ? CLARIFICATION : RESULT);
      },
    });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    expect(await screen.findByTestId("clarification")).toHaveTextContent(
      "Hangi değişkene göre türev alınsın?",
    );
    // A question is not an error: nothing on screen may claim it is.
    expect(screen.queryByTestId("problem-notice")).toBeNull();
    const candidates = screen.getAllByTestId("clarification-candidate");
    expect(candidates).toHaveLength(2);
    expect(candidates[0]).toHaveTextContent("x'e göre türev");

    fireEvent.click(candidates[0] as HTMLElement);

    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();
    const second = mock.mock.calls
      .map((entry) => entry[0] as Request)
      .filter((request) => request.method === "POST" && request.url.includes("/compute/text"))[1];
    expect(second).toBeDefined();
    // The board expression the question carried travels back as context, and the chosen intent
    // is echoed as `confirm_intent` so the CAS runs the reading the lecturer picked.
    expect(await second?.clone().json()).toEqual({
      text: "türevini al",
      workspace_id: "ws-1",
      context: { last_latex: "x^{2}" },
      confirm_intent: "differentiate",
    });
  });
});

describe("workspace — unsupported problem", () => {
  it("renders the contract sentence calmly instead of a crash screen", async () => {
    installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () =>
        problemResponse(
          "UNSUPPORTED_PROBLEM",
          422,
          "Bu problem tipi şu an için desteklenmemektedir.",
        ),
    });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "uzayı çöz" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    const notice = await screen.findByTestId("problem-notice");
    expect(notice).toHaveTextContent("Bu problem tipi şu an için desteklenmemektedir.");
    expect(notice).toHaveAttribute("data-code", "UNSUPPORTED_PROBLEM");
    expect(notice).toHaveTextContent("Komutu biraz farklı yazıp yeniden deneyebilirsiniz.");
    // Calm, not an alert: an unsupported problem is an answer, not a fault.
    expect(notice.getAttribute("role")).toBe("status");
    expect(
      screen.getByText("Henüz bir sonuç yok. Bir komut girerek başlayın."),
    ).toBeInTheDocument();
  });
});

describe("workspace — history", () => {
  it("re-runs a stored command with one tap", async () => {
    const mock = installApiMock({
      ...baseRoutes(),
      "GET /workspaces/ws-1/operations": { items: [OPERATION], next_cursor: null },
      "POST /compute/text": () => computeResponse(RESULT),
    });
    renderWorkspace();

    fireEvent.click(await screen.findByRole("button", { name: "Yeniden çalıştır" }));

    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();
    expect(await bodyOf(mock, "POST", "/compute/text")).toEqual({
      text: "x karenin türevini al",
      workspace_id: "ws-1",
      context: { last_latex: "x^{2}" },
    });
  });

  it("offers an undo before it deletes anything", async () => {
    const mock = installApiMock({
      ...baseRoutes(),
      "GET /workspaces/ws-1/operations": { items: [OPERATION], next_cursor: null },
    });
    renderWorkspace();

    fireEvent.click(await screen.findByRole("button", { name: "Sil" }));
    expect(screen.getByText("İşlem silindi.")).toBeInTheDocument();
    expect(countOf(mock, "DELETE", `/workspaces/ws-1/operations/${OPERATION.id}`)).toBe(0);

    fireEvent.click(screen.getByRole("button", { name: "Geri al" }));
    expect(
      within(screen.getByTestId("history-item")).getByText("x karenin türevini al"),
    ).toBeInTheDocument();
    expect(countOf(mock, "DELETE", `/workspaces/ws-1/operations/${OPERATION.id}`)).toBe(0);
  });
});

describe("workspace — keyboard", () => {
  it("focuses the command bar with /, submits with Ctrl+Enter and clears with Escape", async () => {
    const mock = installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => computeResponse(RESULT),
    });
    renderWorkspace();
    const input = await screen.findByLabelText("Komut");

    (document.activeElement as HTMLElement | null)?.blur();
    fireEvent.keyDown(document.body, { key: "/" });
    expect(document.activeElement).toBe(input);

    fireEvent.change(input, { target: { value: "x karenin türevini al" } });
    fireEvent.keyDown(input, { key: "Enter", ctrlKey: true });
    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();
    expect(countOf(mock, "POST", "/compute/text")).toBe(1);

    fireEvent.change(input, { target: { value: "silinecek" } });
    fireEvent.keyDown(input, { key: "Escape" });
    expect(input).toHaveValue("");
  });

  it("opens the shortcut sheet with ? and lists the documented bindings", async () => {
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    fireEvent.keyDown(document.body, { key: "?" });

    const dialog = await screen.findByRole("dialog", { name: "Klavye kısayolları" });
    expect(dialog).toHaveTextContent("Komut çubuğuna geç");
    expect(dialog).toHaveTextContent("Konuşmak için basılı tut");
    expect(dialog).toHaveTextContent("Sunum modunu aç/kapat");
  });

  it("switches to presentation mode from the keyboard", async () => {
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    fireEvent.keyDown(document.body, { key: "s", altKey: true });

    await waitFor(() => {
      expect(document.documentElement.getAttribute("data-density")).toBe("presentation");
    });
  });

  it("clears the wall when presentation mode starts, and puts it back when it ends", async () => {
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");
    const history = screen.getByRole("complementary", { name: "Ders akışı ve geçmiş" });
    expect(history).not.toHaveAttribute("hidden");

    fireEvent.keyDown(document.body, { key: "s", altKey: true });
    await waitFor(() => {
      expect(history).toHaveAttribute("hidden");
    });

    fireEvent.keyDown(document.body, { key: "s", altKey: true });
    await waitFor(() => {
      expect(history).not.toHaveAttribute("hidden");
    });
  });

  it("still opens the book on the wall in presentation mode (Alt + B stays honest)", async () => {
    // The regression this guards: the book rail was both `chrome-secondary` (a global
    // `display: none !important` in presentation mode) and hidden again by the page's own
    // `.presentation .bookRail` rule, so Alt + B, the "Kitap" button and a `kind: "book"` answer
    // all claimed a panel had opened while the lecturer saw nothing. İP-8 lives on that panel.
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    fireEvent.keyDown(document.body, { key: "s", altKey: true });
    await waitFor(() => {
      expect(document.documentElement.getAttribute("data-density")).toBe("presentation");
    });
    // Entering the mode closes it: a hidden landmark is out of the accessibility tree entirely.
    await waitFor(() => {
      expect(screen.queryByRole("complementary", { name: "Kitap paneli" })).toBeNull();
    });

    fireEvent.keyDown(document.body, { key: "b", altKey: true });

    const book = await screen.findByRole("complementary", { name: "Kitap paneli" });
    expect(book).toBeVisible();
    expect(document.documentElement.getAttribute("data-density")).toBe("presentation");
    // jsdom applies no CSS, so the class is the observable half of the fix: `chrome-secondary`
    // would put the global `display: none !important` back over the panel in a real browser.
    expect(book.className).not.toContain("chrome-secondary");
  });

  it("does not hide either rail with CSS while presentation mode is on", () => {
    // The other half of the same fix, asserted where it actually lives. Without this the page
    // could go back to `display: none` and every jsdom test above would still pass.
    const css = readFileSync(
      resolve("src/features/workspace/WorkspacePage.module.css"),
      "utf-8",
    ).replaceAll(/\/\*[\s\S]*?\*\//g, "");
    const presentationRules = css.slice(css.indexOf(".presentation"));

    expect(presentationRules).not.toMatch(
      /\.presentation[^{]*\.bookRail[^{]*\{[^}]*display:\s*none/,
    );
    expect(presentationRules).not.toMatch(/\.presentation[^{]*\.rail[^{]*\{[^}]*display:\s*none/);
  });
});

describe("workspace — client-side actions", () => {
  it("runs a ui command without replacing the answer, and repeats the previous command", async () => {
    const uiOutcome: ComputeOut = {
      kind: "ui",
      operation_id: null,
      input_latex: null,
      command: { op: "ui", action: "repeat_last" },
      result: null,
      clarification: null,
      book_units: [],
      nlu: { intent: "ui_repeat", score: 0.99, trace: [], source: "rules" },
      timings: { total: 3 },
      model_versions: {},
    };
    let call = 0;
    const mock = installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => {
        call += 1;
        return computeResponse(call === 2 ? uiOutcome : RESULT);
      },
    });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), { target: { value: "türevini al" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));
    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Komut"), { target: { value: "son komutu yinele" } });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    // The instruction is carried out (a third request) and it repeats the *computational* command,
    // not itself; the answer never left the board.
    await waitFor(() => {
      expect(countOf(mock, "POST", "/compute/text")).toBe(3);
    });
    const third = mock.mock.calls
      .map((entry) => entry[0] as Request)
      .filter((request) => request.method === "POST" && request.url.includes("/compute/text"))[2];
    expect(await third?.clone().json()).toMatchObject({ text: "türevini al" });
    expect(screen.getByTestId("result-latex")).toBeInTheDocument();
    expect(screen.queryByText(/Bu komut arayüzde çalıştırıldı/)).toBeNull();
  });
});

describe("workspace — book", () => {
  it("opens the book rail when a command resolves to textbook content", async () => {
    const bookOutcome: ComputeOut = {
      kind: "book",
      operation_id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7S",
      input_latex: null,
      command: { op: "book_fetch", query: "türev tanımı" },
      result: null,
      clarification: null,
      book_units: [
        {
          id: "tanim-2.4",
          type: "definition",
          title_tr: "Türev tanımı",
          chapter: "2",
          number: "2.4",
          aliases: [],
          jsonld: {},
          rendered: {
            body_html: "<p>Fark oranının limiti.</p>",
            latex_blocks: ["f'(x)"],
            assets: [],
          },
        },
      ],
      nlu: { intent: "book_lookup", score: 0.97, trace: [], source: "rules" },
      timings: { book: 8.1, total: 11.2 },
      model_versions: {},
    };
    installApiMock({
      ...baseRoutes(),
      "POST /compute/text": () => computeResponse(bookOutcome),
    });
    renderWorkspace();

    fireEvent.change(await screen.findByLabelText("Komut"), {
      target: { value: "türev tanımını getir" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));

    expect(await screen.findByTestId("book-unit")).toHaveTextContent("Türev tanımı");
    expect(screen.getByText("Fark oranının limiti.")).toBeInTheDocument();
    expect(screen.getByText("Bu komut kitap içeriğini getirdi; panel sağda açıldı.")).toBeVisible();
  });
});
