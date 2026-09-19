/**
 * Application shell and routing.
 *
 * A smoke test per route (the "akıllı duman testi" the module table asks for): the shell renders in
 * Turkish, each Turkish path mounts its page, and an unknown path shows the 404 copy. The workspace
 * route also exercises the store wiring — command bar, canvas, result panel and history in one tree.
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "@/App";
import { appStore, resetAppState } from "@/app/store";
import { ROUTES } from "@/routes";
import { installApiMock } from "./apiMock";

function renderAt(path: string): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  resetAppState();
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.removeAttribute("data-density");
  globalThis.localStorage.clear();
});

describe("App shell", () => {
  it("renders the Turkish navigation and the skip link", () => {
    installApiMock({});
    renderAt(ROUTES.workspace);

    expect(screen.getByRole("navigation", { name: "Ana gezinme" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "İçeriğe geç" })).toHaveAttribute("href", "#main");
    expect(screen.getByRole("link", { name: "Çalışma Alanı" })).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Model Gelişimine Katkıda Bulun" }),
    ).toBeInTheDocument();
  });

  it("applies an explicit theme choice to the document", () => {
    installApiMock({});
    renderAt(ROUTES.workspace);

    fireEvent.change(screen.getByLabelText("Görünüm"), { target: { value: "dark" } });

    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });

  it("offers the presentation switch on every route, not only on the board", () => {
    // Presentation mode is written on <html>, so it survives a route change. Its switch has to
    // survive the same change: a lecturer who walked to the guide with the wall still scaled up
    // must be able to turn it off from there.
    installApiMock({});
    renderAt(ROUTES.guide);

    const toggle = screen.getByRole("button", { name: "Sunum moduna geç" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(toggle);

    expect(document.documentElement.getAttribute("data-density")).toBe("presentation");
    const off = screen.getByRole("button", { name: "Sunum modundan çık" });
    expect(off).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(off);
    expect(document.documentElement.getAttribute("data-density")).toBe("normal");
  });

  it("keeps the shell switch and the board's Alt + S on the same state", async () => {
    installApiMock({});
    renderAt(ROUTES.workspace);
    await screen.findByLabelText("Komut");

    fireEvent.keyDown(document.body, { key: "s", altKey: true });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Sunum modundan çık" })).toHaveAttribute(
        "aria-pressed",
        "true",
      );
    });
    expect(document.documentElement.getAttribute("data-density")).toBe("presentation");
  });

  it("mounts the workspace with its command bar, canvas and result panel", async () => {
    installApiMock({});
    renderAt(ROUTES.workspace);

    expect(screen.getByLabelText("Komut")).toBeInTheDocument();
    expect(screen.getByTestId("ink-canvas")).toBeInTheDocument();
    expect(
      screen.getByText("Henüz bir sonuç yok. Bir komut girerek başlayın."),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Bu çalışma alanında henüz işlem yok.")).toBeInTheDocument();
    });
  });

  it("asks the clarification question when the store holds a clarification outcome", () => {
    installApiMock({});
    renderAt(ROUTES.workspace);

    act(() => {
      appStore.setState({
        phase: "awaiting-confirmation",
        outcome: {
          kind: "clarification",
          operation_id: null,
          input_latex: "x^{2}",
          command: null,
          result: null,
          clarification: {
            question_tr: "Hangi değişkene göre?",
            missing_slot: "var",
            candidates: [
              {
                intent: "differentiate",
                command: { op: "diff", expr: "x^{2}", var: "x" },
                latex_preview: "2x",
                score: 0.61,
                label_tr: "x'e göre türev",
              },
            ],
          },
          book_units: [],
          nlu: null,
          timings: { total: 15 },
          model_versions: {},
        },
      });
    });

    // A clarification is a 200, so the workspace asks it as a question with tappable candidates —
    // the voice confirmation panel is a different thing and must not appear here.
    expect(screen.getByTestId("clarification")).toHaveTextContent("Hangi değişkene göre?");
    expect(screen.getByTestId("clarification-candidate")).toHaveTextContent("x'e göre türev");
    expect(screen.queryByText("Komutu onaylayın")).toBeNull();
  });

  it("mounts the book route", () => {
    installApiMock({});
    renderAt(ROUTES.book);

    expect(screen.getByLabelText("Tanım, teorem veya örnek arayın")).toBeInTheDocument();
  });

  it("mounts the SUS questionnaire on the feedback route", () => {
    installApiMock({});
    renderAt(ROUTES.feedback);

    expect(screen.getByText("Kullanılabilirlik Anketi (SUS)")).toBeInTheDocument();
  });

  it("mounts the consent gate on the contribute route", async () => {
    installApiMock({
      "GET /contribute/consent/current": {
        version: "1",
        text_tr: "Onam metni",
        sha256: "sha256:1",
      },
    });
    renderAt(ROUTES.contribute);

    expect(await screen.findByTestId("consent-text")).toHaveTextContent("Onam metni");
  });

  it("redirects the root path to the workspace", () => {
    installApiMock({});
    renderAt("/");

    expect(screen.getByTestId("ink-canvas")).toBeInTheDocument();
  });

  it("shows the Turkish 404 copy for an unknown path", () => {
    installApiMock({});
    renderAt("/olmayan-sayfa");

    expect(screen.getByText("Aradığınız sayfa bulunamadı.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Çalışma alanına dön" })).toHaveAttribute(
      "href",
      ROUTES.workspace,
    );
  });
});
