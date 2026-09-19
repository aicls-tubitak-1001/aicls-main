/**
 * The voice path and its confirmation step (api.md §3.6, ADR-0014).
 *
 * A wrong command in front of a class is worse than a slow one, so a spoken command is shown as a
 * transcript *and* as the parsed command before it counts. These tests drive the workspace through
 * a fake microphone: the recorder itself is exercised by `wavEncoder.test.ts`, what matters here is
 * what the lecturer sees between "I heard this" and "here is the answer".
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ComputeOut } from "@/api/types";
import { setAccessToken, sessionStore } from "@/app/auth";
import { resetAppState } from "@/app/store";
import { resetLatencyMarks } from "@/lib/latency";
import {
  CONFIRMATION_THRESHOLD,
  needsConfirmation,
  readVoiceConfirm,
} from "@/features/command/voicePreference";
import { WorkspacePage } from "@/features/workspace/WorkspacePage";
import { installApiMock, type RouteHandler } from "./apiMock";

/** Captured options of the mocked recorder, so a test can play the backend's answer back. */
const recorder = vi.hoisted(() => ({
  onOutcome: null as ((outcome: unknown, key: string) => void) | null,
  started: 0,
}));

vi.mock("@/features/command/useVoiceCommand", () => ({
  microphoneSupported: () => true,
  useVoiceCommand: (options: {
    onOutcome: (outcome: unknown, key: string) => void;
  }): Record<string, unknown> => {
    recorder.onOutcome = options.onOutcome;
    return {
      phase: "idle",
      seconds: 0,
      level: 0,
      supported: true,
      active: false,
      start: () => {
        recorder.started += 1;
      },
      stop: () => undefined,
      cancel: () => undefined,
      toggle: () => {
        recorder.started += 1;
      },
    };
  },
}));

function voiceOutcome(confidence: number): ComputeOut {
  return {
    kind: "result",
    operation_id: "01J8VOICE0000000000000000",
    input_latex: "x^{2}",
    command: { op: "diff", expr: "x^{2}", var: "x", order: 1 },
    result: {
      kind: "expr",
      latex: "2 x",
      plain: "2*x",
      srepr: "",
      steps: [],
      plot_url: null,
      cache_hit: false,
    },
    clarification: null,
    book_units: [],
    nlu: { intent: "differentiate", score: confidence, trace: [], source: "asr+rules" },
    timings: { asr: 420, nlu: 12, cas: 41, total: 500 },
    model_versions: { asr: "asr-stub-1" },
    transcript: "x karenin türevini al",
    confidence,
  };
}

function baseRoutes(): Record<string, RouteHandler> {
  return {
    "GET /workspaces": { items: [], next_cursor: null },
    "POST /workspaces": {
      id: "ws-1",
      title: "Ders",
      created_at: "2027-03-01T09:00:00Z",
      updated_at: "2027-03-01T09:00:00Z",
      n_operations: 0,
    },
    "GET /workspaces/ws-1/operations": { items: [], next_cursor: null },
    "POST /telemetry/latency": () => new Response(null, { status: 204 }),
  };
}

function renderWorkspace(): void {
  render(
    <MemoryRouter>
      <WorkspacePage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  recorder.onOutcome = null;
  recorder.started = 0;
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

describe("voice confirmation", () => {
  it("shows the transcript and the parsed command before the answer", async () => {
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    act(() => {
      recorder.onOutcome?.(voiceOutcome(0.62), "voice-1");
    });

    expect(await screen.findByText("Komutu onaylayın")).toBeInTheDocument();
    expect(screen.getByTestId("command-preview-transcript")).toHaveTextContent(
      "x karenin türevini al",
    );
    expect(screen.getByTestId("command-preview-command")).toHaveTextContent("diff(");
    expect(screen.getByText("Güven düşük; lütfen komutu kontrol edin.")).toBeInTheDocument();
    // The answer exists on the server but is not on the board yet.
    expect(screen.queryByTestId("result-latex")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Onayla ve çalıştır" }));

    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();
    expect(screen.getByTestId("result-plain")).toHaveTextContent("2*x");
    expect(screen.queryByText("Komutu onaylayın")).toBeNull();
  });

  it("hands the transcript back to the command bar for a one-tap correction", async () => {
    installApiMock(baseRoutes());
    renderWorkspace();
    const input = await screen.findByLabelText("Komut");

    act(() => {
      recorder.onOutcome?.(voiceOutcome(0.55), "voice-2");
    });
    await screen.findByText("Komutu onaylayın");

    fireEvent.click(screen.getByRole("button", { name: "Düzenle" }));

    expect(input).toHaveValue("x karenin türevini al");
    expect(document.activeElement).toBe(input);
    expect(screen.queryByTestId("result-latex")).toBeNull();
  });

  it("reports the latency of a spoken command when the understanding is displayed", async () => {
    const mock = installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    act(() => {
      recorder.onOutcome?.(voiceOutcome(0.62), "voice-3");
    });
    await screen.findByText("Komutu onaylayın");

    // No mark was opened by this test's fake recorder, so nothing may be posted: the measurement
    // is never invented on the client side.
    await waitFor(() => {
      expect(
        mock.mock.calls.filter((call) => (call[0] as Request).url.includes("/telemetry/latency"))
          .length,
      ).toBe(0);
    });
  });

  it("skips the step for a confident command once the lecturer turns it off", async () => {
    globalThis.localStorage.setItem("aicls.voiceConfirm", "off");
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    act(() => {
      recorder.onOutcome?.(voiceOutcome(0.95), "voice-4");
    });

    expect(await screen.findByTestId("result-latex")).toBeInTheDocument();
    expect(screen.queryByText("Komutu onaylayın")).toBeNull();
    expect(screen.getByText("“x karenin türevini al”")).toBeInTheDocument();
  });

  it("still asks when the confidence is low, whatever the preference says", async () => {
    globalThis.localStorage.setItem("aicls.voiceConfirm", "off");
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    act(() => {
      recorder.onOutcome?.(voiceOutcome(0.4), "voice-5");
    });

    expect(await screen.findByText("Komutu onaylayın")).toBeInTheDocument();
  });

  it("remembers the preference from the confirmation panel", async () => {
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    act(() => {
      recorder.onOutcome?.(voiceOutcome(0.62), "voice-6");
    });
    await screen.findByText("Komutu onaylayın");

    fireEvent.click(screen.getByLabelText(/Sesli komutlarda bir daha sorma/));

    expect(globalThis.localStorage.getItem("aicls.voiceConfirm")).toBe("off");
    expect(readVoiceConfirm()).toBe(false);
  });

  it("starts the recorder from the microphone button", async () => {
    installApiMock(baseRoutes());
    renderWorkspace();
    await screen.findByLabelText("Komut");

    fireEvent.click(screen.getByTestId("mic-button"));

    expect(recorder.started).toBe(1);
  });
});

describe("needsConfirmation", () => {
  it("always asks below the contract threshold", () => {
    expect(needsConfirmation(CONFIRMATION_THRESHOLD - 0.01, false)).toBe(true);
    expect(needsConfirmation(0, false)).toBe(true);
  });

  it("asks above the threshold only while the preference says so", () => {
    expect(needsConfirmation(0.95, true)).toBe(true);
    expect(needsConfirmation(0.95, false)).toBe(false);
  });

  it("asks when the backend reported no confidence at all", () => {
    expect(needsConfirmation(null, false)).toBe(true);
    expect(needsConfirmation(undefined, false)).toBe(true);
  });
});
