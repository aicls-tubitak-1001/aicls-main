/**
 * Handwriting: the live stream, the confirmation and the correction loop.
 *
 * `/ws/v1/ink` itself is covered by `sockets.test.ts`; here the socket is a fake so the assertions
 * are about what the panel does with what it receives — show a partial reading while the pen is
 * still moving, show the confidence, never turn a reading into a command without confirmation, and
 * turn an edited reading into an İP-3 training sample through `POST /ocr/feedback`.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi, type Mock } from "vitest";

import { InkPanel } from "@/features/workspace/InkPanel";
import { installApiMock, type RouteHandler } from "./apiMock";

/** The fake `useInkSocket` state; tests replace it and re-render. */
const socket = vi.hoisted((): { state: Record<string, unknown> } => ({ state: {} }));

vi.mock("@/api/sockets", () => ({
  useInkSocket: () => socket.state,
  useAudioSocket: () => ({}),
  setDefaultTicketSource: () => undefined,
  TARGET_SAMPLE_RATE: 16_000,
}));

function inkState(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    status: "open",
    partialLatex: null,
    finalLatex: null,
    alternatives: [],
    confidence: null,
    recognitionId: null,
    throttled: false,
    problem: null,
    sendStroke: () => undefined,
    penUp: () => undefined,
    finalize: () => undefined,
    clear: () => undefined,
    ...overrides,
  };
}

/** Draw one stroke with three samples. */
function drawStroke(canvas: HTMLElement, from: { x: number; y: number }): void {
  fireEvent.pointerDown(canvas, { clientX: from.x, clientY: from.y, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: from.x + 10, clientY: from.y + 4, pointerId: 1 });
  fireEvent.pointerUp(canvas, { clientX: from.x + 20, clientY: from.y + 9, pointerId: 1 });
}

async function bodyOf(mock: Mock, path: string): Promise<Record<string, unknown>> {
  for (const call of mock.mock.calls) {
    const request = call[0] as Request;
    if (new URL(request.url).pathname === `/api/v1${path}`) {
      return (await request.clone().json()) as Record<string, unknown>;
    }
  }
  throw new Error(`istek bulunamadı: ${path}`);
}

const NO_ROUTES: Record<string, RouteHandler> = {};

afterEach(() => {
  vi.unstubAllGlobals();
  socket.state = {};
});

describe("InkPanel", () => {
  it("shows the partial recognition with its confidence while the pen is still moving", () => {
    installApiMock(NO_ROUTES);
    socket.state = inkState({ partialLatex: "x^{2}", confidence: 0.42 });
    render(<InkPanel streamEnabled onAccept={() => undefined} />);

    const panel = screen.getByTestId("ink-recognition");
    expect(panel).toHaveTextContent("Tanıma güveni: %42");
    expect(panel).toHaveTextContent("Ön tanıma");
    expect(screen.getByLabelText("İfadeyi düzelt")).toHaveValue("x^{2}");
  });

  it("never computes a reading without confirmation: accepting is an explicit tap", () => {
    installApiMock(NO_ROUTES);
    const onAccept = vi.fn();
    socket.state = inkState({
      finalLatex: "\\frac{x}{2}",
      confidence: 0.91,
      recognitionId: "01J8RECOGNITION0000000000",
    });
    render(<InkPanel streamEnabled onAccept={onAccept} />);

    expect(onAccept).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Bu ifadeyi kullan" }));

    expect(onAccept).toHaveBeenCalledWith("\\frac{x}{2}");
  });

  it("offers the alternatives the recogniser returned", () => {
    installApiMock(NO_ROUTES);
    const onAccept = vi.fn();
    socket.state = inkState({
      finalLatex: "x^{2}",
      confidence: 0.71,
      alternatives: [
        ["x^{2}", 0.71],
        ["x^{z}", 0.19],
      ],
      recognitionId: "01J8RECOGNITION0000000000",
    });
    render(<InkPanel streamEnabled onAccept={onAccept} />);

    expect(screen.getByText("Diğer okumalar")).toBeInTheDocument();
    fireEvent.click(screen.getByText("%19").closest("button") as HTMLElement);
    fireEvent.click(screen.getByRole("button", { name: "Bu ifadeyi kullan" }));

    expect(onAccept).toHaveBeenCalledWith("x^{z}");
  });

  it("sends an edited reading back as a corrected sample", async () => {
    const mock = installApiMock({
      "POST /ocr/feedback": () => new Response(null, { status: 204 }),
    });
    socket.state = inkState({
      finalLatex: "x^{2}",
      confidence: 0.44,
      recognitionId: "01J8RECOGNITION0000000000",
    });
    render(<InkPanel streamEnabled onAccept={() => undefined} />);

    fireEvent.change(screen.getByLabelText("İfadeyi düzelt"), { target: { value: "x^{3}" } });
    fireEvent.click(screen.getByRole("button", { name: "Düzeltmeyi kaydet" }));

    await waitFor(() => {
      expect(screen.getByText("Düzeltme kaydedildi. Teşekkürler!")).toBeInTheDocument();
    });
    expect(await bodyOf(mock, "/ocr/feedback")).toEqual({
      recognition_id: "01J8RECOGNITION0000000000",
      corrected_latex: "x^{3}",
    });
  });

  it("falls back to the one-shot endpoint when the stream is not open", async () => {
    const mock = installApiMock({
      "POST /ocr/recognize": {
        latex: "x + 1",
        confidence: 0.83,
        alternatives: [],
        recognition_id: "01J8ONESHOT00000000000000",
        timings: { ocr: 12, total: 14 },
      },
    });
    socket.state = inkState({ status: "closed" });
    render(<InkPanel workspaceId="ws-1" onAccept={() => undefined} />);

    drawStroke(screen.getByTestId("ink-canvas"), { x: 12, y: 30 });
    fireEvent.click(screen.getByRole("button", { name: "Tanı" }));

    await waitFor(() => {
      expect(screen.getByTestId("ink-recognition")).toHaveTextContent("Tanıma güveni: %83");
    });
    const body = await bodyOf(mock, "/ocr/recognize");
    expect(body.workspace_id).toBe("ws-1");
    const canvasJson = body.ink as { version: number; strokes: unknown[] };
    expect(canvasJson.version).toBe(1);
    expect(canvasJson.strokes).toHaveLength(1);
  });

  it("clears the reading with the canvas", () => {
    installApiMock(NO_ROUTES);
    socket.state = inkState({ finalLatex: "x^{2}", confidence: 0.9 });
    render(<InkPanel streamEnabled onAccept={() => undefined} />);
    drawStroke(screen.getByTestId("ink-canvas"), { x: 20, y: 20 });
    expect(screen.getByTestId("ink-recognition")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Tuvali temizle" }));

    expect(screen.queryByTestId("ink-recognition")).toBeNull();
    expect(screen.getByText("Kalem kaldırıldığında tanıma yenilenir.")).toBeInTheDocument();
  });
});
