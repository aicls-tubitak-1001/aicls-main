/**
 * The two streaming hooks (`/ws/v1/ink`, `/ws/v1/audio`, api.md §4).
 *
 * A scriptable socket stands in for the server: the hooks are responsible for the ticket handshake,
 * for turning envelopes into React state, and for closing the connection when the page unmounts.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { WebSocketLike } from "@/api/ws";
import { useAudioSocket } from "@/features/voice/useAudioSocket";
import { useInkSocket } from "@/features/workspace/useInkSocket";

class FakeSocket implements WebSocketLike {
  static instances: FakeSocket[] = [];

  readyState = 0;
  readonly sent: unknown[] = [];
  onopen: ((event: unknown) => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onclose: ((event: { code?: number }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  send(data: unknown): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = 3;
    this.closed = true;
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }

  emit(envelope: unknown): void {
    this.onmessage?.({ data: JSON.stringify(envelope) });
  }

  get envelopes(): { type: string }[] {
    return this.sent
      .filter((item): item is string => typeof item === "string")
      .map((item) => JSON.parse(item) as { type: string });
  }
}

const factory = (url: string): WebSocketLike => new FakeSocket(url);

function latest(): FakeSocket {
  const socket = FakeSocket.instances[FakeSocket.instances.length - 1];
  if (socket === undefined) {
    throw new Error("soket kurulmadı");
  }
  return socket;
}

describe("useInkSocket", () => {
  it("connects with a ticket and turns partial/final envelopes into state", async () => {
    FakeSocket.instances = [];
    const getTicket = vi.fn(() => Promise.resolve("ticket-1"));
    const { result, unmount } = renderHook(() =>
      useInkSocket({ createSocket: factory, getTicket }),
    );

    await waitFor(() => {
      expect(FakeSocket.instances).toHaveLength(1);
    });
    expect(latest().url).toContain("ticket=ticket-1");
    expect(latest().url).toContain("/ws/v1/ink");

    act(() => {
      latest().open();
    });
    await waitFor(() => {
      expect(result.current.status).toBe("open");
    });

    act(() => {
      latest().emit({ type: "partial", payload: { latex: "x^", confidence: 0.4 } });
    });
    expect(result.current.partialLatex).toBe("x^");

    act(() => {
      latest().emit({
        type: "final",
        payload: { latex: "x^{2}", alternatives: [], recognition_id: "rec-1" },
      });
    });
    expect(result.current.finalLatex).toBe("x^{2}");
    expect(result.current.recognitionId).toBe("rec-1");
    expect(result.current.partialLatex).toBeNull();

    unmount();
    expect(latest().closed).toBe(true);
  });

  it("sends stroke, pen_up, finalize and clear envelopes", async () => {
    FakeSocket.instances = [];
    const { result } = renderHook(() =>
      useInkSocket({ createSocket: factory, getTicket: () => Promise.resolve("t") }),
    );
    await waitFor(() => {
      expect(FakeSocket.instances).toHaveLength(1);
    });
    act(() => {
      latest().open();
    });

    act(() => {
      result.current.sendStroke({ points: [{ x: 1, y: 2, t: 0 }] }, { w: 400, h: 120 });
      result.current.penUp();
      result.current.finalize();
      result.current.clear();
    });

    expect(latest().envelopes.map((envelope) => envelope.type)).toEqual([
      "stroke",
      "pen_up",
      "finalize",
      "clear",
    ]);
  });

  it("surfaces backpressure", async () => {
    FakeSocket.instances = [];
    const { result } = renderHook(() =>
      useInkSocket({ createSocket: factory, getTicket: () => Promise.resolve("t") }),
    );
    await waitFor(() => {
      expect(FakeSocket.instances).toHaveLength(1);
    });
    act(() => {
      latest().open();
      latest().emit({ type: "slow_down", payload: { retry_after_ms: 500 } });
    });

    expect(result.current.throttled).toBe(true);
  });

  it("stays closed when the ticket cannot be fetched", async () => {
    FakeSocket.instances = [];
    const { result } = renderHook(() =>
      useInkSocket({
        createSocket: factory,
        getTicket: () => Promise.reject(new Error("401")),
      }),
    );

    await waitFor(() => {
      expect(result.current.status).toBe("closed");
    });
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it("opens no socket when disabled", () => {
    FakeSocket.instances = [];
    renderHook(() => useInkSocket({ enabled: false, createSocket: factory }));

    expect(FakeSocket.instances).toHaveLength(0);
  });
});

describe("useAudioSocket", () => {
  it("announces the PCM16 format, streams frames and receives the command outcome", async () => {
    FakeSocket.instances = [];
    const { result } = renderHook(() =>
      useAudioSocket({ createSocket: factory, getTicket: () => Promise.resolve("t") }),
    );
    await waitFor(() => {
      expect(FakeSocket.instances).toHaveLength(1);
    });
    expect(latest().url).toContain("/ws/v1/audio");
    act(() => {
      latest().open();
    });

    act(() => {
      result.current.start();
      result.current.sendFrame(new ArrayBuffer(8_000));
      result.current.stop();
    });

    const envelopes = latest().envelopes;
    expect(envelopes[0]).toMatchObject({
      type: "start",
      payload: { sample_rate: 16_000, format: "pcm16" },
    });
    expect(envelopes.map((envelope) => envelope.type)).toEqual(["start", "stop"]);
    expect(latest().sent.some((item) => item instanceof ArrayBuffer)).toBe(true);

    act(() => {
      latest().emit({
        type: "final",
        payload: { text: "x karenin türevini al", confidence: 0.91, transcript_id: "tr-1" },
      });
      latest().emit({ type: "command", payload: { outcome: { kind: "result" } } });
    });

    expect(result.current.finalText).toBe("x karenin türevini al");
    expect(result.current.confidence).toBeCloseTo(0.91, 5);
    expect(result.current.outcome).toEqual({ kind: "result" });
  });

  it("clears the pending utterance on cancel", async () => {
    FakeSocket.instances = [];
    const { result } = renderHook(() =>
      useAudioSocket({ createSocket: factory, getTicket: () => Promise.resolve("t") }),
    );
    await waitFor(() => {
      expect(FakeSocket.instances).toHaveLength(1);
    });
    act(() => {
      latest().open();
      latest().emit({ type: "partial", payload: { text: "x kare" } });
    });
    expect(result.current.partialText).toBe("x kare");

    act(() => {
      result.current.cancel();
    });

    expect(result.current.partialText).toBeNull();
    expect(latest().envelopes.map((envelope) => envelope.type)).toEqual(["cancel"]);
  });
});
