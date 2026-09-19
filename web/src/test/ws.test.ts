/**
 * `ws.test.ts` — contractual test from `docs/mimari/moduller.md#web--sahibi-sg`: the typed socket
 * reconnects after an unclean close and honours `slow_down` backpressure (api.md §4, ADR-0014).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resolveWsUrl, TypedSocket, type WebSocketLike } from "@/api/ws";

/** A scriptable `WebSocket` stand-in; every instance is recorded for assertions. */
class FakeSocket implements WebSocketLike {
  static instances: FakeSocket[] = [];

  readyState = 0;
  readonly sent: (string | ArrayBufferLike | ArrayBufferView | Blob)[] = [];
  onopen: ((event: unknown) => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onclose: ((event: { code?: number }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  closedWith: number | null = null;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  send(data: string | ArrayBufferLike | ArrayBufferView | Blob): void {
    this.sent.push(data);
  }

  close(code?: number): void {
    this.readyState = 3;
    this.closedWith = code ?? null;
  }

  /** Simulate the server accepting the connection. */
  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }

  /** Simulate one server envelope. */
  emit(envelope: unknown): void {
    this.onmessage?.({ data: JSON.stringify(envelope) });
  }

  /** Simulate a dropped connection (code 1006 = abnormal closure). */
  drop(): void {
    this.readyState = 3;
    this.onclose?.({ code: 1006 });
  }

  /** Envelopes sent as JSON, parsed back. */
  get envelopes(): { type: string; seq?: number }[] {
    return this.sent
      .filter((item): item is string => typeof item === "string")
      .map((item) => JSON.parse(item) as { type: string; seq?: number });
  }
}

const factory = (url: string): WebSocketLike => new FakeSocket(url);

function latest(): FakeSocket {
  const socket = FakeSocket.instances[FakeSocket.instances.length - 1];
  if (socket === undefined) {
    throw new Error("no socket was created");
  }
  return socket;
}

beforeEach(() => {
  FakeSocket.instances = [];
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("resolveWsUrl", () => {
  it("switches http to ws and appends the ticket", () => {
    expect(resolveWsUrl("/ws/v1/ink", "abc123", "http://127.0.0.1:8000/")).toBe(
      "ws://127.0.0.1:8000/ws/v1/ink?ticket=abc123",
    );
  });

  it("switches https to wss", () => {
    expect(resolveWsUrl("/ws/v1/audio", "t", "https://ai-cls.example/app")).toBe(
      "wss://ai-cls.example/ws/v1/audio?ticket=t",
    );
  });
});

describe("TypedSocket", () => {
  it("queues messages until the socket is open, then flushes them in order", () => {
    const socket = new TypedSocket("/ws/v1/ink", "ticket-1", {
      createSocket: factory,
      origin: "http://127.0.0.1:8000/",
    });

    socket.send({ type: "stroke", payload: { points: [] } });
    expect(socket.pending).toBe(1);
    expect(latest().sent).toHaveLength(0);

    latest().open();

    expect(socket.status).toBe("open");
    expect(socket.pending).toBe(0);
    expect(latest().envelopes).toEqual([{ type: "stroke", payload: { points: [] }, seq: 1 }]);
  });

  it("numbers envelopes with an increasing client sequence", () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
    });
    latest().open();

    socket.send({ type: "pen_up" });
    socket.send({ type: "finalize" });

    expect(latest().envelopes.map((envelope) => envelope.seq)).toEqual([1, 2]);
  });

  it("honours slow_down: sends are held for retry_after_ms and then flushed", async () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
    });
    latest().open();
    socket.send({ type: "stroke", payload: { points: [] } });
    expect(latest().envelopes).toHaveLength(1);

    latest().emit({ type: "slow_down", payload: { retry_after_ms: 500 } });

    expect(socket.isThrottled).toBe(true);
    socket.send({ type: "stroke", payload: { points: [1] } });
    socket.send({ type: "pen_up" });
    expect(socket.pending).toBe(2);
    expect(latest().envelopes).toHaveLength(1); // nothing new reached the wire

    await vi.advanceTimersByTimeAsync(499);
    expect(latest().envelopes).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(1);
    expect(socket.isThrottled).toBe(false);
    expect(socket.pending).toBe(0);
    expect(latest().envelopes.map((envelope) => envelope.type)).toEqual([
      "stroke",
      "stroke",
      "pen_up",
    ]);
  });

  it("delivers slow_down to subscribers as well", () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
    });
    const seen: string[] = [];
    socket.onMessage((message) => seen.push(message.type));
    latest().open();

    latest().emit({ type: "slow_down", payload: {} });
    latest().emit({ type: "partial", payload: { latex: "x" } });

    expect(seen).toEqual(["slow_down", "partial"]);
  });

  it("reconnects after an unclean close and flushes what was queued meanwhile", async () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
      baseReconnectDelayMs: 100,
    });
    latest().open();
    expect(FakeSocket.instances).toHaveLength(1);

    latest().drop();
    expect(socket.status).toBe("reconnecting");

    socket.send({ type: "pen_up" });
    expect(socket.pending).toBe(1);

    await vi.advanceTimersByTimeAsync(100);
    expect(FakeSocket.instances).toHaveLength(2);

    latest().open();
    expect(socket.status).toBe("open");
    expect(latest().envelopes.map((envelope) => envelope.type)).toEqual(["pen_up"]);
  });

  it("backs off exponentially and gives up after maxReconnectAttempts", async () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
      baseReconnectDelayMs: 100,
      maxReconnectAttempts: 2,
    });
    latest().open();

    latest().drop();
    await vi.advanceTimersByTimeAsync(100); // first retry: 100 ms
    expect(FakeSocket.instances).toHaveLength(2);

    latest().drop();
    await vi.advanceTimersByTimeAsync(199);
    expect(FakeSocket.instances).toHaveLength(2); // second retry waits 200 ms
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeSocket.instances).toHaveLength(3);

    latest().drop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(FakeSocket.instances).toHaveLength(3);
    expect(socket.status).toBe("closed");
  });

  it("fetches a fresh ticket for every reconnect (tickets are single-use)", async () => {
    const tickets = ["t2", "t3"];
    const socket = new TypedSocket("/ws/v1/ink", "t1", {
      createSocket: factory,
      origin: "http://x/",
      baseReconnectDelayMs: 10,
      ticketProvider: () => Promise.resolve(tickets.shift() ?? "exhausted"),
    });
    latest().open();
    expect(latest().url).toContain("ticket=t1");

    latest().drop();
    await vi.advanceTimersByTimeAsync(10);

    expect(latest().url).toContain("ticket=t2");
    expect(socket.status).not.toBe("closed");
  });

  it("does not reconnect after close() and stops delivering messages", async () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
      baseReconnectDelayMs: 10,
    });
    const seen: string[] = [];
    socket.onMessage((message) => seen.push(message.type));
    const first = latest();
    first.open();

    socket.close();

    expect(socket.status).toBe("closed");
    expect(first.closedWith).toBe(1000);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeSocket.instances).toHaveLength(1);
    expect(seen).toEqual([]);
  });

  it("unsubscribes a listener when its disposer is called", () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
    });
    const seen: string[] = [];
    const off = socket.onMessage((message) => seen.push(message.type));
    latest().open();

    latest().emit({ type: "partial", payload: {} });
    off();
    latest().emit({ type: "final", payload: {} });

    expect(seen).toEqual(["partial"]);
  });

  it("ignores malformed frames instead of crashing the stream", () => {
    const socket = new TypedSocket("/ws/v1/ink", "t", {
      createSocket: factory,
      origin: "http://x/",
    });
    const seen: string[] = [];
    socket.onMessage((message) => seen.push(message.type));
    latest().open();

    latest().onmessage?.({ data: "{not json" });
    latest().emit({ type: "pong" });

    expect(seen).toEqual(["pong"]);
  });

  it("sends binary frames untouched", () => {
    const socket = new TypedSocket("/ws/v1/audio", "t", {
      createSocket: factory,
      origin: "http://x/",
    });
    latest().open();
    const frame = new ArrayBuffer(8);

    socket.sendBinary(frame);

    expect(latest().sent[0]).toBe(frame);
  });
});
