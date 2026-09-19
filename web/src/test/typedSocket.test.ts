/**
 * `TypedSocket` behaviours the streams depend on beyond the contractual reconnect test in
 * `ws.test.ts`: the close codes the server actually uses, and the backpressure callback the
 * connection badge listens to.
 *
 * The close codes are the ones in `aicls.api.ws.protocol`: 1008 when a ticket is refused (the
 * client must come back with a fresh one, because tickets are single-use) and 1013 when the
 * server closed the socket for flooding.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  AUDIO_PATH,
  INK_PATH,
  TypedSocket,
  WS_POLICY_VIOLATION,
  WS_TRY_AGAIN_LATER,
  type WebSocketLike,
} from "@/api/ws";

class FakeSocket implements WebSocketLike {
  static instances: FakeSocket[] = [];

  readyState = 0;
  readonly sent: unknown[] = [];
  onopen: ((event: unknown) => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onclose: ((event: { code?: number; reason?: string; wasClean?: boolean }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  send(data: unknown): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = 3;
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.({});
  }

  emit(envelope: unknown): void {
    this.onmessage?.({ data: JSON.stringify(envelope) });
  }

  serverClose(code: number, wasClean = true): void {
    this.readyState = 3;
    this.onclose?.({ code, wasClean });
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

beforeEach(() => {
  FakeSocket.instances = [];
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("close codes", () => {
  it("stays closed after a clean server close (1000) instead of looping", async () => {
    const socket = new TypedSocket(INK_PATH, "t", {
      createSocket: factory,
      origin: "http://x/",
      baseReconnectDelayMs: 10,
    });
    latest().open();

    latest().serverClose(1000);

    expect(socket.status).toBe("closed");
    expect(socket.lastCloseCode).toBe(1000);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeSocket.instances).toHaveLength(1);
  });

  it("reconnects with a fresh ticket after 1008 (a ticket is single-use)", async () => {
    const tickets = ["t2"];
    const socket = new TypedSocket(INK_PATH, "t1", {
      createSocket: factory,
      origin: "http://x/",
      baseReconnectDelayMs: 10,
      ticketProvider: () => Promise.resolve(tickets.shift() ?? "exhausted"),
    });
    latest().open();

    latest().serverClose(WS_POLICY_VIOLATION, false);
    expect(socket.status).toBe("reconnecting");
    expect(socket.lastCloseCode).toBe(WS_POLICY_VIOLATION);

    await vi.advanceTimersByTimeAsync(10);

    expect(latest().url).toContain("ticket=t2");
  });

  it("reconnects after the server closes a flooding socket (1013)", async () => {
    const socket = new TypedSocket(AUDIO_PATH, "t", {
      createSocket: factory,
      origin: "http://x/",
      baseReconnectDelayMs: 10,
    });
    latest().open();

    latest().serverClose(WS_TRY_AGAIN_LATER, false);
    await vi.advanceTimersByTimeAsync(10);

    expect(FakeSocket.instances).toHaveLength(2);
    expect(latest().url).toContain(AUDIO_PATH);
    expect(socket.reconnectAttempts).toBe(1);
  });
});

describe("backpressure", () => {
  it("reports the pause the server asked for and lifts it afterwards", async () => {
    const pauses: number[] = [];
    const socket = new TypedSocket(INK_PATH, "t", {
      createSocket: factory,
      origin: "http://x/",
      onSlowDown: (ms) => pauses.push(ms),
    });
    latest().open();

    latest().emit({ type: "slow_down", payload: { retry_after_ms: 250 } });

    expect(pauses).toEqual([250]);
    expect(socket.isThrottled).toBe(true);

    socket.send({ type: "pen_up", payload: {} });
    expect(socket.pending).toBe(1);

    await vi.advanceTimersByTimeAsync(250);

    expect(socket.isThrottled).toBe(false);
    expect(socket.pending).toBe(0);
  });

  it("falls back to the default pause when the envelope omits retry_after_ms", () => {
    const pauses: number[] = [];
    new TypedSocket(INK_PATH, "t", {
      createSocket: factory,
      origin: "http://x/",
      defaultSlowDownMs: 750,
      onSlowDown: (ms) => pauses.push(ms),
    });
    latest().open();

    latest().emit({ type: "slow_down", payload: {} });

    expect(pauses).toEqual([750]);
  });

  it("ignores a frame that is not an envelope", () => {
    const socket = new TypedSocket(INK_PATH, "t", { createSocket: factory, origin: "http://x/" });
    const seen: string[] = [];
    socket.onMessage((message) => seen.push(message.type));
    latest().open();

    latest().emit([1, 2, 3]);
    latest().emit({ payload: {} });
    latest().emit({ type: "pong", payload: {} });

    expect(seen).toEqual(["pong"]);
  });
});
