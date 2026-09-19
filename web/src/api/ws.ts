/**
 * Typed WebSocket client for `/ws/v1/ink` and `/ws/v1/audio` (api.md §4, ADR-0014).
 *
 * The envelope and the message types are transcribed from the server, not guessed: the frame is
 * `{type, seq, payload}` (`aicls.api.ws.protocol.WsEnvelope`), the accepted client types are the
 * `ACCEPTED` sets of `aicls.api.ws.ink` / `aicls.api.ws.audio`, and the server types are the
 * `InkServerType` / `AudioServerType` aliases of `aicls.api.ws.protocol`. The two streams are not
 * part of the OpenAPI document (they are registered with `add_api_websocket_route`), so this file
 * is the client-side half of that contract.
 *
 * What the class adds to a raw `WebSocket`:
 *
 * - **ticket auth** — the URL carries a single-use 60 s ticket from `POST /auth/ws-ticket`, never
 *   a JWT (api.md §1). `ticketProvider` supplies a fresh one for every reconnect, because the
 *   first connection consumes it (`WsTicketStore.consume`).
 * - **envelopes** — outgoing frames get a client-incremented `seq`, which the server echoes on the
 *   frame that answers them.
 * - **reconnect** — an unclean close schedules an exponential backoff (250 ms → 4 s) with a
 *   visible status, and the queue is flushed once the new socket is open. A clean server close
 *   (1000) is respected: reconnecting after a deliberate close would be a loop.
 * - **backpressure** — `slow_down{retry_after_ms}` pauses sending for exactly that long and
 *   queues instead of flooding the recogniser (api.md §4.1, `MessageRate` on the server).
 *
 * The socket factory is injectable (`options.createSocket`) so tests drive open/close/message
 * transitions without a server; production uses `globalThis.WebSocket`.
 */
import type { ComputeOut } from "./types";
import { WS_BASE_PATH } from "./schema";

export { WS_BASE_PATH };

/** Path of the handwriting stream (`app.py`: `add_api_websocket_route(f"{WS_PREFIX}/ink")`). */
export const INK_PATH = `${WS_BASE_PATH}/ink`;

/** Path of the speech stream. */
export const AUDIO_PATH = `${WS_BASE_PATH}/audio`;

/** Close code used when a ticket is missing, expired or already redeemed (`WS_POLICY_VIOLATION`). */
export const WS_POLICY_VIOLATION = 1008;

/** Close code used when the client keeps flooding after `slow_down` (`WS_TRY_AGAIN_LATER`). */
export const WS_TRY_AGAIN_LATER = 1013;

/** Connection state surfaced to the UI. */
export type SocketStatus = "connecting" | "open" | "reconnecting" | "closed";

/** Minimal `WebSocket` surface used by {@link TypedSocket} (keeps the class testable). */
export interface WebSocketLike {
  readyState: number;
  send(data: string | ArrayBufferLike | ArrayBufferView | Blob): void;
  close(code?: number, reason?: string): void;
  onopen: ((event: unknown) => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: ((event: { code?: number; reason?: string; wasClean?: boolean }) => void) | null;
  onerror: ((event: unknown) => void) | null;
}

/** Client → server envelope; `seq` is filled in by {@link TypedSocket.send}. */
export interface ClientEnvelope {
  type: string;
  seq?: number;
  payload?: unknown;
}

/** Server → client envelope. */
export interface ServerEnvelope {
  type: string;
  seq?: number;
  payload?: unknown;
}

/** Payload of the `slow_down` envelope (api.md §4.1). */
export interface SlowDownPayload {
  retry_after_ms?: number;
}

/** Payload of an `error` envelope: the RFC 9457 document the raising error produced. */
export interface WsErrorPayload {
  problem?: unknown;
}

/* --------------------------------------------------------------------------- ink stream --- */

/** One sampled pointer position (`web/src/lib/inkml.ts`, `aicls.ocr.canvas`). */
export interface InkPointPayload {
  x: number;
  y: number;
  t: number;
}

/** Size of the drawing surface in CSS pixels. */
export interface InkCanvasPayload {
  w: number;
  h: number;
}

/** Client → server envelopes of `/ws/v1/ink` (`aicls.api.ws.ink.ACCEPTED`). */
export type InkClientMessage =
  | {
      type: "stroke";
      seq?: number;
      payload: { points: InkPointPayload[]; canvas: InkCanvasPayload };
    }
  | { type: "pen_up"; seq?: number; payload?: Record<string, never> }
  | { type: "clear"; seq?: number; payload?: Record<string, never> }
  | { type: "finalize"; seq?: number; payload?: Record<string, never> }
  | { type: "ping"; seq?: number; payload?: Record<string, never> };

/** Server → client envelopes of `/ws/v1/ink` (`aicls.api.ws.protocol.InkServerType`). */
export type InkServerMessage =
  | { type: "partial"; seq?: number; payload: { latex: string; confidence: number } }
  | {
      type: "final";
      seq?: number;
      payload: {
        latex: string;
        alternatives: [string, number][];
        /** `null` when the recognition was not persisted (an unauthenticated probe). */
        recognition_id: string | null;
        confidence: number;
      };
    }
  | { type: "error"; seq?: number; payload: WsErrorPayload }
  | { type: "slow_down"; seq?: number; payload: SlowDownPayload }
  | { type: "pong"; seq?: number; payload?: Record<string, never> };

/* ------------------------------------------------------------------------- audio stream --- */

/** Client → server envelopes of `/ws/v1/audio` (binary PCM16 frames travel separately). */
export type AudioClientMessage =
  | { type: "start"; seq?: number; payload: { sample_rate: number; format: "pcm16" } }
  | { type: "stop"; seq?: number; payload?: Record<string, never> }
  | { type: "cancel"; seq?: number; payload?: Record<string, never> }
  | { type: "ping"; seq?: number; payload?: Record<string, never> };

/** Server → client envelopes of `/ws/v1/audio` (`aicls.api.ws.protocol.AudioServerType`). */
export type AudioServerMessage =
  | { type: "partial"; seq?: number; payload: { text: string; duration_s?: number } }
  | {
      type: "final";
      seq?: number;
      payload: { text: string; confidence: number; transcript_id: string | null };
    }
  | { type: "command"; seq?: number; payload: { outcome: ComputeOut } }
  | { type: "error"; seq?: number; payload: WsErrorPayload }
  | { type: "slow_down"; seq?: number; payload: SlowDownPayload }
  | { type: "pong"; seq?: number; payload?: Record<string, never> };

/* ------------------------------------------------------------------------------ client --- */

/** Construction options; every field has a production default. */
export interface TypedSocketOptions {
  /** Socket factory — defaults to `new WebSocket(url)`. */
  createSocket?: (url: string) => WebSocketLike;
  /** Fresh ticket for each reconnect (tickets are single-use); defaults to reusing the first one. */
  ticketProvider?: () => string | Promise<string>;
  /** First reconnect delay in ms (doubles each attempt). */
  baseReconnectDelayMs?: number;
  /** Upper bound for the reconnect delay in ms. */
  maxReconnectDelayMs?: number;
  /** Give up after this many consecutive failed reconnects. */
  maxReconnectAttempts?: number;
  /** Default pause applied when `slow_down` omits `retry_after_ms`. */
  defaultSlowDownMs?: number;
  /** Called on every state transition (used by the connection badge). */
  onStatusChange?: (status: SocketStatus) => void;
  /** Called when the server asks for a slower rate, with the pause it requested. */
  onSlowDown?: (retryAfterMs: number) => void;
  /** Origin used to build the `ws(s)://` URL; defaults to `window.location`. */
  origin?: string;
}

const OPEN = 1;

/** Build the absolute `ws(s)://` URL for a WebSocket path and ticket. */
export function resolveWsUrl(path: string, ticket: string, origin?: string): string {
  const base =
    origin ??
    (typeof globalThis.location === "undefined"
      ? "http://127.0.0.1:8000"
      : globalThis.location.href);
  const url = new URL(path, base);
  url.protocol =
    url.protocol === "https:" ? "wss:" : url.protocol === "http:" ? "ws:" : url.protocol;
  url.searchParams.set("ticket", ticket);
  return url.toString();
}

/**
 * A reconnecting, backpressure-aware WebSocket carrying typed envelopes.
 *
 * @typeParam TIn - Client → server envelope union for this stream.
 * @typeParam TOut - Server → client envelope union for this stream.
 */
export class TypedSocket<
  TIn extends ClientEnvelope = ClientEnvelope,
  TOut extends ServerEnvelope = ServerEnvelope,
> {
  private readonly path: string;
  private ticket: string;
  private readonly options: Required<
    Omit<TypedSocketOptions, "ticketProvider" | "origin" | "onSlowDown">
  > &
    Pick<TypedSocketOptions, "ticketProvider" | "origin" | "onSlowDown">;

  private socket: WebSocketLike | null = null;
  private listeners = new Set<(message: TOut) => void>();
  private queue: (TIn | ArrayBufferLike | ArrayBufferView)[] = [];
  private seq = 0;
  private attempt = 0;
  private closedByUser = false;
  private throttleTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private throttled = false;
  private currentStatus: SocketStatus = "connecting";
  private lastClose: { code: number | null; wasClean: boolean } = { code: null, wasClean: false };

  /**
   * Open a socket to `path` authenticated by `ticket`.
   *
   * @param path - WebSocket path, e.g. {@link INK_PATH}.
   * @param ticket - Single-use ticket from `POST /auth/ws-ticket`.
   * @param options - Optional overrides (see {@link TypedSocketOptions}).
   */
  constructor(path: string, ticket: string, options: TypedSocketOptions = {}) {
    this.path = path;
    this.ticket = ticket;
    this.options = {
      createSocket: options.createSocket ?? ((url: string) => new WebSocket(url) as WebSocketLike),
      baseReconnectDelayMs: options.baseReconnectDelayMs ?? 250,
      maxReconnectDelayMs: options.maxReconnectDelayMs ?? 4000,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 6,
      defaultSlowDownMs: options.defaultSlowDownMs ?? 1000,
      onStatusChange: options.onStatusChange ?? (() => undefined),
      ...(options.ticketProvider === undefined ? {} : { ticketProvider: options.ticketProvider }),
      ...(options.onSlowDown === undefined ? {} : { onSlowDown: options.onSlowDown }),
      ...(options.origin === undefined ? {} : { origin: options.origin }),
    };
    this.connect();
  }

  /** Current connection state. */
  get status(): SocketStatus {
    return this.currentStatus;
  }

  /** True while a `slow_down` pause is in effect. */
  get isThrottled(): boolean {
    return this.throttled;
  }

  /** Number of messages waiting for the socket to open or the throttle to lift. */
  get pending(): number {
    return this.queue.length;
  }

  /** How many reconnects have been attempted since the last successful open. */
  get reconnectAttempts(): number {
    return this.attempt;
  }

  /**
   * Close code of the last connection, if it closed.
   *
   * {@link WS_POLICY_VIOLATION} means the ticket was refused (a fresh one is fetched on the next
   * attempt); {@link WS_TRY_AGAIN_LATER} means the server closed the socket for flooding.
   */
  get lastCloseCode(): number | null {
    return this.lastClose.code;
  }

  /**
   * Send an envelope.
   *
   * A missing `seq` is filled in with the next client sequence number. When the socket is not open
   * or a `slow_down` pause is active the envelope is queued and flushed later, in order.
   */
  send(message: TIn): void {
    const envelope = { ...message, seq: message.seq ?? ++this.seq } as TIn;
    if (this.canSendNow()) {
      this.write(envelope);
      return;
    }
    this.queue.push(envelope);
  }

  /** Send a raw binary frame (250 ms PCM16 chunks on `/ws/v1/audio`). */
  sendBinary(data: ArrayBufferLike | ArrayBufferView): void {
    if (this.canSendNow()) {
      this.write(data);
      return;
    }
    this.queue.push(data);
  }

  /**
   * Subscribe to server envelopes.
   *
   * @returns An unsubscribe function.
   */
  onMessage(cb: (message: TOut) => void): () => void {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  }

  /** Close the socket for good: no reconnect, timers cleared, queue dropped. */
  close(): void {
    this.closedByUser = true;
    this.clearTimers();
    this.queue = [];
    this.listeners.clear();
    const socket = this.socket;
    this.socket = null;
    if (socket !== null) {
      this.detach(socket);
      socket.close(1000, "client");
    }
    this.setStatus("closed");
  }

  private canSendNow(): boolean {
    return this.socket !== null && this.socket.readyState === OPEN && !this.throttled;
  }

  private write(data: TIn | ArrayBufferLike | ArrayBufferView): void {
    const socket = this.socket;
    if (socket === null) {
      return;
    }
    if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) {
      socket.send(data);
      return;
    }
    socket.send(JSON.stringify(data));
  }

  private connect(): void {
    if (this.closedByUser) {
      return;
    }
    const url = resolveWsUrl(this.path, this.ticket, this.options.origin);
    const socket = this.options.createSocket(url);
    this.socket = socket;
    socket.onopen = () => {
      this.attempt = 0;
      this.setStatus("open");
      this.flush();
    };
    socket.onmessage = (event) => {
      this.handleMessage(event.data);
    };
    socket.onclose = (event) => {
      if (this.socket === socket) {
        this.socket = null;
      }
      this.detach(socket);
      const code = typeof event?.code === "number" ? event.code : null;
      this.lastClose = { code, wasClean: event?.wasClean === true };
      // A deliberate server-side close (1000) is final; anything else is a dropped connection.
      if (code === 1000) {
        this.setStatus("closed");
        return;
      }
      this.scheduleReconnect();
    };
    socket.onerror = () => {
      // `onclose` always follows `onerror`; reconnection is handled there.
    };
  }

  private detach(socket: WebSocketLike): void {
    socket.onopen = null;
    socket.onmessage = null;
    socket.onclose = null;
    socket.onerror = null;
  }

  private handleMessage(raw: unknown): void {
    let parsed: unknown = raw;
    if (typeof raw === "string") {
      try {
        parsed = JSON.parse(raw);
      } catch {
        return; // malformed frame: drop it rather than crash the stream
      }
    }
    if (typeof parsed !== "object" || parsed === null) {
      return;
    }
    const envelope = parsed as ServerEnvelope;
    if (typeof envelope.type !== "string") {
      return;
    }
    if (envelope.type === "slow_down") {
      const payload: SlowDownPayload =
        typeof envelope.payload === "object" && envelope.payload !== null ? envelope.payload : {};
      this.applyBackpressure(payload);
    }
    for (const listener of [...this.listeners]) {
      listener(envelope as TOut);
    }
  }

  private applyBackpressure(payload: SlowDownPayload): void {
    const delay = payload.retry_after_ms ?? this.options.defaultSlowDownMs;
    this.throttled = true;
    this.options.onSlowDown?.(delay);
    if (this.throttleTimer !== null) {
      clearTimeout(this.throttleTimer);
    }
    this.throttleTimer = setTimeout(() => {
      this.throttleTimer = null;
      this.throttled = false;
      this.flush();
    }, delay);
  }

  private scheduleReconnect(): void {
    if (this.closedByUser) {
      return;
    }
    if (this.attempt >= this.options.maxReconnectAttempts) {
      this.setStatus("closed");
      return;
    }
    const delay = Math.min(
      this.options.baseReconnectDelayMs * 2 ** this.attempt,
      this.options.maxReconnectDelayMs,
    );
    this.attempt += 1;
    this.setStatus("reconnecting");
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.reconnect();
    }, delay);
  }

  private async reconnect(): Promise<void> {
    const provider = this.options.ticketProvider;
    if (provider !== undefined) {
      try {
        this.ticket = await provider();
      } catch {
        this.scheduleReconnect();
        return;
      }
      if (this.closedByUser) {
        return;
      }
    }
    this.connect();
  }

  private flush(): void {
    while (this.queue.length > 0 && this.canSendNow()) {
      const next = this.queue.shift();
      if (next !== undefined) {
        this.write(next);
      }
    }
  }

  private clearTimers(): void {
    if (this.throttleTimer !== null) {
      clearTimeout(this.throttleTimer);
      this.throttleTimer = null;
    }
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.throttled = false;
  }

  private setStatus(status: SocketStatus): void {
    if (this.currentStatus !== status) {
      this.currentStatus = status;
      this.options.onStatusChange(status);
    }
  }
}
