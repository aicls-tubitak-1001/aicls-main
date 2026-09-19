/**
 * React bindings for the two streams: {@link useInkSocket} and {@link useAudioSocket}.
 *
 * Both own the whole lifetime of one {@link TypedSocket}: they fetch a ticket, open the socket,
 * translate envelopes into state, and close cleanly on unmount — an abandoned socket would keep a
 * recogniser slot busy on the server for the rest of the lecture.
 *
 * Errors are state, not exceptions: an `error` envelope becomes a parsed {@link AiclsProblem} with
 * a Turkish `detail_tr` that the surrounding panel can show without unmounting the last result.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { t } from "@/i18n";
import { TARGET_SAMPLE_RATE } from "@/lib/wavEncoder";
import { parseProblem, type AiclsProblem } from "./problem";
import type { ComputeOut } from "./types";
import {
  AUDIO_PATH,
  INK_PATH,
  TypedSocket,
  type AudioClientMessage,
  type AudioServerMessage,
  type InkCanvasPayload,
  type InkClientMessage,
  type InkPointPayload,
  type InkServerMessage,
  type SocketStatus,
  type WebSocketLike,
} from "./ws";

export { TARGET_SAMPLE_RATE };

/** Options shared by both hooks. */
export interface UseSocketOptions {
  /** Open the connection (false while the user is not on the page that needs it). */
  enabled?: boolean;
  /** Socket factory override (tests). */
  createSocket?: (url: string) => WebSocketLike;
  /** Ticket source; defaults to `POST /auth/ws-ticket` via `app/auth.ts`. */
  getTicket?: () => Promise<string>;
}

/** The ticket source is injected lazily so `api/` does not depend on `app/`. */
type TicketSource = () => Promise<string>;

let defaultTicketSource: TicketSource = () => Promise.reject(new Error(t("ws.ticketFailed")));

/** Register the function both hooks use when the caller passes no `getTicket` (`app/auth.ts`). */
export function setDefaultTicketSource(source: TicketSource): void {
  defaultTicketSource = source;
}

/**
 * Turn an `error` envelope payload (or a thrown ticket failure) into a renderable problem.
 *
 * The server wraps the RFC 9457 document in `{problem: …}` (`aicls.api.ws.protocol.error_envelope`);
 * anything else is passed through so a transport failure still produces Turkish text.
 */
function problemOf(payload: unknown): AiclsProblem {
  const document =
    typeof payload === "object" && payload !== null && "problem" in payload
      ? payload.problem
      : payload;
  return parseProblem(document, { fallbackDetailTr: t("ws.errorFallback") });
}

/* --------------------------------------------------------------------------------- ink --- */

/** What {@link useInkSocket} exposes to the workspace. */
export interface InkSocketState {
  status: SocketStatus;
  /** Latest partial recognition, if any. */
  partialLatex: string | null;
  /** Latest final recognition, if any. */
  finalLatex: string | null;
  /** Alternatives of the last `final`, best first. */
  alternatives: [string, number][];
  /** Confidence of the last recognition (`0..1`). */
  confidence: number | null;
  /** Recognition id of the last `final` (used by `POST /ocr/feedback`). */
  recognitionId: string | null;
  /** True while the server has asked for a slower rate. */
  throttled: boolean;
  /** The last `error` envelope, already parsed; cleared by {@link InkSocketState.clear}. */
  problem: AiclsProblem | null;
  /** Send one completed trace. */
  sendStroke: (stroke: { points: InkPointPayload[] }, canvas: InkCanvasPayload) => void;
  /** Tell the server the pen was lifted (starts the 400 ms debounce). */
  penUp: () => void;
  /** Ask for the final result now. */
  finalize: () => void;
  /** Drop the server-side buffer and the local recognition state. */
  clear: () => void;
}

export type UseInkSocketOptions = UseSocketOptions;

/**
 * Open (and own the lifetime of) the ink WebSocket.
 *
 * The stream exists so recognition overlaps writing: every pen-up ships the new trace, the server
 * debounces 400 ms and answers `partial`, and `finalize` (or 1.5 s idle) produces `final`. That
 * overlap is what keeps the contractual "command → result < 3 s average" reachable (ADR-0014).
 */
export function useInkSocket(options: UseInkSocketOptions = {}): InkSocketState {
  const { enabled = true, createSocket, getTicket } = options;
  // The two callbacks live in refs so an inline lambda from the caller cannot retrigger the
  // effect on every render — that would tear the connection down and reconnect forever.
  const createSocketRef = useRef(createSocket);
  createSocketRef.current = createSocket;
  const getTicketRef = useRef(getTicket);
  getTicketRef.current = getTicket;
  const socketRef = useRef<TypedSocket<InkClientMessage, InkServerMessage> | null>(null);
  const [status, setStatus] = useState<SocketStatus>("closed");
  const [partialLatex, setPartialLatex] = useState<string | null>(null);
  const [finalLatex, setFinalLatex] = useState<string | null>(null);
  const [alternatives, setAlternatives] = useState<[string, number][]>([]);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [recognitionId, setRecognitionId] = useState<string | null>(null);
  const [throttled, setThrottled] = useState(false);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    let disposed = false;
    setStatus("connecting");
    void (async () => {
      let ticket: string;
      try {
        ticket = await (getTicketRef.current ?? defaultTicketSource)();
      } catch (cause) {
        if (!disposed) {
          setStatus("closed");
          setProblem(problemOf(cause));
        }
        return;
      }
      if (disposed) {
        return;
      }
      const socket = new TypedSocket<InkClientMessage, InkServerMessage>(INK_PATH, ticket, {
        ticketProvider: () => (getTicketRef.current ?? defaultTicketSource)(),
        onStatusChange: setStatus,
        onSlowDown: () => {
          setThrottled(true);
        },
        ...(createSocketRef.current === undefined ? {} : { createSocket: createSocketRef.current }),
      });
      socket.onMessage((message) => {
        switch (message.type) {
          case "partial":
            setPartialLatex(message.payload.latex);
            setConfidence(message.payload.confidence);
            break;
          case "final":
            setFinalLatex(message.payload.latex);
            setAlternatives(message.payload.alternatives);
            setConfidence(message.payload.confidence);
            setRecognitionId(message.payload.recognition_id);
            setPartialLatex(null);
            break;
          case "error":
            setProblem(problemOf(message.payload));
            break;
          case "slow_down":
            setThrottled(true);
            break;
          default:
            break;
        }
      });
      socketRef.current = socket;
    })();
    return () => {
      disposed = true;
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [enabled]);

  const send = useCallback((message: InkClientMessage): void => {
    const socket = socketRef.current;
    if (socket === null) {
      return;
    }
    socket.send(message);
    setThrottled(socket.isThrottled);
  }, []);

  const sendStroke = useCallback(
    (stroke: { points: InkPointPayload[] }, canvas: InkCanvasPayload) => {
      send({ type: "stroke", payload: { points: stroke.points, canvas } });
    },
    [send],
  );

  const penUp = useCallback(() => {
    send({ type: "pen_up", payload: {} });
  }, [send]);

  const finalize = useCallback(() => {
    send({ type: "finalize", payload: {} });
  }, [send]);

  const clear = useCallback(() => {
    setPartialLatex(null);
    setFinalLatex(null);
    setAlternatives([]);
    setConfidence(null);
    setRecognitionId(null);
    setProblem(null);
    send({ type: "clear", payload: {} });
  }, [send]);

  return {
    status,
    partialLatex,
    finalLatex,
    alternatives,
    confidence,
    recognitionId,
    throttled,
    problem,
    sendStroke,
    penUp,
    finalize,
    clear,
  };
}

/* ------------------------------------------------------------------------------- audio --- */

/** What {@link useAudioSocket} exposes to the voice UI. */
export interface AudioSocketState {
  status: SocketStatus;
  /** Partial transcript (only for recordings longer than 5 s, api.md §4.2). */
  partialText: string | null;
  /** Final transcript. */
  finalText: string | null;
  /** ASR confidence of the final transcript. */
  confidence: number | null;
  /** Id of the last transcript (used by `POST /asr/feedback`). */
  transcriptId: string | null;
  /** The compute outcome the server produced from the transcript, awaiting confirmation. */
  outcome: ComputeOut | null;
  throttled: boolean;
  /** The last `error` envelope, already parsed. */
  problem: AiclsProblem | null;
  /** Announce the format and begin an utterance. */
  start: () => void;
  /** Push one 250 ms PCM16 frame. */
  sendFrame: (frame: ArrayBuffer) => void;
  /** End the utterance and ask for the transcript. */
  stop: () => void;
  /** Abandon the utterance. */
  cancel: () => void;
}

export type UseAudioSocketOptions = UseSocketOptions;

/**
 * Open (and own the lifetime of) the audio WebSocket.
 *
 * The client announces `start{sample_rate, format:"pcm16"}` and pushes 250 ms binary PCM16 frames.
 * The server ends the utterance on 600 ms of silence or on `stop`, answers `final`, then runs NLU
 * and the CAS on the same connection and delivers a `command` envelope carrying a full
 * `ComputeOut`. Because the input is voice, the UI always asks for confirmation before the result
 * is treated as final (api.md §3.6) — this hook surfaces the outcome without committing it.
 */
export function useAudioSocket(options: UseAudioSocketOptions = {}): AudioSocketState {
  const { enabled = true, createSocket, getTicket } = options;
  const createSocketRef = useRef(createSocket);
  createSocketRef.current = createSocket;
  const getTicketRef = useRef(getTicket);
  getTicketRef.current = getTicket;
  const socketRef = useRef<TypedSocket<AudioClientMessage, AudioServerMessage> | null>(null);
  const [status, setStatus] = useState<SocketStatus>("closed");
  const [partialText, setPartialText] = useState<string | null>(null);
  const [finalText, setFinalText] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [transcriptId, setTranscriptId] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<ComputeOut | null>(null);
  const [throttled, setThrottled] = useState(false);
  const [problem, setProblem] = useState<AiclsProblem | null>(null);

  useEffect(() => {
    if (!enabled) {
      return undefined;
    }
    let disposed = false;
    setStatus("connecting");
    void (async () => {
      let ticket: string;
      try {
        ticket = await (getTicketRef.current ?? defaultTicketSource)();
      } catch (cause) {
        if (!disposed) {
          setStatus("closed");
          setProblem(problemOf(cause));
        }
        return;
      }
      if (disposed) {
        return;
      }
      const socket = new TypedSocket<AudioClientMessage, AudioServerMessage>(AUDIO_PATH, ticket, {
        ticketProvider: () => (getTicketRef.current ?? defaultTicketSource)(),
        onStatusChange: setStatus,
        onSlowDown: () => {
          setThrottled(true);
        },
        ...(createSocketRef.current === undefined ? {} : { createSocket: createSocketRef.current }),
      });
      socket.onMessage((message) => {
        switch (message.type) {
          case "partial":
            setPartialText(message.payload.text);
            break;
          case "final":
            setFinalText(message.payload.text);
            setConfidence(message.payload.confidence);
            setTranscriptId(message.payload.transcript_id);
            setPartialText(null);
            break;
          case "command":
            setOutcome(message.payload.outcome);
            break;
          case "error":
            setProblem(problemOf(message.payload));
            break;
          case "slow_down":
            setThrottled(true);
            break;
          default:
            break;
        }
      });
      socketRef.current = socket;
    })();
    return () => {
      disposed = true;
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [enabled]);

  const start = useCallback(() => {
    setPartialText(null);
    setFinalText(null);
    setOutcome(null);
    setProblem(null);
    socketRef.current?.send({
      type: "start",
      payload: { sample_rate: TARGET_SAMPLE_RATE, format: "pcm16" },
    });
  }, []);

  const sendFrame = useCallback((frame: ArrayBuffer) => {
    const socket = socketRef.current;
    if (socket === null) {
      return;
    }
    socket.sendBinary(frame);
    setThrottled(socket.isThrottled);
  }, []);

  const stop = useCallback(() => {
    socketRef.current?.send({ type: "stop", payload: {} });
  }, []);

  const cancel = useCallback(() => {
    setPartialText(null);
    setFinalText(null);
    setOutcome(null);
    socketRef.current?.send({ type: "cancel", payload: {} });
  }, []);

  return {
    status,
    partialText,
    finalText,
    confidence,
    transcriptId,
    outcome,
    throttled,
    problem,
    start,
    sendFrame,
    stop,
    cancel,
  };
}
