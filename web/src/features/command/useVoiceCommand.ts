/**
 * The voice path: microphone → 16 kHz mono PCM16 WAV → `POST /asr/command` → `ComputeOut`.
 *
 * ADR-0017 rules out `MediaRecorder` (webm/opus, which the server cannot decode without ffmpeg), so
 * the raw Float32 stream is tapped with a `ScriptProcessorNode` — deprecated but present in every
 * browser the project targets, Safari included — and handed to `encodeWav16k`.
 *
 * Two behaviours matter in a lecture theatre:
 *
 * - **Hold or toggle.** The caller decides; this hook only exposes `start`, `stop`, `cancel` and
 *   `toggle`, so a press-and-hold on the microphone button and a tap-to-latch produce the same
 *   recording. `cancel` throws the take away without asking the server anything.
 * - **The clock starts when the lecturer stops speaking.** `markSubmit` runs at `stop`, because
 *   that is the moment a spoken command has been "entered"; the key travels with the outcome so
 *   whichever panel renders the answer can close the measurement (ADR-0014).
 *
 * Failures are values, never exceptions: a refused microphone, a browser without `AudioContext`,
 * an over-long take and an API error all arrive as a Turkish {@link AiclsProblem}.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { api, call, filePart, formDataSerializer } from "@/api/client";
import { transportProblem, type AiclsProblem } from "@/api/problem";
import type { ComputeOut } from "@/api/types";
import { t } from "@/i18n";
import { markSubmit } from "@/lib/latency";
import { encodeWav16k, MAX_DURATION_S } from "@/lib/wavEncoder";

/** What the microphone is doing right now. */
export type VoicePhase = "idle" | "recording" | "encoding" | "uploading";

/** Options for {@link useVoiceCommand}. */
export interface UseVoiceCommandOptions {
  /** Workspace the resulting operation belongs to. */
  workspaceId?: string | null;
  /** Expression on the board, sent as `last_latex` so "türevini al" resolves. */
  lastLatex?: string | null;
  /** Called with the backend's answer and the latency key opened at `stop`. */
  onOutcome: (outcome: ComputeOut, latencyKey: string) => void;
  /** Called when the take never became an answer. */
  onProblem?: (problem: AiclsProblem) => void;
  /** Hard limit in seconds; defaults to the contract's 30 s (ADR-0017). */
  maxSeconds?: number;
}

/** What {@link useVoiceCommand} hands the command bar. */
export interface VoiceCommandApi {
  phase: VoicePhase;
  /** Whole seconds captured so far. */
  seconds: number;
  /** Peak level of the last audio block in `[0, 1]`, for the meter. */
  level: number;
  /** False when the browser exposes no microphone API at all. */
  supported: boolean;
  /** True while recording or sending — the microphone button reads this. */
  active: boolean;
  start: () => void;
  /** Stop, encode and send. */
  stop: () => void;
  /** Abandon the take without sending it. */
  cancel: () => void;
  /** Start if idle, stop if recording. */
  toggle: () => void;
}

const PROCESSOR_BUFFER = 4096;

interface Capture {
  stream: MediaStream;
  context: AudioContext;
  processor: ScriptProcessorNode;
  source: MediaStreamAudioSourceNode;
}

let voiceCounter = 0;

function nextVoiceKey(): string {
  voiceCounter += 1;
  return `voice-${String(voiceCounter)}-${String(Date.now())}`;
}

/** Is a microphone reachable in this browser at all? */
export function microphoneSupported(): boolean {
  const media = globalThis.navigator?.mediaDevices;
  const audioContext = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext;
  return (
    media !== undefined && typeof media.getUserMedia === "function" && audioContext !== undefined
  );
}

/**
 * Record a spoken command and turn it into a `ComputeOut`.
 *
 * @example
 * ```ts
 * const voice = useVoiceCommand({ workspaceId, onOutcome: (out, key) => preview(out, key) });
 * <button onPointerDown={voice.start} onPointerUp={voice.stop}>Basılı tutup konuşun</button>
 * ```
 */
export function useVoiceCommand(options: UseVoiceCommandOptions): VoiceCommandApi {
  const { workspaceId = null, lastLatex = null, onOutcome, onProblem, maxSeconds } = options;
  const limit = maxSeconds ?? MAX_DURATION_S;

  const [phase, setPhase] = useState<VoicePhase>("idle");
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);

  const captureRef = useRef<Capture | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const rateRef = useRef(48_000);
  const abandonRef = useRef(false);
  // The callbacks live in refs so a caller may pass inline lambdas without re-creating `stop`
  // on every render — the microphone must not restart because the parent re-rendered.
  const onOutcomeRef = useRef(onOutcome);
  onOutcomeRef.current = onOutcome;
  const onProblemRef = useRef(onProblem);
  onProblemRef.current = onProblem;
  const contextRef = useRef({ workspaceId, lastLatex });
  contextRef.current = { workspaceId, lastLatex };

  const teardown = useCallback((): void => {
    const capture = captureRef.current;
    captureRef.current = null;
    if (capture === null) {
      return;
    }
    capture.processor.onaudioprocess = null;
    capture.processor.disconnect();
    capture.source.disconnect();
    for (const track of capture.stream.getTracks()) {
      track.stop();
    }
    void capture.context.close();
  }, []);

  useEffect(() => teardown, [teardown]);

  const send = useCallback(async (wav: Blob): Promise<void> => {
    const key = nextVoiceKey();
    markSubmit(key, "audio");
    setPhase("uploading");
    const board = contextRef.current.lastLatex;
    const workspace = contextRef.current.workspaceId;
    const { data, problem } = await call(
      api.POST("/asr/command", {
        body: {
          audio: filePart(wav, "komut.wav"),
          ...(workspace === null || workspace === "" ? {} : { workspace_id: workspace }),
          ...(board === null || board === "" ? {} : { last_latex: board }),
        },
        bodySerializer: formDataSerializer,
      }),
      { fallbackDetailTr: t("errors.network") },
    );
    setPhase("idle");
    if (data === null) {
      onProblemRef.current?.(problem ?? transportProblem(t("errors.network")));
      return;
    }
    onOutcomeRef.current(data, key);
  }, []);

  const finish = useCallback(async (): Promise<void> => {
    const chunks = chunksRef.current;
    chunksRef.current = [];
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (total === 0 || abandonRef.current) {
      setPhase("idle");
      return;
    }
    setPhase("encoding");
    try {
      const wav = await encodeWav16k(chunks, rateRef.current);
      await send(wav);
    } catch (cause) {
      setPhase("idle");
      onProblemRef.current?.(transportProblem(t("errors.unknown"), cause));
    }
  }, [send]);

  const stop = useCallback((): void => {
    if (captureRef.current === null) {
      return;
    }
    teardown();
    void finish();
  }, [finish, teardown]);

  const cancel = useCallback((): void => {
    abandonRef.current = true;
    teardown();
    chunksRef.current = [];
    setSeconds(0);
    setLevel(0);
    setPhase("idle");
  }, [teardown]);

  const start = useCallback((): void => {
    if (captureRef.current !== null || phase !== "idle") {
      return;
    }
    const media = globalThis.navigator?.mediaDevices;
    const AudioCtor = (globalThis as { AudioContext?: typeof AudioContext }).AudioContext;
    if (media === undefined || AudioCtor === undefined) {
      onProblemRef.current?.(transportProblem(t("voice.unsupported")));
      return;
    }
    abandonRef.current = false;
    chunksRef.current = [];
    setSeconds(0);
    setLevel(0);
    setPhase("recording");
    void (async () => {
      let stream: MediaStream;
      try {
        stream = await media.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
      } catch (cause) {
        setPhase("idle");
        onProblemRef.current?.(transportProblem(t("voice.permissionDenied"), cause));
        return;
      }
      if (abandonRef.current) {
        for (const track of stream.getTracks()) {
          track.stop();
        }
        return;
      }
      const context = new AudioCtor();
      rateRef.current = context.sampleRate;
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(PROCESSOR_BUFFER, 1, 1);
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
        let peak = 0;
        for (const sample of input) {
          const magnitude = Math.abs(sample);
          if (magnitude > peak) {
            peak = magnitude;
          }
        }
        setLevel(peak);
        const captured = chunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
        const elapsed = captured / context.sampleRate;
        setSeconds(Math.floor(elapsed));
        if (elapsed >= limit) {
          stop();
        }
      };
      source.connect(processor);
      processor.connect(context.destination);
      captureRef.current = { stream, context, processor, source };
    })();
  }, [limit, phase, stop]);

  const toggle = useCallback((): void => {
    if (captureRef.current === null) {
      start();
      return;
    }
    stop();
  }, [start, stop]);

  return {
    phase,
    seconds,
    level,
    supported: microphoneSupported(),
    active: phase !== "idle",
    start,
    stop,
    cancel,
    toggle,
  };
}
