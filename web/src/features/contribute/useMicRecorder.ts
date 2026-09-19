/**
 * Microphone capture for the data-collection tab: permission, live level, WAV bytes.
 *
 * Three requirements shape it, all of them about not wasting a volunteer's goodwill:
 *
 * 1. **Permission is asked for once, with an explanation, before anything records.** The browser
 *    prompt is triggered by {@link MicRecorder.open}, which the UI only calls from a button whose
 *    label says what will happen. The stream then stays open, so the meter is live *before* the
 *    first take and pressing "Kaydı başlat" starts capturing in the same tick.
 * 2. **The level is visible, and clipping is caught before the upload.** `aicls.asr.qc` refuses a
 *    take whose clipped-sample ratio exceeds {@link CLIPPING_LIMIT}; the same ratio is measured
 *    here while recording, so the warning appears while the microphone is still in the volunteer's
 *    hand instead of after a round trip.
 * 3. **Nothing is lost by accident.** Capture stops at the hard 30 s contract limit rather than
 *    dying, and the encoded blob is handed back to the caller, which keeps it until the upload has
 *    actually succeeded.
 *
 * `ScriptProcessorNode` is deprecated but universal (ADR-0017 rules out `MediaRecorder`, whose
 * webm/opus output the server cannot decode); `AudioWorklet` would need a separate module file and
 * is not available in the jsdom test environment.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { t } from "@/i18n";
import { encodeWav16k, MAX_DURATION_S } from "@/lib/wavEncoder";

/** Sample block size of the capture node (about 85 ms at 48 kHz). */
const PROCESSOR_BUFFER = 4096;

/** Amplitude at or above which a sample counts as clipped (`aicls.asr.qc._CLIP_LEVEL`). */
export const CLIP_LEVEL = 0.99;

/** Clipped-sample ratio the server refuses (`QcThresholds.max_clipping_ratio`). */
export const CLIPPING_LIMIT = 0.001;

/** Peak level under which a take is probably too quiet to pass the SNR check. */
export const QUIET_PEAK = 0.05;

/** Longest take the automatic quality control accepts (`QcThresholds.max_duration_s`). */
export const QC_MAX_DURATION_S = 20;

/** Where the microphone permission stands. */
export type MicPermission = "unknown" | "granted" | "denied" | "unsupported";

/** What the recorder is doing right now. */
export type RecorderPhase = "idle" | "opening" | "ready" | "recording" | "encoding" | "error";

/** One finished take, still in the browser. */
export interface Take {
  /** 16 kHz mono PCM16 WAV bytes, ready for `POST /contribute/recordings`. */
  readonly wav: Blob;
  /** Length of the take in seconds. */
  readonly seconds: number;
  /** Share of samples at digital full scale (compare with {@link CLIPPING_LIMIT}). */
  readonly clippingRatio: number;
  /** Loudest sample of the take, in [0, 1]. */
  readonly peak: number;
}

/** What {@link useMicRecorder} hands a component. */
export interface MicRecorder {
  readonly phase: RecorderPhase;
  readonly permission: MicPermission;
  /** Smoothed input level in [0, 1]; live as soon as the microphone is open. */
  readonly level: number;
  /** Seconds captured in the current take. */
  readonly seconds: number;
  /** Clipped-sample ratio of the current take. */
  readonly clippingRatio: number;
  /** Turkish explanation of the last failure, or `null`. */
  readonly error: string | null;
  /** Ask for permission and keep the stream open; resolves to `true` when the microphone is live. */
  readonly open: () => Promise<boolean>;
  /** Begin a take (opens the microphone first when needed). */
  readonly start: () => Promise<void>;
  /** End the take and encode it; resolves to `null` when nothing was captured. */
  readonly stop: () => Promise<Take | null>;
  /** Release the microphone (the browser's recording indicator goes away). */
  readonly close: () => void;
}

interface Capture {
  stream: MediaStream;
  context: AudioContext;
  processor: ScriptProcessorNode;
  source: MediaStreamAudioSourceNode;
}

interface AudioProcessEvent {
  inputBuffer: { getChannelData: (channel: number) => Float32Array };
}

/**
 * Open the microphone, watch its level, and produce WAV takes.
 *
 * @param maxSeconds - Hard stop, defaulting to the contract's 30 s (`MAX_DURATION_S`).
 *
 * @example
 * ```tsx
 * const mic = useMicRecorder();
 * <Button onClick={() => void mic.open()}>{t("contribute.micAllow")}</Button>
 * ```
 */
export function useMicRecorder(maxSeconds: number = MAX_DURATION_S): MicRecorder {
  const [phase, setPhase] = useState<RecorderPhase>("idle");
  const [permission, setPermission] = useState<MicPermission>("unknown");
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [clippingRatio, setClippingRatio] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const captureRef = useRef<Capture | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const recordingRef = useRef(false);
  const clippedRef = useRef(0);
  const peakRef = useRef(0);
  const sampleRateRef = useRef(48_000);
  const stopRef = useRef<() => Promise<Take | null>>(() => Promise.resolve(null));

  const close = useCallback((): void => {
    const capture = captureRef.current;
    captureRef.current = null;
    recordingRef.current = false;
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
    setLevel(0);
  }, []);

  // Release the microphone when the tab is left: the browser's "recording" indicator staying on
  // after a lecturer navigates away would be alarming and is never intended.
  useEffect(() => close, [close]);

  const open = useCallback(async (): Promise<boolean> => {
    if (captureRef.current !== null) {
      return true;
    }
    setError(null);
    const media = globalThis.navigator?.mediaDevices;
    const AudioCtor = (globalThis as unknown as { AudioContext?: typeof AudioContext })
      .AudioContext;
    if (media === undefined || AudioCtor === undefined) {
      setPermission("unsupported");
      setError(t("voice.unsupported"));
      setPhase("error");
      return false;
    }
    setPhase("opening");
    let stream: MediaStream;
    try {
      stream = await media.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
    } catch {
      setPermission("denied");
      setError(t("voice.permissionDenied"));
      setPhase("error");
      return false;
    }
    const context = new AudioCtor();
    sampleRateRef.current = context.sampleRate;
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(PROCESSOR_BUFFER, 1, 1);

    processor.onaudioprocess = (event: AudioProcessEvent): void => {
      const input = event.inputBuffer.getChannelData(0);
      let peak = 0;
      let clipped = 0;
      for (const sample of input) {
        const magnitude = Math.abs(sample);
        if (magnitude > peak) {
          peak = magnitude;
        }
        if (magnitude >= CLIP_LEVEL) {
          clipped += 1;
        }
      }
      // Fall towards silence rather than snapping to it: a bar that only ever jumps is unreadable
      // from the back of a room.
      setLevel((previous) => (peak > previous ? peak : previous * 0.75 + peak * 0.25));
      if (!recordingRef.current) {
        return;
      }
      chunksRef.current.push(new Float32Array(input));
      clippedRef.current += clipped;
      peakRef.current = Math.max(peakRef.current, peak);
      let captured = 0;
      for (const chunk of chunksRef.current) {
        captured += chunk.length;
      }
      setSeconds(captured / context.sampleRate);
      setClippingRatio(captured === 0 ? 0 : clippedRef.current / captured);
      if (captured / context.sampleRate >= maxSeconds) {
        void stopRef.current();
      }
    };
    source.connect(processor);
    processor.connect(context.destination);
    captureRef.current = { stream, context, processor, source };
    setPermission("granted");
    setPhase("ready");
    return true;
  }, [maxSeconds]);

  const start = useCallback(async (): Promise<void> => {
    const opened = captureRef.current !== null || (await open());
    if (!opened) {
      return;
    }
    chunksRef.current = [];
    clippedRef.current = 0;
    peakRef.current = 0;
    recordingRef.current = true;
    setSeconds(0);
    setClippingRatio(0);
    setError(null);
    setPhase("recording");
  }, [open]);

  const stop = useCallback(async (): Promise<Take | null> => {
    if (!recordingRef.current) {
      return null;
    }
    recordingRef.current = false;
    const chunks = chunksRef.current;
    chunksRef.current = [];
    let total = 0;
    for (const chunk of chunks) {
      total += chunk.length;
    }
    if (total === 0) {
      setPhase("ready");
      return null;
    }
    setPhase("encoding");
    try {
      const wav = await encodeWav16k(chunks, sampleRateRef.current);
      setPhase("ready");
      return {
        wav,
        seconds: total / sampleRateRef.current,
        clippingRatio: clippedRef.current / total,
        peak: peakRef.current,
      };
    } catch {
      setError(t("errors.unknown"));
      setPhase("error");
      return null;
    }
  }, []);

  // The capture callback stops the take itself at the hard limit; it reads the latest `stop`
  // through this ref so the closure never captures a stale one. Written in an effect rather than
  // during render, so rendering stays free of side effects.
  useEffect(() => {
    stopRef.current = stop;
  }, [stop]);

  return {
    phase,
    permission,
    level,
    seconds,
    clippingRatio,
    error,
    open,
    start,
    stop,
    close,
  };
}
