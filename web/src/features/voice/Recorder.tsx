/**
 * Microphone recorder producing 16 kHz mono PCM16 WAV.
 *
 * ADR-0017 rules out `MediaRecorder`: it emits webm/opus or mp4/aac, which the server cannot decode
 * without ffmpeg. Instead the raw Float32 stream is tapped with a `ScriptProcessorNode` — deprecated
 * but universally available, including Safari — buffered, and handed to `encodeWav16k`, which
 * resamples with `OfflineAudioContext` and writes the RIFF header.
 *
 * The component enforces the contract limits locally (≤ 30 s) so a bad take is not uploaded at all.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { t } from "@/i18n";
import { encodeWav16k, MAX_DURATION_S } from "@/lib/wavEncoder";
import styles from "./Recorder.module.css";

/** What the recorder is doing right now. */
export type RecorderPhase = "idle" | "recording" | "encoding" | "error";

export interface RecorderProps {
  /** Receives the encoded WAV once the user stops (or the limit is hit). */
  onRecorded: (wav: Blob, durationSeconds: number) => void;
  /** Hard limit in seconds; defaults to the contract's 30 s. */
  maxSeconds?: number;
  /** Disable the control (e.g. while an upload is in flight). */
  disabled?: boolean;
  /** Label of the start button; defaults to `voice.start`. */
  startLabel?: string;
}

const PROCESSOR_BUFFER = 4096;

interface Capture {
  stream: MediaStream;
  context: AudioContext;
  processor: ScriptProcessorNode;
  source: MediaStreamAudioSourceNode;
}

/** Start/stop button with a live level meter. */
export function Recorder({
  onRecorded,
  maxSeconds = MAX_DURATION_S,
  disabled = false,
  startLabel,
}: RecorderProps): JSX.Element {
  const [phase, setPhase] = useState<RecorderPhase>("idle");
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [errorText, setErrorText] = useState<string | null>(null);
  const captureRef = useRef<Capture | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const sampleRateRef = useRef(48_000);

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

  const stop = useCallback(async (): Promise<void> => {
    if (captureRef.current === null) {
      return;
    }
    teardown();
    const chunks = chunksRef.current;
    chunksRef.current = [];
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    if (total === 0) {
      setPhase("idle");
      return;
    }
    setPhase("encoding");
    try {
      const wav = await encodeWav16k(chunks, sampleRateRef.current);
      onRecorded(wav, total / sampleRateRef.current);
      setPhase("idle");
    } catch {
      setErrorText(t("errors.unknown"));
      setPhase("error");
    }
  }, [onRecorded, teardown]);

  const start = useCallback(async (): Promise<void> => {
    setErrorText(null);
    const media = globalThis.navigator?.mediaDevices;
    const AudioCtor = (globalThis as unknown as { AudioContext?: typeof AudioContext })
      .AudioContext;
    if (media === undefined || AudioCtor === undefined) {
      setErrorText(t("voice.unsupported"));
      setPhase("error");
      return;
    }
    let stream: MediaStream;
    try {
      stream = await media.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
    } catch {
      setErrorText(t("voice.permissionDenied"));
      setPhase("error");
      return;
    }
    const context = new AudioCtor();
    sampleRateRef.current = context.sampleRate;
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(PROCESSOR_BUFFER, 1, 1);
    chunksRef.current = [];
    setSeconds(0);
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
      if (elapsed >= maxSeconds) {
        void stop();
      }
    };
    source.connect(processor);
    processor.connect(context.destination);
    captureRef.current = { stream, context, processor, source };
    setPhase("recording");
  }, [maxSeconds, stop]);

  const recording = phase === "recording";
  return (
    <div className={styles.recorder}>
      <div className={styles.row}>
        <Button
          variant={recording ? "danger" : "primary"}
          disabled={disabled || phase === "encoding"}
          onClick={() => {
            void (recording ? stop() : start());
          }}
        >
          {recording ? t("voice.stop") : (startLabel ?? t("voice.start"))}
        </Button>
        <span className={styles.level} aria-hidden="true">
          <span
            className={styles.levelFill}
            style={{ width: `${String(Math.round(level * 100))}%` }}
          />
        </span>
        <span className={styles.status} role="status">
          {phase === "encoding"
            ? t("voice.processing")
            : recording
              ? `${t("voice.recording")} ${t("voice.duration", { seconds })}`
              : t("voice.maxDuration")}
        </span>
      </div>
      {errorText === null ? null : (
        <p className={styles.error} role="alert">
          {errorText}
        </p>
      )}
    </div>
  );
}
