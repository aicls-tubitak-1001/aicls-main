/**
 * Browser-side WAV encoder: microphone Float32 chunks → 16 kHz mono PCM16 RIFF/WAVE bytes.
 *
 * ADR-0017 makes the browser responsible for the audio format: the server has no ffmpeg and no
 * torchaudio, accepts WAV only, and both quality control and training see exactly the same bytes.
 * Resampling uses `OfflineAudioContext` where available (the browser's own high-quality resampler)
 * and falls back to a Hann-windowed sinc kernel — the same technique the server uses with numpy —
 * so headless environments and older browsers still produce a valid file.
 *
 * The emitted header is the canonical 44-byte RIFF header: `RIFF`, `WAVE`, a 16-byte `fmt ` chunk
 * (PCM, 1 channel, 16 000 Hz, 32 000 B/s, block align 2, 16 bits) and a `data` chunk.
 */

/** Target sample rate for every uploaded recording (ADR-0017). */
export const TARGET_SAMPLE_RATE = 16_000;

/** Bytes per sample of the target format (PCM16). */
export const BYTES_PER_SAMPLE = 2;

/** Size of the canonical RIFF/WAVE header this module writes. */
export const WAV_HEADER_BYTES = 44;

/** Contract limit: recordings longer than this are rejected (api.md §1). */
export const MAX_DURATION_S = 30;

/** Sample rates a browser is allowed to hand us (ADR-0017). */
const MIN_INPUT_RATE = 4_000;
const MAX_INPUT_RATE = 192_000;

/** Raised when the caller passes an unusable sample rate. */
export class AudioFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AudioFormatError";
  }
}

/** Concatenate the recorder's Float32 chunks into a single mono buffer. */
export function concatChunks(chunks: readonly Float32Array[]): Float32Array {
  let total = 0;
  for (const chunk of chunks) {
    total += chunk.length;
  }
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function sinc(x: number): number {
  if (x === 0) {
    return 1;
  }
  const piX = Math.PI * x;
  return Math.sin(piX) / piX;
}

/**
 * Resample with a Hann-windowed sinc kernel (band-limited, DC-preserving).
 *
 * Used when `OfflineAudioContext` is unavailable (tests, headless browsers). The kernel support
 * widens when downsampling so the low-pass cutoff sits at the output Nyquist frequency.
 *
 * @param input - Mono samples in [-1, 1].
 * @param inputRate - Source sample rate.
 * @param outputRate - Destination sample rate.
 * @param taps - Kernel half-width in output periods (16 is transparent for speech).
 */
export function resampleSinc(
  input: Float32Array,
  inputRate: number,
  outputRate: number,
  taps = 16,
): Float32Array {
  if (inputRate === outputRate || input.length === 0) {
    return input.slice();
  }
  const ratio = outputRate / inputRate;
  const cutoff = Math.min(1, ratio);
  const half = taps / cutoff;
  const outLength = Math.max(1, Math.round(input.length * ratio));
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i += 1) {
    const center = i / ratio;
    const first = Math.ceil(center - half);
    const last = Math.floor(center + half);
    let acc = 0;
    let weight = 0;
    for (let j = first; j <= last; j += 1) {
      const distance = j - center;
      const window = 0.5 * (1 + Math.cos((Math.PI * distance) / half));
      const value = sinc(distance * cutoff) * window;
      weight += value;
      if (j >= 0 && j < input.length) {
        acc += (input[j] as number) * value;
      }
    }
    out[i] = weight === 0 ? 0 : acc / weight;
  }
  return out;
}

async function resampleWithAudioContext(
  input: Float32Array,
  inputRate: number,
): Promise<Float32Array | null> {
  const Ctor = (globalThis as unknown as { OfflineAudioContext?: typeof OfflineAudioContext })
    .OfflineAudioContext;
  if (Ctor === undefined || input.length === 0) {
    return null;
  }
  const outLength = Math.max(1, Math.round((input.length * TARGET_SAMPLE_RATE) / inputRate));
  const context = new Ctor(1, outLength, TARGET_SAMPLE_RATE);
  const buffer = context.createBuffer(1, input.length, inputRate);
  // `copyToChannel` wants a Float32Array backed by a plain ArrayBuffer; copy to guarantee that.
  buffer.copyToChannel(new Float32Array(input), 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start();
  const rendered = await context.startRendering();
  return rendered.getChannelData(0).slice();
}

/** Write an ASCII tag into `view` at `offset` (RIFF chunk identifiers are ASCII). */
function writeTag(view: DataView, offset: number, tag: string): void {
  for (let i = 0; i < tag.length; i += 1) {
    view.setUint8(offset + i, tag.charCodeAt(i));
  }
}

/**
 * Wrap mono float samples in a canonical 44-byte RIFF/WAVE PCM16 container.
 *
 * @param samples - Mono samples in [-1, 1]; values outside the range are clipped.
 * @param sampleRate - Sample rate written into the header.
 * @returns The complete file bytes (header + data).
 */
export function encodeWavBytes(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const dataBytes = samples.length * BYTES_PER_SAMPLE;
  const bytes = new ArrayBuffer(WAV_HEADER_BYTES + dataBytes);
  const view = new DataView(bytes);
  writeTag(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true); // file size - 8
  writeTag(view, 8, "WAVE");
  writeTag(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // audioFormat = PCM
  view.setUint16(22, 1, true); // numChannels = mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * BYTES_PER_SAMPLE, true); // byteRate
  view.setUint16(32, BYTES_PER_SAMPLE, true); // blockAlign
  view.setUint16(34, 8 * BYTES_PER_SAMPLE, true); // bitsPerSample
  writeTag(view, 36, "data");
  view.setUint32(40, dataBytes, true);
  let offset = WAV_HEADER_BYTES;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i] as number));
    // Asymmetric scaling keeps -1.0 at exactly -32768 and +1.0 at +32767.
    view.setInt16(offset, Math.round(clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff), true);
    offset += BYTES_PER_SAMPLE;
  }
  return bytes;
}

/**
 * Encode recorded microphone chunks as a 16 kHz mono PCM16 WAV blob.
 *
 * @param chunks - Float32 mono chunks in capture order (as delivered by the audio worklet).
 * @param inputRate - The `AudioContext.sampleRate` the chunks were captured at.
 * @returns A `Blob` of type `audio/wav` ready for `POST /contribute/recordings` or `/asr/*`.
 * @throws {AudioFormatError} When `inputRate` is not a plausible capture rate.
 */
export async function encodeWav16k(chunks: Float32Array[], inputRate: number): Promise<Blob> {
  if (!Number.isFinite(inputRate) || inputRate < MIN_INPUT_RATE || inputRate > MAX_INPUT_RATE) {
    throw new AudioFormatError(`Unusable capture sample rate: ${String(inputRate)} Hz.`);
  }
  const mono = concatChunks(chunks);
  let samples: Float32Array;
  if (inputRate === TARGET_SAMPLE_RATE) {
    samples = mono;
  } else {
    samples =
      (await resampleWithAudioContext(mono, inputRate)) ??
      resampleSinc(mono, inputRate, TARGET_SAMPLE_RATE);
  }
  return new Blob([encodeWavBytes(samples, TARGET_SAMPLE_RATE)], { type: "audio/wav" });
}

/** Duration in seconds of a mono buffer at the given rate (for the 30 s guard). */
export function durationSeconds(sampleCount: number, sampleRate = TARGET_SAMPLE_RATE): number {
  return sampleCount / sampleRate;
}
