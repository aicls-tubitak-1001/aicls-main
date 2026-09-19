/**
 * `wavEncoder.test.ts` — contractual test from `docs/mimari/moduller.md#web--sahibi-sg`:
 * the encoder emits a canonical 16 kHz mono PCM16 RIFF/WAVE header with correct sizes (ADR-0017).
 *
 * jsdom has no `OfflineAudioContext`, so these tests exercise the windowed-sinc fallback path — the
 * one that must also work in an old browser. The header is byte-checked field by field because the
 * server validates exactly these bytes before it will accept a recording.
 */
import { describe, expect, it } from "vitest";

import {
  BYTES_PER_SAMPLE,
  concatChunks,
  durationSeconds,
  encodeWav16k,
  encodeWavBytes,
  resampleSinc,
  TARGET_SAMPLE_RATE,
  WAV_HEADER_BYTES,
  AudioFormatError,
} from "@/lib/wavEncoder";

/** One second of a 440 Hz sine at the given rate. */
function tone(rate: number, seconds = 0.25, frequency = 440): Float32Array {
  const samples = new Float32Array(Math.round(rate * seconds));
  for (let i = 0; i < samples.length; i += 1) {
    samples[i] = 0.5 * Math.sin((2 * Math.PI * frequency * i) / rate);
  }
  return samples;
}

/**
 * Read a `Blob` as bytes.
 *
 * jsdom's `Blob` implements neither `arrayBuffer()` nor `stream()`, so the browser-era `FileReader`
 * is the only portable way to get at the bytes inside the test environment.
 */
async function blobBytes(blob: Blob): Promise<ArrayBuffer> {
  return await new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      resolve(reader.result as ArrayBuffer);
    };
    reader.onerror = () => {
      reject(reader.error ?? new Error("blob okunamadı"));
    };
    reader.readAsArrayBuffer(blob);
  });
}

function ascii(view: DataView, offset: number, length: number): string {
  let out = "";
  for (let i = 0; i < length; i += 1) {
    out += String.fromCharCode(view.getUint8(offset + i));
  }
  return out;
}

describe("encodeWav16k", () => {
  it("writes a 16 kHz mono PCM16 header with consistent sizes", async () => {
    const chunks = [tone(48_000, 0.1), tone(48_000, 0.1)];
    const blob = await encodeWav16k(chunks, 48_000);
    expect(blob.type).toBe("audio/wav");

    const bytes = await blobBytes(blob);
    const view = new DataView(bytes);

    expect(ascii(view, 0, 4)).toBe("RIFF");
    expect(ascii(view, 8, 4)).toBe("WAVE");
    expect(ascii(view, 12, 4)).toBe("fmt ");
    expect(view.getUint32(16, true)).toBe(16); // PCM fmt chunk size
    expect(view.getUint16(20, true)).toBe(1); // audioFormat = PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(TARGET_SAMPLE_RATE);
    expect(view.getUint32(28, true)).toBe(TARGET_SAMPLE_RATE * BYTES_PER_SAMPLE); // byteRate
    expect(view.getUint16(32, true)).toBe(BYTES_PER_SAMPLE); // blockAlign
    expect(view.getUint16(34, true)).toBe(16); // bitsPerSample
    expect(ascii(view, 36, 4)).toBe("data");

    const dataBytes = view.getUint32(40, true);
    expect(bytes.byteLength).toBe(WAV_HEADER_BYTES + dataBytes);
    expect(view.getUint32(4, true)).toBe(bytes.byteLength - 8);
    // 0.2 s at 16 kHz = 3200 samples = 6400 bytes (rounding tolerance of one sample).
    expect(Math.abs(dataBytes - 6400)).toBeLessThanOrEqual(BYTES_PER_SAMPLE);
    expect(dataBytes % BYTES_PER_SAMPLE).toBe(0);
  });

  it("keeps 16 kHz input untouched", async () => {
    const samples = tone(TARGET_SAMPLE_RATE, 0.05);
    const blob = await encodeWav16k([samples], TARGET_SAMPLE_RATE);
    const view = new DataView(await blobBytes(blob));

    expect(view.getUint32(24, true)).toBe(TARGET_SAMPLE_RATE);
    expect(view.getUint32(40, true)).toBe(samples.length * BYTES_PER_SAMPLE);
  });

  it("rejects an implausible sample rate", async () => {
    await expect(encodeWav16k([new Float32Array(8)], 10)).rejects.toBeInstanceOf(AudioFormatError);
  });

  it("clips out-of-range samples to the PCM16 limits", () => {
    const bytes = encodeWavBytes(Float32Array.from([2, -2, 0]), TARGET_SAMPLE_RATE);
    const view = new DataView(bytes);

    expect(view.getInt16(WAV_HEADER_BYTES, true)).toBe(32767);
    expect(view.getInt16(WAV_HEADER_BYTES + 2, true)).toBe(-32768);
    expect(view.getInt16(WAV_HEADER_BYTES + 4, true)).toBe(0);
  });
});

describe("resampleSinc", () => {
  it("changes the length by the rate ratio", () => {
    const input = tone(48_000, 0.05);
    const output = resampleSinc(input, 48_000, TARGET_SAMPLE_RATE);

    expect(output.length).toBe(Math.round(input.length / 3));
  });

  it("preserves a constant signal (DC gain of one)", () => {
    const input = new Float32Array(600).fill(0.5);
    const output = resampleSinc(input, 44_100, TARGET_SAMPLE_RATE);
    const middle = output[Math.floor(output.length / 2)] ?? 0;

    expect(middle).toBeCloseTo(0.5, 3);
  });

  it("returns a copy when the rates match", () => {
    const input = tone(TARGET_SAMPLE_RATE, 0.01);
    const output = resampleSinc(input, TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE);

    expect(output).not.toBe(input);
    expect(Array.from(output)).toEqual(Array.from(input));
  });
});

describe("helpers", () => {
  it("concatenates chunks in order", () => {
    const joined = concatChunks([Float32Array.from([1, 2]), Float32Array.from([3])]);

    expect(Array.from(joined)).toEqual([1, 2, 3]);
  });

  it("computes duration from the sample count", () => {
    expect(durationSeconds(TARGET_SAMPLE_RATE * 3)).toBe(3);
  });
});
