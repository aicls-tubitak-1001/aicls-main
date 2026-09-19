/**
 * The voice data-collection flow (proposal A.2.3, api.md §3.8, ADR-0017).
 *
 * What is asserted here is what the İP-3 campaign actually depends on: that nothing recordable is
 * reachable before consent, that a rejected take explains itself in Turkish and can be retried on
 * the spot, that withdrawal says what it deletes before it deletes it, and that a closed ethics
 * gate reads as "not open yet" rather than as a failure.
 *
 * The microphone is replaced by a scriptable `AudioContext`: `getUserMedia` yields a fake stream
 * and the `ScriptProcessorNode` callback is driven by hand, so capture → `encodeWav16k` → upload →
 * verdict runs deterministically without audio hardware.
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ConsentTextOut, ProgressOut, PromptOut, RecordingOut } from "@/api/types";
import { ContributePage } from "@/features/contribute/ContributePage";
import {
  clearParticipant,
  participantStore,
  setParticipant,
} from "@/features/contribute/participant";
import { RecorderPanel } from "@/features/contribute/RecorderPanel";
import { WithdrawDialog } from "@/features/contribute/WithdrawDialog";
import { filePart, formDataSerializer } from "@/api/client";
import { installApiMock, jsonResponse, problemResponse, requestBody } from "./apiMock";

const CONSENT: ConsentTextOut = {
  version: "2026-09-01",
  text_tr: "Ses kayıtlarınız takma adla saklanacaktır.",
  sha256: "sha256:0",
  effective_from: "2026-09-01",
};

const PROMPT: PromptOut = {
  id: "p-1",
  text_tr: "x karenin türevini al",
  scenario_tag: "turev",
  difficulty: 1,
};

const PROGRESS: ProgressOut = { n_accepted: 3, seconds_accepted: 42, target_seconds: 1800 };

const ACCEPTED: RecordingOut = {
  id: "rec-1",
  accepted: true,
  duration_s: 2.4,
  qc: {
    passed: true,
    checks: [{ name: "snr_db", value: 24.5, threshold: 15, passed: true }],
    reasons_tr: [],
  },
};

const REJECTED: RecordingOut = {
  id: "rec-2",
  accepted: false,
  duration_s: 0.4,
  qc: {
    passed: false,
    checks: [{ name: "duration_min", value: 0.4, threshold: 0.8, passed: false }],
    reasons_tr: ["Kayıt çok kısa: 0.40 saniye (en az 0.80 saniye olmalı)."],
  },
};

/** The `onaudioprocess` callback the hook installed, so a test can push samples. */
let audioCallback:
  ((event: { inputBuffer: { getChannelData: () => Float32Array } }) => void) | null = null;

/** Replace `navigator.mediaDevices` and `AudioContext` with scriptable doubles. */
function installAudioStubs(sampleRate = 48_000): void {
  audioCallback = null;
  const processor = {
    onaudioprocess: null as unknown,
    connect: () => undefined,
    disconnect: () => undefined,
  };
  const context = {
    sampleRate,
    createMediaStreamSource: () => ({ connect: () => undefined, disconnect: () => undefined }),
    createScriptProcessor: () => processor,
    destination: {},
    close: () => Promise.resolve(),
  };
  vi.stubGlobal("AudioContext", function AudioContextStub() {
    return context;
  });
  Object.defineProperty(globalThis.navigator, "mediaDevices", {
    configurable: true,
    value: {
      getUserMedia: () => Promise.resolve({ getTracks: () => [{ stop: () => undefined }] }),
    },
  });
  // The hook installs the callback synchronously while opening the microphone.
  Object.defineProperty(processor, "onaudioprocess", {
    configurable: true,
    get: () => audioCallback,
    set: (value: typeof audioCallback) => {
      audioCallback = value;
    },
  });
}

/** Feed one block of loud-but-unclipped speech into the capture callback. */
function pushAudio(samples = 4_800): void {
  const block = new Float32Array(samples);
  for (let i = 0; i < block.length; i += 1) {
    block[i] = 0.3 * Math.sin(i / 8);
  }
  act(() => {
    audioCallback?.({ inputBuffer: { getChannelData: () => block } });
  });
}

/** Record one take: start, push audio, stop. */
async function recordOnce(): Promise<void> {
  fireEvent.click(screen.getByRole("button", { name: "Kaydı başlat" }));
  await screen.findByRole("button", { name: "Kaydı durdur" });
  pushAudio();
  fireEvent.click(screen.getByRole("button", { name: "Kaydı durdur" }));
}

beforeEach(() => {
  clearParticipant();
  installAudioStubs();
});

afterEach(() => {
  vi.unstubAllGlobals();
  clearParticipant();
});

describe("consent gate", () => {
  it("shows the consent text and offers nothing recordable before it is accepted", async () => {
    installApiMock({ "GET /contribute/consent/current": CONSENT });
    render(
      <MemoryRouter>
        <ContributePage />
      </MemoryRouter>,
    );

    expect(await screen.findByTestId("consent-text")).toHaveTextContent(
      "Ses kayıtlarınız takma adla saklanacaktır.",
    );
    expect(screen.queryByRole("button", { name: "Kaydı başlat" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Onayla ve başla" })).toBeDisabled();
  });

  it("sends band-only demographics and then unlocks the reading loop", async () => {
    const mock = installApiMock({
      "GET /contribute/consent/current": CONSENT,
      "POST /contribute/consent": (): Response =>
        jsonResponse({ speaker_pseudonym: "P00012", token: "spk-token" }, 201),
      "GET /contribute/prompts/next": [PROMPT],
      "GET /contribute/me/progress": PROGRESS,
    });
    render(
      <MemoryRouter>
        <ContributePage />
      </MemoryRouter>,
    );

    fireEvent.click(await screen.findByLabelText("Onam metnini okudum ve kabul ediyorum."));
    fireEvent.click(screen.getByRole("button", { name: "Onayla ve başla" }));

    expect(await screen.findByTestId("prompt-sentence")).toHaveTextContent("x karenin türevini al");
    expect(participantStore.getState().participant?.token).toBe("spk-token");

    const consentCall = mock.mock.calls.findIndex((call) => {
      const request = call[0] as Request;
      return request.method === "POST" && request.url.endsWith("/contribute/consent");
    });
    expect(await requestBody(mock, consentCall)).toEqual({
      version: "2026-09-01",
      accepted: true,
      demographics: { age_band: "25-34", gender: "na", university: "other" },
    });

    const promptsCall = mock.mock.calls.find((call) => {
      const request = call[0] as Request;
      return request.url.includes("/contribute/prompts/next");
    });
    const promptsRequest = promptsCall?.[0] as Request;
    expect(promptsRequest.headers.get("X-Participant-Token")).toBe("spk-token");
  });

  it("renders a calm 'not open yet' state when the ethics gate answers 503", async () => {
    installApiMock({
      "GET /contribute/consent/current": (): Response =>
        problemResponse("GATE_BLOCKED", 503, "Bu işlem onay kapısı tarafından engellendi."),
    });
    render(
      <MemoryRouter>
        <ContributePage />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Ses toplama henüz açık değil")).toBeInTheDocument();
    expect(screen.queryByTestId("consent-text")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("quality control", () => {
  it("explains a rejection in Turkish and lets the volunteer read the sentence again", async () => {
    installApiMock({
      "POST /contribute/recordings": (): Response => jsonResponse(REJECTED),
    });
    render(
      <RecorderPanel
        prompt={PROMPT}
        token="spk-token"
        onNext={vi.fn()}
        onSkip={vi.fn()}
        onAccepted={vi.fn()}
      />,
    );

    await recordOnce();

    const verdict = await screen.findByTestId("qc-verdict");
    expect(verdict).toHaveTextContent("Kayıt kabul edilmedi");
    expect(verdict).toHaveTextContent("Kayıt çok kısa: 0.40 saniye (en az 0.80 saniye olmalı).");
    expect(screen.getByRole("button", { name: "Yeniden kaydet" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Yeniden kaydet" }));
    expect(await screen.findByRole("button", { name: "Kaydı durdur" })).toBeInTheDocument();
    expect(screen.queryByTestId("qc-verdict")).not.toBeInTheDocument();
  });

  it("uploads with the participant token and reports an accepted take", async () => {
    const onAccepted = vi.fn();
    const mock = installApiMock({
      "POST /contribute/recordings": (): Response => jsonResponse(ACCEPTED),
    });
    render(
      <RecorderPanel
        prompt={PROMPT}
        token="spk-token"
        onNext={vi.fn()}
        onSkip={vi.fn()}
        onAccepted={onAccepted}
      />,
    );

    await recordOnce();

    expect(await screen.findByTestId("qc-verdict")).toHaveTextContent("Kayıt kabul edildi");
    await waitFor(() => {
      expect(onAccepted).toHaveBeenCalledWith(ACCEPTED);
    });

    const request = mock.mock.calls[0]?.[0] as Request;
    expect(request.method).toBe("POST");
    expect(request.url).toContain("/contribute/recordings");
    expect(request.headers.get("X-Participant-Token")).toBe("spk-token");
  });

  it("packs the prompt id and the WAV file into the multipart body", () => {
    const form = formDataSerializer({
      audio: filePart(new Blob([new Uint8Array([1, 2, 3])], { type: "audio/wav" }), "kayit.wav"),
      prompt_id: "p-1",
    });

    // jsdom's `FormData` is not the one Node's `Request` accepts, so the serializer is checked
    // directly instead of through a request body it cannot carry in this environment.
    expect(form.get("prompt_id")).toBe("p-1");
    expect(form.get("audio")).toBeInstanceOf(File);
  });
});

describe("withdrawal", () => {
  it("says what is deleted and what is kept before it deletes anything", async () => {
    const mock = installApiMock({
      "POST /contribute/withdraw": (): Response => new Response(null, { status: 204 }),
    });
    setParticipant({ pseudonym: "P00012", token: "spk-token", consentVersion: "2026-09-01" });
    const onWithdrawn = vi.fn();

    render(<WithdrawDialog open onClose={vi.fn()} token="spk-token" onWithdrawn={onWithdrawn} />);

    expect(
      screen.getByText(
        "Geri çektiğinizde ses kayıtlarınız sunucudan ve yedeklerden silinir, eğitim kümesinden çıkarılır.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Yalnızca onam verdiğinize dair takma adlı kayıt saklanır; bu, mevzuat gereği silinemez ve kimliğinizi içermez.",
      ),
    ).toBeInTheDocument();
    expect(mock).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Evet, katılımımı geri çek" }));

    await waitFor(() => {
      expect(onWithdrawn).toHaveBeenCalled();
    });
    const request = mock.mock.calls[0]?.[0] as Request;
    expect(request.headers.get("X-Participant-Token")).toBe("spk-token");
    expect(participantStore.getState().participant).toBeNull();
  });
});
