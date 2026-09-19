/**
 * The administration surface (api.md §3.11).
 *
 * Three things are load-bearing and therefore tested: the role gate (a student must not reach the
 * area at all, an instructor must not be offered the administrator's panels), the honesty of the
 * model registry and of the evaluation history (a stand-in must say so; a run that is not
 * reportable must carry its warning), and the refusal path of the book loader, whose whole value
 * is the list of headings that were not tagged.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ModelVersionOut, UserOut } from "@/api/types";
import { clearSession, setAccessToken, setUser } from "@/app/auth";
import { AdminPage } from "@/features/admin/AdminPage";
import { BookLoadPanel } from "@/features/admin/BookLoadPanel";
import { EvalRunsPanel } from "@/features/admin/EvalRunsPanel";
import { ModelRegistryPanel } from "@/features/admin/ModelRegistryPanel";
import { QcQueuePanel } from "@/features/admin/QcQueuePanel";
import type { EvalRun, QcQueueItem } from "@/features/admin/api";
import { installApiMock, jsonResponse, requestBody } from "./apiMock";

function signIn(role: UserOut["role"]): void {
  setAccessToken("access-1", 900);
  setUser({
    id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
    email: "kisi@alparslan.edu.tr",
    display_name: "Kişi",
    role,
    created_at: "2026-09-01T08:00:00Z",
  });
}

const QUEUE_ITEM: QcQueueItem = {
  id: "rec-1",
  speaker_id: "P00012",
  prompt_id: "p-1",
  duration_s: 2.4,
  qc_passed: true,
  review_status: "pending",
  transcript_final: "x karenin türevini al",
};

const MODELS: ModelVersionOut[] = [
  {
    component: "asr",
    version: "asr-stub-1",
    path: "",
    sha256: "",
    metrics: {},
    run_id: null,
    is_active: true,
    promoted_at: null,
  },
  {
    component: "asr",
    version: "asr-2027.03",
    path: "models/asr",
    sha256: "sha256:abc",
    metrics: { wer: 0.084 },
    run_id: "run-1",
    is_active: false,
    promoted_at: null,
  },
];

const EVAL_RUNS: EvalRun[] = [
  {
    id: "run-1",
    kind: "asr",
    name: "ASR ara değerlendirme",
    n: 120,
    metrics: { wer: 0.1342 },
    dataset_fingerprint: "",
    model_version: "asr-stub-1",
    is_reportable: false,
    created_at: "2027-02-01T10:00:00Z",
  },
  {
    id: "run-2",
    kind: "cas",
    name: "CAS müfredat kümesi",
    n: 1000,
    metrics: { accuracy: 0.994 },
    dataset_fingerprint: "sha256:deadbeef",
    model_version: null,
    is_reportable: true,
    created_at: "2027-03-01T10:00:00Z",
  },
];

afterEach(() => {
  vi.unstubAllGlobals();
  clearSession();
});

describe("role gate", () => {
  it("refuses a student with the Turkish explanation instead of a redirect", () => {
    installApiMock({});
    signIn("student");
    render(<AdminPage />);

    expect(screen.getByText("Bu bölüme erişim yetkiniz yok")).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Kalite denetimi" })).not.toBeInTheDocument();
  });

  it("offers an instructor the staff panels but not the administrator's", async () => {
    installApiMock({ "GET /admin/qc/queue": [] });
    signIn("instructor");
    render(<AdminPage />);

    expect(await screen.findByRole("tab", { name: "Kalite denetimi" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Gecikme ölçümleri" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Model kayıt defteri" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Kullanıcılar ve roller" })).not.toBeInTheDocument();
  });

  it("offers an administrator every panel", async () => {
    installApiMock({ "GET /admin/qc/queue": [] });
    signIn("admin");
    render(<AdminPage />);

    expect(await screen.findByRole("tab", { name: "Model kayıt defteri" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Kitap yükleme" })).toBeInTheDocument();
  });
});

describe("quality-control queue", () => {
  it("sends the human verdict and takes the decided row out of the queue", async () => {
    const mock = installApiMock({
      "GET /admin/qc/queue": [QUEUE_ITEM],
      "POST /admin/qc/rec-1/review": (): Response => new Response(null, { status: 204 }),
    });
    signIn("instructor");
    render(<QcQueuePanel />);

    expect(await screen.findByText("P00012")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Düzeltilmiş metin"), {
      target: { value: "x karenin türevini al" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Düzeltip kabul et" }));

    expect(await screen.findByText("Karar kaydedildi.")).toBeInTheDocument();
    expect(screen.queryByText("P00012")).not.toBeInTheDocument();

    const reviewCall = mock.mock.calls.findIndex((call) =>
      (call[0] as Request).url.includes("/admin/qc/rec-1/review"),
    );
    expect(await requestBody(mock, reviewCall)).toEqual({
      verdict: "fixed",
      transcript_fix: "x karenin türevini al",
    });
  });
});

describe("model registry", () => {
  it("says in Turkish that the active model is a stand-in", async () => {
    installApiMock({ "GET /admin/models": MODELS });
    signIn("admin");
    render(<ModelRegistryPanel />);

    expect(await screen.findByText("asr-2027.03")).toBeInTheDocument();
    expect(screen.getAllByText("taklit model").length).toBeGreaterThan(0);
    expect(
      screen.getByText(
        "Etkin sürüm taklit (eğitilmemiş) bir modeldir; çıktıları başarım ölçüsü olarak raporlanamaz.",
      ),
    ).toBeInTheDocument();
  });

  it("promotes a checkpoint and reloads the registry", async () => {
    const mock = installApiMock({
      "GET /admin/models": MODELS,
      "POST /admin/models/activate": (): Response => new Response(null, { status: 204 }),
    });
    signIn("admin");
    render(<ModelRegistryPanel />);

    fireEvent.click(await screen.findByRole("button", { name: "Etkinleştir" }));

    expect(await screen.findByText("asr-2027.03 sürümü etkinleştirildi.")).toBeInTheDocument();
    const activateCall = mock.mock.calls.findIndex((call) =>
      (call[0] as Request).url.includes("/admin/models/activate"),
    );
    expect(await requestBody(mock, activateCall)).toEqual({
      component: "asr",
      version: "asr-2027.03",
    });
  });
});

describe("evaluation history", () => {
  it("separates a citable run from one that must not be quoted", async () => {
    installApiMock({ "GET /admin/eval-runs": EVAL_RUNS });
    signIn("admin");
    render(<EvalRunsPanel />);

    expect(await screen.findByText("ASR ara değerlendirme")).toBeInTheDocument();
    expect(screen.getByText("alıntılanabilir")).toBeInTheDocument();
    expect(screen.getByText("alıntılanamaz")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Bu koşu raporlanabilir olarak işaretlenmemiş: veri kümesi veya model kaynağı doğrulanmamış olabilir. Sayıları TÜBİTAK raporuna yazmayın.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("parmak izi yok")).toBeInTheDocument();
    expect(screen.getByText("sha256:deadbeef")).toBeInTheDocument();
    expect(screen.getByText("0.1342")).toBeInTheDocument();
  });
});

describe("book loading", () => {
  it("lists the untagged headings when the loader refuses the archive", async () => {
    installApiMock({
      "POST /admin/book/load": (): Response =>
        new Response(
          JSON.stringify({
            type: "https://ai-cls.example/errors/VALIDATION_ERROR",
            title: "İstek doğrulanamadı",
            status: 422,
            code: "VALIDATION_ERROR",
            detail_tr: "İstek doğrulanamadı; lütfen alanları kontrol edin.",
            tagged_ratio: 0.8,
            untagged_headings: ["2.4 Zincir Kuralı", "2.5 Kapalı Türev"],
          }),
          { status: 422, headers: { "content-type": "application/problem+json" } },
        ),
    });
    signIn("admin");
    render(<BookLoadPanel />);

    const input = screen.getByLabelText("ZIP arşivi");
    const archive = new File([new Uint8Array([1, 2, 3])], "bolum.zip", { type: "application/zip" });
    fireEvent.change(input, { target: { files: [archive] } });
    fireEvent.click(screen.getByRole("button", { name: "Yükle" }));

    const list = await screen.findByTestId("untagged-headings");
    expect(list).toHaveTextContent("2.4 Zincir Kuralı");
    expect(list).toHaveTextContent("2.5 Kapalı Türev");
    expect(list).toHaveTextContent("%80");
  });

  it("reports the four figures of a successful load", async () => {
    installApiMock({
      "POST /admin/book/load": (): Response =>
        jsonResponse({ units: 42, triples: 210, aliases: 17, tagged_ratio: 1 }),
    });
    signIn("admin");
    render(<BookLoadPanel />);

    const archive = new File([new Uint8Array([1])], "bolum.zip", { type: "application/zip" });
    fireEvent.change(screen.getByLabelText("ZIP arşivi"), { target: { files: [archive] } });
    fireEvent.click(screen.getByRole("button", { name: "Yükle" }));

    expect(await screen.findByText("Kitap yüklendi.")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("42")).toBeInTheDocument();
    });
    expect(screen.getByText("%100")).toBeInTheDocument();
  });
});
