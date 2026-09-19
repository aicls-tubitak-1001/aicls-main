/**
 * The in-site guide.
 *
 * Four things are worth a test here, and they are the four ways this page could quietly rot:
 * the route stops rendering, the table of contents points at sections that no longer exist, the
 * status page prints numbers that no longer match the measurement, and the mathematics silently
 * falls back to raw LaTeX. A fifth test guards the technical page's factual claims against the
 * committed OpenAPI document, so the API summary cannot drift away from the schema it describes.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";

import { ERROR_CODES } from "@/api/problem";
import { App } from "@/App";
import { ToastProvider } from "@/components/ui/Toast";
import { API_TAGS, API_TOTALS, ERROR_ROWS } from "@/features/guide/apiSurface";
import { FirstRunCard } from "@/features/guide/FirstRunCard";
import { GuidePage } from "@/features/guide/GuidePage";
import { FIRST_RUN_STORAGE_KEY } from "@/features/guide/firstRun";
import { GUIDE_ROUTES } from "@/features/guide/routes";
import { StatusGuide } from "@/features/guide/StatusGuide";
import {
  CONTRACT_TARGETS,
  METRICS_SNAPSHOT,
  type ContractTarget,
  type MetricsSnapshot,
} from "@/features/guide/metrics";
import { GUIDE_SECTIONS, type GuidePageId } from "@/features/guide/sections";
import { t } from "@/i18n";
import { installApiMock } from "./apiMock";

function renderAt(path: string): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <ToastProvider>
        <App />
      </ToastProvider>
    </MemoryRouter>,
  );
}

function renderGuide(page: GuidePageId, path: string): void {
  render(
    <MemoryRouter initialEntries={[path]}>
      <GuidePage page={page} />
    </MemoryRouter>,
  );
}

/** A deliberately small snapshot: one run, one metric, one provenance warning. */
const FIXTURE: MetricsSnapshot = {
  gitSha: "0123456789abcdef",
  generatedAt: "2026-01-02T03:04:05Z",
  definitionsRef: "docs/ml/degerlendirme.md#2-metrikler",
  runs: [
    {
      id: "ocr",
      titleKey: "guide.status.run.ocr",
      evidence: "stub",
      n: 128,
      datasetVersion: "fixture-ink-v1",
      datasetFingerprint: "sha256:aaaabbbbccccdddd",
      gitSha: "0123456789abcdef",
      generatedAt: "2026-01-02T03:04:05Z",
      modelVersion: "stub-0.1.0",
      citable: false,
      noticeKeys: ["guide.status.notice.ocrStub"],
      metrics: [
        {
          id: "ocr.cer",
          labelKey: "guide.status.metric.cer",
          value: 0.1234,
          unit: "ratio",
          ci95: [0.1, 0.2],
          target: { text: "< 0,10", met: false },
        },
      ],
    },
  ],
};

const FIXTURE_CONTRACT: readonly ContractTarget[] = [
  {
    id: "ocr",
    labelKey: "guide.status.contract.ocr",
    todayKey: "guide.status.contract.ocrToday",
    evidence: "stub",
  },
];

afterEach(() => {
  globalThis.localStorage.clear();
});

describe("Kılavuz route", () => {
  it("is reachable from the main navigation and opens the lecturer's page", () => {
    installApiMock({});
    renderAt(GUIDE_ROUTES.root);

    expect(screen.getByRole("link", { name: "Kılavuz" })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 1, name: "Kullanım Kılavuzu" }),
    ).toBeInTheDocument();
  });

  it("keeps exactly one h1 and a labelled sub-navigation on every page", () => {
    const pages: [GuidePageId, string, string][] = [
      ["usage", GUIDE_ROUTES.usage, "Kullanım Kılavuzu"],
      ["tech", GUIDE_ROUTES.tech, "Teknik Kılavuz"],
      ["status", GUIDE_ROUTES.status, "Durum: bugün ne ölçüldü"],
    ];
    for (const [page, path, title] of pages) {
      const { unmount } = render(
        <MemoryRouter initialEntries={[path]}>
          <GuidePage page={page} />
        </MemoryRouter>,
      );
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(title);
      expect(screen.getByRole("navigation", { name: "Kılavuz bölümleri" })).toBeInTheDocument();
      unmount();
    }
  });
});

describe("İçindekiler", () => {
  const pages: [GuidePageId, string][] = [
    ["usage", GUIDE_ROUTES.usage],
    ["tech", GUIDE_ROUTES.tech],
    ["status", GUIDE_ROUTES.status],
  ];

  it.each(pages)("links only to sections that exist (%s)", (page, path) => {
    renderGuide(page, path);

    const toc = screen.getByTestId("guide-toc");
    const links = within(toc).getAllByRole("link");
    expect(links).toHaveLength(GUIDE_SECTIONS[page].length);

    for (const [index, link] of links.entries()) {
      const section = GUIDE_SECTIONS[page][index];
      expect(section).toBeDefined();
      const id = section?.id ?? "";
      expect(link).toHaveAttribute("href", `#${id}`);

      const target = document.getElementById(id);
      expect(target, `#${id} bölümü sayfada yok`).not.toBeNull();
      const heading = within(target as HTMLElement).getAllByRole("heading", { level: 2 })[0];
      expect(heading).toHaveTextContent(t(section?.titleKey ?? "guide.title"));
      expect(link).toHaveTextContent(t(section?.titleKey ?? "guide.title"));
    }
  });

  it("marks the first section as the current location", () => {
    renderGuide("usage", GUIDE_ROUTES.usage);

    const toc = screen.getByTestId("guide-toc");
    const links = within(toc).getAllByRole("link");
    expect(links[0]).toHaveAttribute("aria-current", "location");
  });
});

describe("Durum sayfası", () => {
  it("renders the measured values, the interval and the verdict from a fixture", () => {
    render(
      <MemoryRouter>
        <StatusGuide snapshot={FIXTURE} contract={FIXTURE_CONTRACT} />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("metric-ocr.cer")).toHaveTextContent("0,1234");
    expect(screen.getByText("[0,1000; 0,2000]")).toBeInTheDocument();
    const run = screen.getByTestId("guide-run-ocr");
    expect(within(run).getByText("tutmadı")).toBeInTheDocument();
    expect(within(run).getByText(/fixture-ink-v1/)).toBeInTheDocument();
    expect(within(run).getByText(/stub-0\.1\.0/)).toBeInTheDocument();
  });

  it("prints the provenance warning and the unquotable notice next to the number", () => {
    render(
      <MemoryRouter>
        <StatusGuide snapshot={FIXTURE} contract={FIXTURE_CONTRACT} />
      </MemoryRouter>,
    );

    const run = screen.getByTestId("guide-run-ocr");
    expect(within(run).getByText(t("guide.status.notice.ocrStub"))).toBeInTheDocument();
    expect(within(run).getByText(t("guide.status.notCitable"))).toBeInTheDocument();
    // The evidence level is stated twice on purpose: once as the run's badge, once as the
    // label of the warning itself, so neither can be skimmed past.
    expect(within(run).getAllByText("taklit model")).toHaveLength(2);
  });

  it("shows the committed snapshot's failing ASR row without softening it", () => {
    render(
      <MemoryRouter>
        <StatusGuide />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("metric-asr.wer_tr_v1")).toHaveTextContent("1,1118");
    const asr = screen.getByTestId("guide-run-asr");
    expect(within(asr).getAllByText("tutmadı").length).toBeGreaterThanOrEqual(2);
  });

  it("states the evidence level of every contractual target", () => {
    render(
      <MemoryRouter>
        <StatusGuide />
      </MemoryRouter>,
    );

    const table = screen.getByTestId("guide-contract-table");
    const rows = within(table).getAllByRole("row");
    // One header row plus one row per contractual target.
    expect(rows).toHaveLength(CONTRACT_TARGETS.length + 1);
    expect(within(table).getByText("Kanıt yok")).toBeInTheDocument();
    expect(within(table).getAllByText("Yalnızca bağlantı denetimi").length).toBe(
      CONTRACT_TARGETS.filter((target) => target.evidence === "stub").length,
    );
  });

  it("keeps every committed run marked as unquotable", () => {
    expect(METRICS_SNAPSHOT.runs.every((run) => !run.citable)).toBe(true);
  });
});

describe("Kılavuz matematiği", () => {
  it("typesets the worked examples with KaTeX", () => {
    const { container } = render(
      <MemoryRouter>
        <GuidePage page="usage" />
      </MemoryRouter>,
    );

    const example = screen.getByTestId("guide-example-yazarak");
    // Attribute queries rather than `getByRole`: jsdom's getComputedStyle throws on the MathML
    // subtree KaTeX emits for screen readers, which any role query would walk.
    expect(example.querySelectorAll('[role="math"]')).toHaveLength(2);
    expect(example.querySelector(".katex")).not.toBeNull();
    expect(example.querySelector("[data-mathjax-fallback]")).toBeNull();
    // Every expression in the guide reaches KaTeX; none of them lands in the fallback slot.
    expect(container.querySelectorAll(".katex").length).toBeGreaterThanOrEqual(6);
    expect(container.querySelectorAll("[data-mathjax-fallback]")).toHaveLength(0);
  });

  it("shows the integral, the limit and their plain-text forms", () => {
    render(
      <MemoryRouter>
        <GuidePage page="usage" />
      </MemoryRouter>,
    );

    expect(screen.getByText("integrate(x**2, (x, 0, 1))")).toBeInTheDocument();
    expect(screen.getByText("limit(sin(x)/x, x, 0)")).toBeInTheDocument();
    expect(screen.getByText("2*x")).toBeInTheDocument();
  });

  it("quotes the frozen contract sentence for an unsupported problem", () => {
    renderGuide("usage", GUIDE_ROUTES.usage);

    expect(screen.getByText("Bu problem tipi şu an için desteklenmemektedir.")).toBeInTheDocument();
  });

  it("renders the keyboard table from the binding set", () => {
    renderGuide("usage", GUIDE_ROUTES.usage);

    const table = screen.getByTestId("guide-shortcut-table");
    expect(within(table).getByText("Alt + S")).toBeInTheDocument();
    expect(within(table).getByText("Sunum modunu aç/kapat")).toBeInTheDocument();
  });
});

describe("Teknik sayfa", () => {
  // Vitest runs with `web/` as the working directory (that is where vite.config.ts lives).
  const specPath = resolve("src/api/generated/openapi.json");
  const spec = JSON.parse(readFileSync(specPath, "utf-8")) as {
    paths: Record<string, Record<string, { tags?: string[] }>>;
    components: { schemas: Record<string, unknown> };
  };
  const METHODS = new Set(["get", "post", "put", "patch", "delete"]);

  it("summarises the committed OpenAPI document without drifting from it", () => {
    const counts = new Map<string, number>();
    let operations = 0;
    for (const operationsByMethod of Object.values(spec.paths)) {
      for (const [method, operation] of Object.entries(operationsByMethod)) {
        if (!METHODS.has(method)) {
          continue;
        }
        operations += 1;
        for (const tag of operation.tags ?? []) {
          counts.set(tag, (counts.get(tag) ?? 0) + 1);
        }
      }
    }

    expect(API_TOTALS.paths).toBe(Object.keys(spec.paths).length);
    expect(API_TOTALS.operations).toBe(operations);
    expect(API_TOTALS.schemas).toBe(Object.keys(spec.components.schemas).length);
    expect(API_TAGS.map((row) => row.tag).sort()).toEqual([...counts.keys()].sort());
    for (const row of API_TAGS) {
      expect(counts.get(row.tag), `${row.tag} etiketinin işlem sayısı`).toBe(row.count);
    }
  });

  it("documents every error code the client knows about", () => {
    expect(ERROR_ROWS.map((row) => row.code).sort()).toEqual([...ERROR_CODES].sort());
  });

  it("renders the pipeline diagram with a text equivalent", () => {
    renderGuide("tech", GUIDE_ROUTES.tech);

    const figure = screen.getByRole("img", { name: /AI-CLS boru hattı/ });
    expect(figure).toBeInTheDocument();
    expect(screen.getByText("Boru hattı, metin olarak")).toBeInTheDocument();
    expect(
      screen.getByText(/SymPy hesabı ayrı bir süreçte, zaman sınırlı olarak yürütür\./),
    ).toBeInTheDocument();
  });
});

describe("İlk beş dakika kartı", () => {
  function renderCard(): void {
    render(
      <MemoryRouter>
        <ToastProvider>
          <FirstRunCard />
        </ToastProvider>
      </MemoryRouter>,
    );
  }

  it("links into the guide and is shown by default", () => {
    renderCard();

    expect(screen.getByTestId("first-run-card")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Kılavuzu aç" })).toHaveAttribute(
      "href",
      GUIDE_ROUTES.usage,
    );
  });

  it("hides itself when dismissed, remembers it, and offers an undo", () => {
    renderCard();

    fireEvent.click(screen.getByRole("button", { name: "Bu kartı gizle" }));

    expect(screen.queryByTestId("first-run-card")).toBeNull();
    expect(globalThis.localStorage.getItem(FIRST_RUN_STORAGE_KEY)).toBe("dismissed");

    fireEvent.click(screen.getByRole("button", { name: "Geri al" }));

    expect(screen.getByTestId("first-run-card")).toBeInTheDocument();
    expect(globalThis.localStorage.getItem(FIRST_RUN_STORAGE_KEY)).toBeNull();
  });

  it("stays hidden after a reload once it was dismissed", () => {
    globalThis.localStorage.setItem(FIRST_RUN_STORAGE_KEY, "dismissed");
    renderCard();

    expect(screen.queryByTestId("first-run-card")).toBeNull();
  });
});
