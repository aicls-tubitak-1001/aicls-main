/**
 * `e2e/browsers.spec.ts` — contractual smoke test: a Turkish command produces a KaTeX-rendered
 * result in chromium, firefox and webkit (proposal A.1.2 browser compatibility).
 *
 * The API is stubbed at the network layer so the spec measures the front end only; the full-stack
 * path is covered by the pytest e2e suite.
 */
import { expect, test, type Page } from "@playwright/test";

const COMPUTE_RESPONSE = {
  kind: "result",
  operation_id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  input_latex: "x^{2}",
  command: { op: "diff", expr: "x^{2}", var: "x", order: 1 },
  result: {
    kind: "expr",
    latex: "2 x",
    plain: "2*x",
    srepr: "Mul(Integer(2), Symbol('x'))",
    steps: [],
    plot_url: null,
    cache_hit: false,
  },
  clarification: null,
  book_units: [],
  nlu: { intent: "differentiate", score: 0.93, trace: [], source: "rules" },
  timings: { nlu: 12.3, parse: 1.1, cas: 41.0, render: 0.4, total: 58.2 },
  model_versions: { ocr: null, asr: null },
};

/** Stub the two endpoints the workspace touches on load. */
async function stubApi(page: Page): Promise<void> {
  await page.route("**/api/v1/compute/text", async (route) => {
    await route.fulfill({ json: COMPUTE_RESPONSE });
  });
  await page.route("**/api/v1/telemetry/latency", async (route) => {
    await route.fulfill({ status: 204, body: "" });
  });
}

test.describe("çapraz tarayıcı komut akışı", () => {
  test.beforeEach(async ({ page }) => {
    await stubApi(page);
  });

  test("Türkçe komut KaTeX ile işlenmiş sonuç üretir", async ({ page }) => {
    await page.goto("/calisma-alani");

    await page.getByLabel("Komut", { exact: true }).fill("x karenin türevini al");
    await page.getByRole("button", { name: "Çalıştır" }).click();

    const result = page.getByTestId("result-latex");
    await expect(result).toBeVisible();
    // KaTeX renders into .katex; a MathJax fallback slot would mean the expression was rejected.
    await expect(result.locator(".katex")).toBeVisible();
    await expect(result.locator("[data-mathjax-fallback]")).toHaveCount(0);
    await expect(page.getByText("2*x")).toBeVisible();
  });

  test("sayfa Türkçe dil etiketiyle açılır", async ({ page }) => {
    await page.goto("/calisma-alani");

    await expect(page.locator("html")).toHaveAttribute("lang", "tr");
    await expect(page.getByRole("navigation", { name: "Ana gezinme" })).toBeVisible();
  });

  test("koyu tema seçimi belgeye yazılır", async ({ page }) => {
    await page.goto("/calisma-alani");

    await page.getByLabel("Görünüm").selectOption("dark");

    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });
});
