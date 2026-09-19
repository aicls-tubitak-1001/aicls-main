/**
 * `e2e/workspace.spec.ts` — handwriting and the confirmation step in a real browser.
 *
 * The canvas is driven with real mouse events, which is exactly what jsdom cannot do: this is the
 * spec that proves `StrokeBuffer` receives coordinates from a genuine pointer pipeline.
 */
import { expect, test, type Page } from "@playwright/test";

const CLARIFICATION_OUTCOME = {
  kind: "clarification",
  operation_id: null,
  input_latex: "x^{2}",
  command: null,
  result: null,
  clarification: {
    question_tr: "Hangi değişkene göre türev alınsın?",
    missing_slot: "var",
    candidates: [
      {
        intent: "differentiate",
        command: { op: "diff", expr: "x^{2}", var: "x", order: 1 },
        latex_preview: "\\frac{d}{dx} x^{2}",
        score: 0.61,
        label_tr: "x'e göre türev",
      },
    ],
  },
  book_units: [],
  nlu: { intent: "differentiate", score: 0.61, trace: [], source: "rules" },
  timings: { nlu: 14.2, total: 15.0 },
  model_versions: { ocr: null, asr: null },
};

const RESULT_OUTCOME = {
  ...CLARIFICATION_OUTCOME,
  kind: "result",
  operation_id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7T",
  clarification: null,
  result: {
    kind: "expr",
    latex: "2 x",
    plain: "2*x",
    srepr: "Mul(Integer(2), Symbol('x'))",
    steps: [],
    plot_url: null,
    cache_hit: false,
  },
};

async function stubApi(page: Page): Promise<void> {
  let call = 0;
  await page.route("**/api/v1/compute/text", async (route) => {
    call += 1;
    await route.fulfill({ json: call === 1 ? CLARIFICATION_OUTCOME : RESULT_OUTCOME });
  });
  await page.route("**/api/v1/telemetry/latency", async (route) => {
    await route.fulfill({ status: 204, body: "" });
  });
}

test.describe("çalışma alanı", () => {
  test.beforeEach(async ({ page }) => {
    await stubApi(page);
  });

  test("belirsiz komut önce soru sorar, seçilen adayla hesaplanır", async ({ page }) => {
    await page.goto("/calisma-alani");

    await page.getByLabel("Komut", { exact: true }).fill("türevini al");
    await page.getByRole("button", { name: "Çalıştır" }).click();

    // Bir açıklama isteği 200'dür: hata gibi değil, soru gibi görünür.
    await expect(page.getByTestId("clarification")).toContainText(
      "Hangi değişkene göre türev alınsın?",
    );
    const candidate = page.getByTestId("clarification-candidate").first();
    await expect(candidate).toBeVisible();
    await expect(candidate).toContainText("x'e göre türev");

    await candidate.click();

    await expect(page.getByTestId("result-latex").locator(".katex")).toBeVisible();
  });

  test("sunum modu ikinci bir düzen değil, aynı ekranın büyüğüdür", async ({ page }) => {
    await page.goto("/calisma-alani");

    await page.getByRole("button", { name: "Sunum moduna geç" }).click();

    await expect(page.locator("html")).toHaveAttribute("data-density", "presentation");
    // Komut çubuğu ve sonuç alanı sunum modunda da yerinde kalır.
    await expect(page.getByLabel("Komut", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Sunum modundan çık" })).toBeVisible();
  });

  test("sunum modunda kitap paneli hâlâ açılabilir", async ({ page }) => {
    // İP-8'in bütün değeri bu panelde: ders ortasında "türev tanımını getir" dendiğinde tanımın
    // duvara gelmesi gerekir. Sunum modu paneli kapatır, ama erişilemez yapmaz.
    await page.goto("/calisma-alani");
    const book = page.getByRole("complementary", { name: "Kitap paneli" });

    await page.getByRole("button", { name: "Sunum moduna geç" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-density", "presentation");
    await expect(book).toBeHidden();

    await page.getByRole("button", { name: "Kitap", exact: true }).click();

    await expect(book).toBeVisible();
    await expect(page.locator("html")).toHaveAttribute("data-density", "presentation");
  });

  test("fare ile çizilen iki iz sayaca yansır", async ({ page }) => {
    await page.goto("/calisma-alani");
    const canvas = page.getByTestId("ink-canvas");
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    if (box === null) {
      return;
    }

    for (const offset of [0, 120]) {
      await page.mouse.move(box.x + 40 + offset, box.y + 60);
      await page.mouse.down();
      await page.mouse.move(box.x + 70 + offset, box.y + 100, { steps: 8 });
      await page.mouse.up();
    }

    await expect(page.getByTestId("stroke-count")).toHaveText("İz sayısı: 2");

    await page.getByRole("button", { name: "Tuvali temizle" }).click();
    await expect(page.getByTestId("stroke-count")).toHaveText("İz sayısı: 0");
  });
});
