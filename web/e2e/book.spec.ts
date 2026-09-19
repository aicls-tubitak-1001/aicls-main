/**
 * `e2e/book.spec.ts` — "türev tanımını getir" must open the book panel with the tagged definition
 * instead of running the CAS (İP-8, api.md §3.7).
 */
import { expect, test, type Page } from "@playwright/test";

const DEFINITION_UNIT = {
  id: "tanim-2.4",
  type: "definition",
  title_tr: "Türev tanımı",
  chapter: "2",
  rendered: {
    body_html: "<p>Bir fonksiyonun bir noktadaki türevi, fark oranının limitidir.</p>",
    latex_blocks: ["f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}"],
    assets: [],
  },
};

const BOOK_OUTCOME = {
  kind: "book",
  operation_id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7S",
  input_latex: null,
  command: null,
  result: null,
  clarification: null,
  book_units: [DEFINITION_UNIT],
  nlu: { intent: "book_lookup", score: 0.97, trace: [], source: "rules" },
  timings: { nlu: 8.1, total: 11.2 },
  model_versions: { ocr: null, asr: null },
};

async function stubApi(page: Page): Promise<void> {
  await page.route("**/api/v1/compute/text", async (route) => {
    await route.fulfill({ json: BOOK_OUTCOME });
  });
  await page.route("**/api/v1/book/search**", async (route) => {
    await route.fulfill({ json: [{ unit: DEFINITION_UNIT, score: 0.9, matched_alias: "türev" }] });
  });
}

test.describe("kitap entegrasyonu", () => {
  test.beforeEach(async ({ page }) => {
    await stubApi(page);
  });

  test("'türev tanımını getir' kitap panelini açar", async ({ page }) => {
    await page.goto("/calisma-alani");

    await page.getByLabel("Komut", { exact: true }).fill("türev tanımını getir");
    await page.getByRole("button", { name: "Çalıştır" }).click();

    const panel = page.getByTestId("book-panel");
    await expect(panel).toBeVisible();
    await expect(panel.getByText("Türev tanımı")).toBeVisible();
    await expect(panel.getByText("Tanım")).toBeVisible();
    await expect(panel.locator(".katex")).toBeVisible();
  });

  test("kitap sayfasından arama yapılabilir", async ({ page }) => {
    await page.goto("/kitap");

    await page.getByLabel("Tanım, teorem veya örnek arayın").fill("türev");
    await page.getByRole("button", { name: "Ara" }).click();

    await expect(page.getByText("Türev tanımı")).toBeVisible();
  });
});
