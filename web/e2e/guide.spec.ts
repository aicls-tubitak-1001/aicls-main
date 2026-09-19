/**
 * `e2e/guide.spec.ts` — the in-site guide in a real browser.
 *
 * Three behaviours cannot be judged under jsdom and are exactly the ones a lecturer depends on:
 * a deep link actually scrolls to its section, the table of contents follows the scroll, and the
 * page has a print stylesheet that removes the chrome. The fourth check is the first-run card:
 * dismissing it must survive a reload, which needs a real `localStorage` across navigations.
 *
 * Like every spec in this directory it is written but never executed by the scaffolding agent —
 * the browser binaries are a download and downloads are gated (ADR-0012).
 */
import { expect, test } from "@playwright/test";

const GUIDE = "/kilavuz";
const USAGE = "/kilavuz/kullanim";
const TECH = "/kilavuz/teknik";
const STATUS = "/kilavuz/durum";

test.describe("kılavuz", () => {
  test("ana gezinmeden açılır ve kullanım sayfasına yönlendirir", async ({ page }) => {
    await page.goto("/calisma-alani");
    await page.getByRole("link", { name: "Kılavuz" }).click();

    await expect(page).toHaveURL(new RegExp(`${USAGE}$`));
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("Kullanım Kılavuzu");
  });

  test("kök yol kullanım sayfasına yönlenir", async ({ page }) => {
    await page.goto(GUIDE);
    await expect(page).toHaveURL(new RegExp(`${USAGE}$`));
  });

  test("derin bağlantı ilgili bölüme kaydırır", async ({ page }) => {
    await page.goto(`${TECH}#neden-cebir-motoru`);

    const section = page.locator("#neden-cebir-motoru");
    await expect(section).toBeInViewport();
    await expect(section.getByRole("heading", { level: 2 })).toHaveText(
      /Sonucu neden dil modeli değil/,
    );
  });

  test("içindekiler kaydırmayı izler", async ({ page }) => {
    await page.goto(USAGE);
    const toc = page.getByTestId("guide-toc");

    await expect(toc.getByRole("link", { name: "AI-CLS nedir" })).toHaveAttribute(
      "aria-current",
      "location",
    );

    await page.locator("#klavye-kisayollari").scrollIntoViewIfNeeded();
    await expect(toc.getByRole("link", { name: "Klavye kısayolları" })).toHaveAttribute(
      "aria-current",
      "location",
    );
  });

  test("matematik KaTeX ile dizilir, yedek gösterime düşmez", async ({ page }) => {
    await page.goto(USAGE);

    await expect(page.locator("#komut-vermenin-uc-yolu .katex").first()).toBeVisible();
    await expect(page.locator("[data-mathjax-fallback]")).toHaveCount(0);
  });

  test("durum sayfası taklit model uyarısını sayının yanında gösterir", async ({ page }) => {
    await page.goto(STATUS);

    const asr = page.getByTestId("guide-run-asr");
    await expect(asr.getByTestId("metric-asr.wer_tr_v1")).toHaveText("1,1118");
    await expect(asr).toContainText("taklit model");
    await expect(asr).toContainText("kanıt değildir");
  });

  test("komut örneği panoya kopyalanır", async ({ page, context }) => {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.goto(USAGE);

    await page
      .getByRole("button", { name: /Kopyala: x karenin türevini al/ })
      .first()
      .click();

    await expect(page.getByText("Kopyalandı.").first()).toBeVisible();
    const clipboard = await page.evaluate(() => navigator.clipboard.readText());
    expect(clipboard).toBe("x karenin türevini al");
  });

  test("yazdırma görünümünde gezinme ve kopyalama düğmeleri gizlenir", async ({ page }) => {
    await page.goto(TECH);
    await page.emulateMedia({ media: "print" });

    await expect(page.getByRole("navigation", { name: "Kılavuz bölümleri" })).toBeHidden();
    await expect(page.getByRole("button", { name: "Yazdır" })).toBeHidden();
    await expect(page.locator("#boru-hatti")).toBeVisible();
  });

  test("ilk beş dakika kartı bir kez gizlenir ve yenilemeden sonra geri gelmez", async ({
    page,
  }) => {
    await page.goto("/calisma-alani");

    const card = page.getByTestId("first-run-card");
    await expect(card).toBeVisible();
    await card.getByRole("link", { name: "Kılavuzu aç" }).click();
    await expect(page).toHaveURL(new RegExp(`${USAGE}$`));

    await page.goto("/calisma-alani");
    await page.getByRole("button", { name: "Bu kartı gizle" }).click();
    await expect(card).toHaveCount(0);

    await page.reload();
    await expect(page.getByTestId("first-run-card")).toHaveCount(0);
  });
});
