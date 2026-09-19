/**
 * Pilot-study feedback (İP-9, api.md §3.9).
 *
 * The SUS score is a contractual figure, so the questionnaire is tested for the two properties
 * that decide whether the figure exists at all: it cannot be submitted half-filled, and a
 * submission that fails (an expired session, a dropped network) must not cost the participant the
 * ten answers they already gave.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { clearSession } from "@/app/auth";
import { FeedbackForm } from "@/features/feedback/FeedbackForm";
import { FeedbackPage } from "@/features/feedback/FeedbackPage";
import { SusForm } from "@/features/feedback/SusForm";
import { installApiMock, jsonResponse, problemResponse, requestBody } from "./apiMock";

/** Answer every statement with `value`. */
function answerAll(value: number): void {
  const radios = screen.getAllByRole("radio");
  for (let question = 0; question < 10; question += 1) {
    fireEvent.click(radios[question * 5 + (value - 1)] as HTMLElement);
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  clearSession();
});

describe("SUS questionnaire", () => {
  it("refuses to submit before all ten statements are answered", () => {
    installApiMock({});
    render(<SusForm cohort="pilot" />);

    expect(screen.getAllByRole("radio")).toHaveLength(50);
    expect(screen.getByRole("button", { name: "Anketi gönder" })).toBeDisabled();
    expect(screen.getByText("0 / 10 madde yanıtlandı")).toBeInTheDocument();
  });

  it("posts the ten answers and shows the score the server computed", async () => {
    const mock = installApiMock({
      "POST /feedback/sus": (): Response => jsonResponse({ score: 82.5 }, 201),
    });
    const onScored = vi.fn();
    render(<SusForm cohort="pilot-2027" onScored={onScored} />);

    answerAll(4);
    expect(screen.getByText("10 / 10 madde yanıtlandı")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Anketi gönder" }));

    expect(await screen.findByText("SUS puanınız: 82.5")).toBeInTheDocument();
    await waitFor(() => {
      expect(onScored).toHaveBeenCalledWith(82.5);
    });
    expect(await requestBody(mock)).toEqual({
      answers: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
      cohort: "pilot-2027",
    });
  });

  it("keeps the answers when the submission fails", async () => {
    installApiMock({
      "POST /feedback/sus": (): Response =>
        problemResponse(
          "AUTH_INVALID",
          401,
          "Kimlik doğrulama başarısız; lütfen yeniden giriş yapın.",
        ),
      "POST /auth/refresh": (): Response =>
        problemResponse(
          "AUTH_INVALID",
          401,
          "Kimlik doğrulama başarısız; lütfen yeniden giriş yapın.",
        ),
    });
    render(<SusForm cohort="pilot" />);

    answerAll(5);
    fireEvent.click(screen.getByRole("button", { name: "Anketi gönder" }));

    expect(
      await screen.findByText("Kimlik doğrulama başarısız; lütfen yeniden giriş yapın."),
    ).toBeInTheDocument();
    expect(screen.getByText("10 / 10 madde yanıtlandı")).toBeInTheDocument();
    expect(screen.getAllByRole("radio", { checked: true })).toHaveLength(10);
  });
});

describe("free-text feedback", () => {
  it("asks for a description before it sends anything", () => {
    const mock = installApiMock({});
    render(<FeedbackForm page="/calisma-alani" />);

    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));

    expect(screen.getByText("Lütfen bir açıklama yazın.")).toBeInTheDocument();
    expect(mock).not.toHaveBeenCalled();
  });

  it("sends the category, the text and the page it was written on", async () => {
    const mock = installApiMock({
      "POST /feedback": (): Response =>
        jsonResponse(
          { id: "fb-1", category: "usability", text: "Komut çubuğu küçük.", status: "open" },
          201,
        ),
    });
    render(<FeedbackForm page="/calisma-alani" />);

    fireEvent.change(screen.getByLabelText("Kategori"), { target: { value: "critical" } });
    fireEvent.change(screen.getByLabelText(/Açıklama/u), {
      target: { value: "Komut çubuğu küçük." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Gönder" }));

    expect(await screen.findByText("Geri bildiriminiz alındı. Teşekkürler!")).toBeInTheDocument();
    expect(await requestBody(mock)).toEqual({
      category: "critical",
      text: "Komut çubuğu küçük.",
      page: "/calisma-alani",
    });
  });
});

describe("feedback route", () => {
  it("mounts both forms with one heading", () => {
    installApiMock({});
    render(
      <MemoryRouter>
        <FeedbackPage />
      </MemoryRouter>,
    );

    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByText("Kullanılabilirlik Anketi (SUS)")).toBeInTheDocument();
    expect(screen.getByText("Serbest metin geri bildirimi")).toBeInTheDocument();
  });
});
