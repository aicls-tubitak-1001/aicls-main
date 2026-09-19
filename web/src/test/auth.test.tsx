/**
 * Identity screens: sign-in, sign-up, the account view and the session-expiry notice.
 *
 * The rule under test is that the *backend* owns the Turkish wording of a refusal. The client only
 * says "this box is empty"; everything else — a refused login, an address that is already
 * registered — is rendered from the problem document, above the form or under the field the API
 * attributed it to.
 */
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { UserOut } from "@/api/types";
import { clearSession, sessionStore, setAccessToken, setUser } from "@/app/auth";
import { AccountPage } from "@/features/auth/AccountPage";
import { LoginPage } from "@/features/auth/LoginPage";
import { RegisterPage } from "@/features/auth/RegisterPage";
import { SessionNotice } from "@/features/auth/SessionNotice";
import { installApiMock, jsonResponse, problemResponse, requestBody } from "./apiMock";

const LECTURER: UserOut = {
  id: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  email: "hoca@alparslan.edu.tr",
  display_name: "Dr. Öğr. Üyesi",
  role: "instructor",
  created_at: "2026-09-01T08:00:00Z",
};

function renderAt(element: JSX.Element): void {
  render(<MemoryRouter>{element}</MemoryRouter>);
}

afterEach(() => {
  vi.unstubAllGlobals();
  clearSession();
});

describe("sign-in", () => {
  it("names the empty fields in Turkish without calling the API", () => {
    const mock = installApiMock({});
    renderAt(<LoginPage />);

    fireEvent.click(screen.getByRole("button", { name: "Giriş yap" }));

    expect(screen.getByText("E-posta adresinizi yazın.")).toBeInTheDocument();
    expect(screen.getByText("Parolanızı yazın.")).toBeInTheDocument();
    expect(mock).not.toHaveBeenCalled();
  });

  it("shows the backend's sentence when the credentials are refused", async () => {
    installApiMock({
      "POST /auth/login": (): Response =>
        problemResponse(
          "AUTH_INVALID",
          401,
          "Kimlik doğrulama başarısız; lütfen yeniden giriş yapın.",
        ),
    });
    renderAt(<LoginPage />);

    fireEvent.change(screen.getByLabelText(/E-posta/u), {
      target: { value: "hoca@alparslan.edu.tr" },
    });
    fireEvent.change(screen.getByLabelText(/Parola/u), { target: { value: "yanlis-parola" } });
    fireEvent.click(screen.getByRole("button", { name: "Giriş yap" }));

    expect(
      await screen.findByText("Kimlik doğrulama başarısız; lütfen yeniden giriş yapın."),
    ).toBeInTheDocument();
    expect(sessionStore.getState().status).toBe("anonymous");
  });

  it("exchanges the credentials for a session and loads the profile", async () => {
    const mock = installApiMock({
      "POST /auth/login": (): Response =>
        jsonResponse({ access_token: "access-1", expires_in: 900, token_type: "bearer" }),
      "GET /users/me": LECTURER,
    });
    renderAt(<LoginPage />);

    fireEvent.change(screen.getByLabelText(/E-posta/u), {
      target: { value: "hoca@alparslan.edu.tr" },
    });
    fireEvent.change(screen.getByLabelText(/Parola/u), { target: { value: "dogru-parola" } });
    fireEvent.click(screen.getByRole("button", { name: "Giriş yap" }));

    await waitFor(() => {
      expect(sessionStore.getState().role).toBe("instructor");
    });
    expect(await requestBody(mock)).toEqual({
      email: "hoca@alparslan.edu.tr",
      password: "dogru-parola",
    });
  });
});

describe("sign-up", () => {
  it("puts a field-level problem under the field the API named", async () => {
    installApiMock({
      "POST /auth/register": (): Response =>
        new Response(
          JSON.stringify({
            type: "https://ai-cls.example/errors/VALIDATION_ERROR",
            title: "İstek doğrulanamadı",
            status: 422,
            code: "VALIDATION_ERROR",
            detail_tr: "İstek doğrulanamadı; lütfen alanları kontrol edin.",
            errors: [{ loc: ["body", "email"], msg: "Bu e-posta adresi zaten kayıtlı." }],
          }),
          { status: 422, headers: { "content-type": "application/problem+json" } },
        ),
    });
    renderAt(<RegisterPage />);

    fireEvent.change(screen.getByLabelText(/Görünen ad/u), { target: { value: "Ayşe Yılmaz" } });
    fireEvent.change(screen.getByLabelText(/E-posta/u), { target: { value: "a@ornek.edu.tr" } });
    fireEvent.change(screen.getByLabelText(/^Parola/u), { target: { value: "cok-uzun-parola" } });
    fireEvent.click(screen.getByRole("button", { name: "Kayıt ol" }));

    expect(await screen.findByText("Bu e-posta adresi zaten kayıtlı.")).toBeInTheDocument();
  });

  it("refuses a password shorter than the API's minimum before sending it", () => {
    const mock = installApiMock({});
    renderAt(<RegisterPage />);

    fireEvent.change(screen.getByLabelText(/Görünen ad/u), { target: { value: "Ayşe" } });
    fireEvent.change(screen.getByLabelText(/E-posta/u), { target: { value: "a@ornek.edu.tr" } });
    fireEvent.change(screen.getByLabelText(/^Parola/u), { target: { value: "kisa" } });
    fireEvent.click(screen.getByRole("button", { name: "Kayıt ol" }));

    expect(screen.getByText("Parola en az 10 karakter olmalı.")).toBeInTheDocument();
    expect(mock).not.toHaveBeenCalled();
  });
});

describe("session state", () => {
  it("shows the account with its role and signs out", async () => {
    installApiMock({ "POST /auth/logout": (): Response => new Response(null, { status: 204 }) });
    setAccessToken("access-1", 900);
    setUser(LECTURER);
    renderAt(<AccountPage />);

    expect(screen.getByText("hoca@alparslan.edu.tr")).toBeInTheDocument();
    expect(screen.getAllByText("Öğretim elemanı").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Çıkış yap" }));

    expect(await screen.findByText("Çıkış yapıldı.")).toBeInTheDocument();
    expect(sessionStore.getState().accessToken).toBeNull();
  });

  it("says the session expired without taking the screen away", () => {
    installApiMock({});
    clearSession({
      type: "https://ai-cls.example/errors/AUTH_INVALID",
      title: "Kimlik doğrulama başarısız",
      status: 401,
      code: "AUTH_INVALID",
      detail_tr: "Oturumunuzun süresi doldu. Lütfen yeniden giriş yapın.",
      instance: null,
      extra: {},
    });
    renderAt(<SessionNotice />);

    expect(screen.getByText("Oturum sona erdi")).toBeInTheDocument();
    expect(
      screen.getByText("Oturumunuzun süresi doldu. Lütfen yeniden giriş yapın."),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Giriş yap" })).toBeInTheDocument();
  });

  it("stays out of the way while the session is healthy", () => {
    installApiMock({});
    setAccessToken("access-1", 900);
    const { container } = render(
      <MemoryRouter>
        <SessionNotice />
      </MemoryRouter>,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
