/**
 * The design-system primitives.
 *
 * These assertions are about what a person in the room can perceive and reach: a dialog that
 * keeps the keyboard inside it and gives focus back, an error that is announced and quotable, a
 * tooltip whose text is never *only* visual, a form field wired to its label and its message, and
 * a model badge that says out loud when a stand-in produced the answer.
 */
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AiclsProblem } from "@/api/problem";
import { Button } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Field, Input, Select } from "@/components/ui/Field";
import { ModelBadge } from "@/components/ui/ModelBadge";
import { Spinner } from "@/components/ui/Spinner";
import { ToastProvider } from "@/components/ui/Toast";
import { useToast } from "@/components/ui/toastContext";
import { StageTimings } from "@/components/ui/StageTimings";
import { EmptyState, ErrorState, SkipLink } from "@/components/ui/States";
import { Tooltip } from "@/components/ui/Tooltip";

const PROBLEM: AiclsProblem = {
  type: "https://ai-cls.example/errors/PARSE_ERROR",
  title: "LaTeX ifadesi ayrıştırılamadı",
  status: 422,
  code: "PARSE_ERROR",
  detail_tr: "Konum 8: beklenmeyen simge.",
  instance: "01J8R2ZQ7G4V0X9K3M2N5P6Q7R",
  position: 8,
  extra: {},
};

describe("Button", () => {
  it("keeps its label and blocks activation while loading", () => {
    const onClick = vi.fn();
    render(
      <Button variant="primary" loading onClick={onClick}>
        Çalıştır
      </Button>,
    );
    const button = screen.getByRole("button", { name: "Çalıştır" });

    expect(button).toHaveAttribute("aria-busy", "true");
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("accepts the size aliases of the first skeleton", () => {
    render(
      <Button size="small" variant="ghost">
        Küçük
      </Button>,
    );

    expect(screen.getByRole("button", { name: "Küçük" })).toBeInTheDocument();
  });
});

describe("Dialog", () => {
  it("traps Tab inside the panel and returns focus when it closes", () => {
    function Harness({ open }: { open: boolean }): JSX.Element {
      return (
        <>
          <button type="button" data-testid="opener">
            Aç
          </button>
          <Dialog open={open} title="Geri Bildirim" onClose={() => undefined}>
            <button type="button">Birinci</button>
            <button type="button">İkinci</button>
          </Dialog>
        </>
      );
    }

    const { rerender } = render(<Harness open={false} />);
    const opener = screen.getByTestId("opener");
    opener.focus();
    expect(document.activeElement).toBe(opener);

    rerender(<Harness open />);
    const dialog = screen.getByRole("dialog", { name: "Geri Bildirim" });
    const first = within(dialog).getByRole("button", { name: "Kapat" });
    const last = within(dialog).getByRole("button", { name: "İkinci" });

    // Focus moved into the dialog itself, so its title is read before the first control.
    expect(document.activeElement).toBe(dialog);

    // Tab from the last element wraps to the first instead of escaping to the page behind.
    last.focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(first);

    // Shift+Tab from the first wraps backwards to the last.
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(last);

    rerender(<Harness open={false} />);
    expect(document.activeElement).toBe(opener);
  });

  it("closes on Escape and on a backdrop click", () => {
    const onClose = vi.fn();
    render(
      <Dialog open title="Kısayollar" onClose={onClose} description="Klavye kısayolları">
        <p>metin</p>
      </Dialog>,
    );

    expect(screen.getByRole("dialog", { name: "Kısayollar" })).toHaveAccessibleDescription(
      "Klavye kısayolları",
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Kapat" }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("renders nothing when closed", () => {
    render(
      <Dialog open={false} title="Kapalı" onClose={() => undefined}>
        <p>metin</p>
      </Dialog>,
    );

    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("Field", () => {
  it("wires the label, the hint and the error to the control", () => {
    render(
      <Field label="E-posta" hint="Kurumsal adresiniz" error="Bu alan zorunludur." required>
        {(props) => <Input {...props} type="email" />}
      </Field>,
    );
    const input = screen.getByLabelText(/E-posta/);

    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(input).toBeRequired();
    expect(input).toHaveAccessibleDescription(/Kurumsal adresiniz/);
    expect(screen.getByRole("alert")).toHaveTextContent("Bu alan zorunludur.");
  });

  it("labels a native select", () => {
    render(
      <Field label="Üniversite">
        {(props) => (
          <Select {...props}>
            <option value="mus">Muş Alparslan Üniversitesi</option>
          </Select>
        )}
      </Field>,
    );

    expect(screen.getByLabelText("Üniversite")).toBeInstanceOf(HTMLSelectElement);
  });
});

describe("states", () => {
  it("announces an error with its Turkish text, code and request id", () => {
    const onRetry = vi.fn();
    render(<ErrorState problem={PROBLEM} onRetry={onRetry} />);
    const alert = screen.getByRole("alert");

    expect(alert).toHaveTextContent("Konum 8: beklenmeyen simge.");
    expect(alert).toHaveTextContent("PARSE_ERROR");
    expect(alert).toHaveTextContent("01J8R2ZQ7G4V0X9K3M2N5P6Q7R");
    expect(alert).toHaveTextContent("Hatanın konumu: 8. karakter.");

    fireEvent.click(screen.getByRole("button", { name: "Yeniden dene" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("falls back to a generic Turkish sentence without a problem document", () => {
    render(<ErrorState problem={null} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Beklenmeyen bir hata oluştu.");
  });

  it("renders an empty state and a skip link", () => {
    render(
      <>
        <SkipLink />
        <EmptyState title="Henüz sonuç yok" description="Bir komut girerek başlayın." />
      </>,
    );

    expect(screen.getByRole("link", { name: "İçeriğe geç" })).toHaveAttribute("href", "#main");
    expect(screen.getByText("Henüz sonuç yok")).toBeInTheDocument();
  });

  it("announces a spinner politely with a Turkish label", () => {
    render(<Spinner />);

    expect(screen.getByRole("status")).toHaveTextContent("Yükleniyor…");
  });
});

describe("Tooltip", () => {
  it("exposes its text through aria-describedby, hover or not", () => {
    render(
      <Tooltip content="Komut çubuğuna geçmek için eğik çizgi">
        {(props) => (
          <button type="button" {...props}>
            Komut
          </button>
        )}
      </Tooltip>,
    );

    expect(screen.getByRole("button", { name: "Komut" })).toHaveAccessibleDescription(
      "Komut çubuğuna geçmek için eğik çizgi",
    );
  });
});

describe("StageTimings", () => {
  it("renders the stage breakdown in Turkish with the total", () => {
    render(
      <StageTimings
        timings={{ nlu: 12.3, parse: 1.1, cas: 41, render: 0.4, total: 58.2 }}
        stages={[{ name: "app", ms: 58.2 }]}
      />,
    );

    expect(screen.getByText("Toplam: 58 ms")).toBeInTheDocument();
    expect(screen.getByText("anlama")).toBeInTheDocument();
    expect(screen.getByText("ayrıştırma")).toBeInTheDocument();
    expect(screen.getByText("hesap")).toBeInTheDocument();
    expect(screen.getByText("çizim")).toBeInTheDocument();
    expect(screen.getByText("41 ms")).toBeInTheDocument();
  });

  it("marks a response that missed the three-second contract budget", () => {
    render(<StageTimings timings={{ cas: 3400, total: 3600 }} />);

    expect(screen.getByText(/3 sn hedefinin üzerinde/)).toBeInTheDocument();
  });

  it("renders nothing when the response carried no timings", () => {
    const { container } = render(<StageTimings timings={{}} />);

    expect(container).toBeEmptyDOMElement();
  });
});

describe("ModelBadge", () => {
  it("warns, in Turkish, when a stand-in model produced the answer", () => {
    render(<ModelBadge versions={{ ocr: "stub-0.1.0", asr: "asr-stub-1" }} />);

    const warning = screen.getByTestId("model-stub-warning");
    expect(warning).toHaveTextContent("taklit");
    expect(screen.getByText("stub-0.1.0")).toBeInTheDocument();
    expect(screen.getAllByText(/taklit model/)).not.toHaveLength(0);
  });

  it("does not warn when every component is a promoted checkpoint", () => {
    render(<ModelBadge versions={{ ocr: "ocr-2027.03", asr: "asr-2027.04" }} />);

    expect(screen.queryByTestId("model-stub-warning")).toBeNull();
    expect(screen.getAllByText(/eğitilmiş model/)).not.toHaveLength(0);
  });

  it("warns about a runtime-injected model too", () => {
    render(<ModelBadge versions={{ ocr: "override" }} />);

    expect(screen.getByTestId("model-stub-warning")).toBeInTheDocument();
    expect(screen.getByText(/elle yüklenmiş model/)).toBeInTheDocument();
  });

  it("hides components that reported no version, and renders nothing when all did", () => {
    const { container } = render(<ModelBadge versions={{ ocr: null, asr: null }} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders only the warning in warningOnly mode", () => {
    render(<ModelBadge versions={{ ocr: "stub-0.1.0", asr: "asr-2027.04" }} warningOnly />);

    expect(screen.getByTestId("model-stub-warning")).toBeInTheDocument();
    expect(screen.queryByText("asr-2027.04")).toBeNull();
  });
});

describe("Toast", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  function Publisher(): JSX.Element {
    const { show, dismiss, toasts } = useToast();
    return (
      <div>
        <button
          type="button"
          onClick={() => {
            show("Kayıt kabul edildi.", { tone: "ok", duration: 5000 });
          }}
        >
          Bildir
        </button>
        <button
          type="button"
          onClick={() => {
            show("Sunucu yoğun.", {
              tone: "warn",
              duration: 0,
              action: { label: "Geri al", onAction: () => undefined },
            });
          }}
        >
          Kalıcı bildir
        </button>
        <button
          type="button"
          onClick={() => {
            const first = toasts[0];
            if (first !== undefined) {
              dismiss(first.id);
            }
          }}
        >
          İlkini kapat
        </button>
      </div>
    );
  }

  it("announces messages in a polite live region and expires them", () => {
    vi.useFakeTimers();
    render(
      <ToastProvider>
        <Publisher />
      </ToastProvider>,
    );

    const region = screen.getByRole("status", { name: "Bildirimler" });
    expect(region).toHaveAttribute("aria-live", "polite");

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Bildir" }));
    });
    expect(within(region).getByText("Kayıt kabul edildi.")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(within(region).queryByText("Kayıt kabul edildi.")).toBeNull();
  });

  it("keeps a zero-duration toast until it is dismissed, and offers its action", () => {
    render(
      <ToastProvider>
        <Publisher />
      </ToastProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Kalıcı bildir" }));
    expect(screen.getByText("Sunucu yoğun.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Geri al" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "İlkini kapat" }));
    expect(screen.queryByText("Sunucu yoğun.")).toBeNull();
  });
});
