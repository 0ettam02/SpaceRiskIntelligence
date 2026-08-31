import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EmptyState } from "@/components/feedback/EmptyState";
import { ErrorState } from "@/components/feedback/ErrorState";
import { LoadingSkeleton } from "@/components/feedback/LoadingSkeleton";

describe("EmptyState", () => {
  it("announces itself as a status region with a message", () => {
    render(<EmptyState title="Nessuna cella selezionata" description="Seleziona un punto sulla mappa." />);
    expect(screen.getByRole("status")).toHaveTextContent("Nessuna cella selezionata");
  });
});

describe("ErrorState", () => {
  it("announces itself as an alert and triggers retry on click", async () => {
    const onRetry = vi.fn();
    render(<ErrorState title="Impossibile caricare i dati" onRetry={onRetry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Impossibile caricare i dati");
    await userEvent.click(screen.getByRole("button", { name: /riprova/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});

describe("LoadingSkeleton", () => {
  it("renders a hidden placeholder that does not confuse assistive tech", () => {
    const { container } = render(<LoadingSkeleton variant="kpis" />);
    expect(container.querySelector("[aria-hidden='true']")).toBeTruthy();
  });
});
