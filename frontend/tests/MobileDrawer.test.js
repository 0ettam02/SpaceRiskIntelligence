import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MobileDrawer } from "@/components/layout/MobileDrawer";

vi.mock("next/navigation", () => ({ usePathname: () => "/" }));

describe("MobileDrawer", () => {
  it("renders nothing when closed", () => {
    render(<MobileDrawer isOpen={false} onClose={() => {}} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows the navigation dialog when open and closes on request", async () => {
    const onClose = vi.fn();
    render(<MobileDrawer isOpen onClose={onClose} />);
    expect(screen.getByRole("dialog", { name: /navigazione principale/i })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /chiudi navigazione/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("closes when a navigation link is selected", async () => {
    const onClose = vi.fn();
    render(<MobileDrawer isOpen onClose={onClose} />);
    await userEvent.click(screen.getByRole("link", { name: /mappa globale/i }));
    expect(onClose).toHaveBeenCalled();
  });
});
