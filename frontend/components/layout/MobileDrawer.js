"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { X } from "lucide-react";
import { APP_NAME, NAV_ITEMS, isNavItemActive } from "@/lib/constants";

// Drawer di navigazione per mobile: dialog accessibile, si chiude con
// Escape, click sul backdrop o selezione di una voce.
export function MobileDrawer({ isOpen, onClose }) {
  const pathname = usePathname();
  const firstLinkRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return undefined;
    firstLinkRef.current?.focus();
    const handleKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 lg:hidden">
      <button type="button" aria-label="Chiudi il menu di navigazione" className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div role="dialog" aria-modal="true" aria-label="Navigazione principale" className="absolute inset-y-0 left-0 flex w-72 max-w-[85vw] flex-col bg-surface shadow-panel">
        <div className="flex h-16 items-center justify-between border-b px-4">
          <p className="font-bold">{APP_NAME}</p>
          <button type="button" onClick={onClose} aria-label="Chiudi navigazione" className="grid h-10 w-10 place-items-center rounded-lg border bg-elevated">
            <X aria-hidden="true" size={18} />
          </button>
        </div>
        <nav className="flex-1 space-y-1 overflow-y-auto p-4" aria-label="Navigazione principale">
          {NAV_ITEMS.map(({ label, href, icon: Icon }, index) => {
            const active = isNavItemActive(pathname, href);
            return (
              <Link
                key={href}
                href={href}
                ref={index === 0 ? firstLinkRef : undefined}
                aria-current={active ? "page" : undefined}
                onClick={onClose}
                className={`flex min-h-11 items-center gap-3 rounded-xl px-3 text-sm font-medium ${
                  active ? "bg-brand-400/12 text-brand-300 ring-1 ring-inset ring-brand-400/20" : "text-muted hover:bg-elevated hover:text-ink"
                }`}
              >
                <Icon aria-hidden="true" size={18} />
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </div>
  );
}
