"use client";

import { RotateCcw, SlidersHorizontal, X } from "lucide-react";

// Contenitore riutilizzabile per i filtri di una pagina: card fissa su
// desktop, drawer a comparsa su mobile (isOpen/onClose gestiti dalla pagina
// chiamante tramite useDisclosure).
export function FilterPanel({ title = "Filtri", onReset, isOpen, onClose, children }) {
  return (
    <>
      <div className="hidden rounded-2xl border bg-surface p-4 shadow-panel lg:block">
        <FilterPanelHeader title={title} onReset={onReset} />
        <div className="mt-4 space-y-4">{children}</div>
      </div>

      {isOpen ? (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button type="button" aria-label="Chiudi il pannello dei filtri" className="absolute inset-0 bg-black/50" onClick={onClose} />
          <div className="absolute inset-x-0 bottom-0 max-h-[85vh] overflow-y-auto rounded-t-2xl border-t bg-surface p-4 shadow-panel">
            <div className="mb-2 flex items-center justify-between">
              <FilterPanelHeader title={title} onReset={onReset} />
              <button type="button" onClick={onClose} aria-label="Chiudi filtri" className="grid h-10 w-10 place-items-center rounded-lg border bg-elevated">
                <X aria-hidden="true" size={18} />
              </button>
            </div>
            <div className="space-y-4 pb-4">{children}</div>
          </div>
        </div>
      ) : null}
    </>
  );
}

function FilterPanelHeader({ title, onReset }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <p className="flex items-center gap-2 font-semibold">
        <SlidersHorizontal aria-hidden="true" size={16} className="text-brand-300" />
        {title}
      </p>
      {onReset ? (
        <button type="button" onClick={onReset} className="inline-flex min-h-9 items-center gap-1.5 rounded-lg px-2 text-xs font-semibold text-muted hover:text-ink">
          <RotateCcw aria-hidden="true" size={14} />
          Reimposta
        </button>
      ) : null}
    </div>
  );
}
