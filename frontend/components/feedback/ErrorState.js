"use client";

import { RefreshCw, TriangleAlert } from "lucide-react";

export function ErrorState({ title = "Impossibile caricare i dati", description, onRetry }) {
  return (
    <div role="alert" className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-red-500/30 bg-red-500/[0.06] px-6 py-14 text-center">
      <span className="grid h-12 w-12 place-items-center rounded-full bg-red-500/10 text-red-400">
        <TriangleAlert aria-hidden="true" size={22} />
      </span>
      <p className="font-semibold">{title}</p>
      {description ? <p className="max-w-sm text-sm text-muted">{description}</p> : null}
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 inline-flex min-h-11 items-center gap-2 rounded-xl border bg-surface px-4 text-sm font-semibold text-ink hover:bg-elevated"
        >
          <RefreshCw aria-hidden="true" size={16} />
          Riprova
        </button>
      ) : null}
    </div>
  );
}
