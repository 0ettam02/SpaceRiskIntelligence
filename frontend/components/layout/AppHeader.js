"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { ChevronRight, CircleUserRound, Menu, Moon, RefreshCw, Sun } from "lucide-react";
import { resolvePageMeta } from "@/lib/constants";
import { useTheme } from "@/hooks/useTheme";
import { emitRefresh } from "@/hooks/useRefreshSignal";
import { formatDateTime } from "@/lib/formatters";

export function AppHeader({ onOpenMobileNav }) {
  const pathname = usePathname();
  const meta = resolvePageMeta(pathname);
  const { theme, toggleTheme } = useTheme();
  const [lastUpdated, setLastUpdated] = useState(() => new Date("2026-07-17T09:00:00Z"));
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = () => {
    setRefreshing(true);
    emitRefresh();
    window.setTimeout(() => {
      setLastUpdated(new Date());
      setRefreshing(false);
    }, 500);
  };

  return (
    <header className="sticky top-0 z-40 border-b bg-canvas/95 backdrop-blur supports-[backdrop-filter]:bg-canvas/80">
      <div className="flex h-20 flex-wrap items-center gap-3 px-4 sm:px-6 lg:px-8">
        <button
          type="button"
          onClick={onOpenMobileNav}
          className="grid h-11 w-11 shrink-0 place-items-center rounded-xl border bg-surface lg:hidden"
          aria-label="Apri navigazione"
        >
          <Menu aria-hidden="true" size={20} />
        </button>

        <div className="min-w-0 flex-1">
          <nav aria-label="Breadcrumb" className="hidden items-center gap-1 text-xs text-muted sm:flex">
            {meta.breadcrumb.map((crumb, index) => (
              <span key={crumb} className="flex items-center gap-1">
                {index > 0 ? <ChevronRight aria-hidden="true" size={12} /> : null}
                {crumb}
              </span>
            ))}
          </nav>
          <p className="truncate text-sm font-semibold sm:text-base">{meta.title}</p>
        </div>

        <span className="hidden items-center gap-2 rounded-full border border-amber-400/30 bg-amber-400/10 px-3 py-1.5 text-xs font-semibold text-amber-300 md:flex">
          Dati sperimentali
        </span>

        <p className="hidden text-xs text-muted xl:block">
          Aggiornato: <span className="font-medium text-ink">{formatDateTime(lastUpdated)}</span>
        </p>

        <button
          type="button"
          onClick={handleRefresh}
          className="grid h-11 w-11 shrink-0 place-items-center rounded-xl border bg-surface text-muted hover:text-ink disabled:opacity-60"
          aria-label="Aggiorna dati"
          disabled={refreshing}
        >
          <RefreshCw aria-hidden="true" size={18} className={refreshing ? "animate-spin" : ""} />
        </button>

        <button
          type="button"
          onClick={toggleTheme}
          className="grid h-11 w-11 shrink-0 place-items-center rounded-xl border bg-surface text-muted hover:text-ink"
          aria-label={theme === "dark" ? "Attiva tema chiaro" : "Attiva tema scuro"}
        >
          {theme === "dark" ? <Sun aria-hidden="true" size={18} /> : <Moon aria-hidden="true" size={18} />}
        </button>

        <button
          type="button"
          className="hidden min-h-11 items-center gap-2 rounded-xl border bg-surface px-3 text-sm font-medium sm:flex"
          aria-label="Menu utente dimostrativo"
        >
          <CircleUserRound aria-hidden="true" size={18} />
          <span className="hidden lg:inline">Utente dimostrativo</span>
        </button>
      </div>
    </header>
  );
}
