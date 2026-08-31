"use client";

import { useState } from "react";
import { Moon, Sun } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { MethodologyAlert } from "@/components/feedback/MethodologyAlert";
import { useTheme } from "@/hooks/useTheme";
import { API_BASE_URL, APP_NAME, DATA_SOURCE } from "@/lib/constants";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [emailDigest, setEmailDigest] = useState(false);

  return (
    <div className="max-w-2xl space-y-6">
      <PageHeader title="Impostazioni" description="Preferenze dell'interfaccia e informazioni sulla sorgente dati di questa demo." />

      <section className="rounded-2xl border bg-surface p-5 shadow-panel">
        <h2 className="font-semibold">Aspetto</h2>
        <p className="mt-1 text-sm text-muted">La preferenza scelta viene salvata su questo dispositivo.</p>
        <div role="radiogroup" aria-label="Tema dell'interfaccia" className="mt-4 grid grid-cols-2 gap-3">
          <button
            type="button"
            role="radio"
            aria-checked={theme === "dark"}
            onClick={() => setTheme("dark")}
            className={`flex min-h-16 items-center justify-center gap-2 rounded-xl border text-sm font-semibold ${
              theme === "dark" ? "border-brand-400/40 bg-brand-400/10 text-brand-300" : "bg-elevated text-muted"
            }`}
          >
            <Moon aria-hidden="true" size={18} />
            Scuro
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={theme === "light"}
            onClick={() => setTheme("light")}
            className={`flex min-h-16 items-center justify-center gap-2 rounded-xl border text-sm font-semibold ${
              theme === "light" ? "border-brand-400/40 bg-brand-400/10 text-brand-300" : "bg-elevated text-muted"
            }`}
          >
            <Sun aria-hidden="true" size={18} />
            Chiaro
          </button>
        </div>
      </section>

      <section className="rounded-2xl border bg-surface p-5 shadow-panel">
        <h2 className="font-semibold">Sorgente dati</h2>
        <dl className="mt-3 space-y-2 text-sm">
          <div className="flex justify-between gap-2">
            <dt className="text-muted">Modalità attiva</dt>
            <dd className="font-mono font-semibold">{DATA_SOURCE}</dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt className="text-muted">Endpoint API</dt>
            <dd className="font-mono text-xs text-muted">{API_BASE_URL || "non configurato"}</dd>
          </div>
        </dl>
        <p className="mt-3 text-xs leading-5 text-muted">
          Questa build di {APP_NAME} utilizza dati mock generati localmente. Impostare <code>NEXT_PUBLIC_DATA_SOURCE=api</code> e{" "}
          <code>NEXT_PUBLIC_API_BASE_URL</code> quando sarà disponibile un backend reale, senza modificare i componenti.
        </p>
      </section>

      <section className="rounded-2xl border bg-surface p-5 shadow-panel">
        <h2 className="font-semibold">Notifiche (dimostrativo)</h2>
        <label className="mt-3 flex items-center justify-between gap-3 text-sm">
          <span>
            Riepilogo settimanale via email
            <span className="ml-2 rounded-full border bg-elevated px-2 py-0.5 text-[10px] uppercase tracking-wide text-muted">Demo, non attivo</span>
          </span>
          <input
            type="checkbox"
            checked={emailDigest}
            onChange={(event) => setEmailDigest(event.target.checked)}
            className="h-5 w-5 accent-brand-500"
            aria-describedby="email-digest-note"
          />
        </label>
        <p id="email-digest-note" className="mt-2 text-xs text-muted">
          Nessuna email viene realmente inviata: questa demo non dispone di un servizio di notifica operativo.
        </p>
      </section>

      <section className="rounded-2xl border bg-surface p-5 shadow-panel">
        <h2 className="font-semibold">Account dimostrativo</h2>
        <p className="mt-2 text-sm text-muted">
          {APP_NAME} non implementa autenticazione reale. L&apos;utente mostrato nell&apos;header è un profilo dimostrativo
          senza accesso a dati riservati.
        </p>
      </section>

      <MethodologyAlert
        title="Promemoria"
        items={["Le impostazioni di questa pagina riguardano solo l'interfaccia dimostrativa e non modificano alcun sistema esterno."]}
      />
    </div>
  );
}
