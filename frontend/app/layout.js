import "./globals.css";
import { AppShell } from "@/components/layout/AppShell";

export const metadata = {
  title: {
    default: "SpaceRiskIntelligence",
    template: "%s | SpaceRiskIntelligence",
  },
  description: "Dashboard sperimentale per l’analisi geospaziale dei rilevamenti satellitari NASA FIRMS.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="it" data-theme="dark" suppressHydrationWarning>
      <body>
        <a
          href="#contenuto-principale"
          className="sr-only z-[100] rounded-lg bg-brand-400 px-4 py-3 font-semibold text-slate-950 focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
        >
          Vai al contenuto
        </a>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
