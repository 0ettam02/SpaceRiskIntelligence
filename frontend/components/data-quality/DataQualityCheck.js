import { StatusBadge } from "@/components/feedback/StatusBadge";

export function DataQualityCheck({ checks }) {
  return (
    <div className="overflow-hidden rounded-2xl border bg-surface shadow-panel">
      <div className="hidden lg:block">
        <table className="w-full border-collapse text-sm">
          <caption className="sr-only">Tabella dei controlli di qualità dei dati, con esito e dettaglio per ciascun controllo.</caption>
          <thead>
            <tr className="border-b text-left text-xs uppercase tracking-wide text-muted">
              <th scope="col" className="px-4 py-3 font-semibold">
                Controllo
              </th>
              <th scope="col" className="px-4 py-3 font-semibold">
                Esito
              </th>
              <th scope="col" className="px-4 py-3 font-semibold">
                Dettaglio
              </th>
            </tr>
          </thead>
          <tbody>
            {checks.map((check) => (
              <tr key={check.id} className="border-b last:border-0 align-top">
                <th scope="row" className="whitespace-nowrap px-4 py-3 text-left font-medium">
                  {check.label}
                </th>
                <td className="px-4 py-3">
                  <StatusBadge status={check.status} />
                </td>
                <td className="px-4 py-3 text-muted">{check.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ul className="divide-y lg:hidden">
        {checks.map((check) => (
          <li key={check.id} className="p-4">
            <div className="flex items-center justify-between gap-2">
              <p className="font-medium">{check.label}</p>
              <StatusBadge status={check.status} />
            </div>
            <p className="mt-2 text-sm text-muted">{check.detail}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
