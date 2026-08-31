import { Inbox } from "lucide-react";

export function EmptyState({ icon: Icon = Inbox, title = "Nessun dato disponibile", description, action }) {
  return (
    <div role="status" className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed bg-elevated/60 px-6 py-14 text-center">
      <span className="grid h-12 w-12 place-items-center rounded-full bg-surface text-muted">
        <Icon aria-hidden="true" size={22} />
      </span>
      <p className="font-semibold">{title}</p>
      {description ? <p className="max-w-sm text-sm text-muted">{description}</p> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}
