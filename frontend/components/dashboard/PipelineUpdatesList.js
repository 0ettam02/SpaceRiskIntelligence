import { GitCommitVertical } from "lucide-react";
import { formatDate } from "@/lib/formatters";

export function PipelineUpdatesList({ updates }) {
  return (
    <ol className="space-y-4">
      {updates.map((update, index) => (
        <li key={update.date + update.title} className="relative flex gap-3 pl-1">
          <div className="flex flex-col items-center">
            <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-brand-400/10 text-brand-300">
              <GitCommitVertical aria-hidden="true" size={14} />
            </span>
            {index < updates.length - 1 ? <span className="mt-1 w-px flex-1 bg-line" aria-hidden="true" /> : null}
          </div>
          <div className="pb-4">
            <p className="text-xs text-muted">{formatDate(update.date)}</p>
            <p className="font-semibold">{update.title}</p>
            <p className="mt-1 text-sm leading-6 text-muted">{update.description}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}
