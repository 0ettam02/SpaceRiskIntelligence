function Bar({ className = "" }) {
  return <div className={`animate-pulse rounded-lg bg-elevated ${className}`} />;
}

export function LoadingSkeleton({ variant = "card", rows = 3 }) {
  if (variant === "kpis") {
    return (
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6" aria-hidden="true">
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={index} className="space-y-4 rounded-2xl border bg-surface p-4">
            <Bar className="h-3 w-2/3" />
            <Bar className="h-7 w-1/2" />
            <Bar className="h-3 w-3/4" />
          </div>
        ))}
      </div>
    );
  }

  if (variant === "chart") {
    return (
      <div className="rounded-2xl border bg-surface p-5" aria-hidden="true">
        <Bar className="mb-4 h-4 w-1/3" />
        <Bar className="h-64 w-full" />
      </div>
    );
  }

  if (variant === "table") {
    return (
      <div className="space-y-2 rounded-2xl border bg-surface p-4" aria-hidden="true">
        {Array.from({ length: rows }).map((_, index) => (
          <Bar key={index} className="h-10 w-full" />
        ))}
      </div>
    );
  }

  if (variant === "map") {
    return (
      <div className="rounded-2xl border bg-surface p-2" aria-hidden="true">
        <Bar className="h-[28rem] w-full" />
      </div>
    );
  }

  return (
    <div className="space-y-3 rounded-2xl border bg-surface p-5" aria-hidden="true">
      {Array.from({ length: rows }).map((_, index) => (
        <Bar key={index} className="h-4 w-full" />
      ))}
    </div>
  );
}
