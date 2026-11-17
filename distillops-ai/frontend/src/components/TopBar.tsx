type TopBarProps = {
  connected: boolean;
  modelLabel: string;
};

export function TopBar({ connected, modelLabel }: TopBarProps) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-4 md:px-8">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-blue-700">DistillOps AI</p>
        <h1 className="text-xl font-semibold text-slate-900">Invoice workspace</h1>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-1">
          <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-500" : "bg-rose-500"}`} aria-hidden="true" />
          <span>{connected ? "Backend connected" : "Backend unavailable"}</span>
        </span>
        <span className="rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-blue-800">
          Active model: {modelLabel}
        </span>
      </div>
    </header>
  );
}
