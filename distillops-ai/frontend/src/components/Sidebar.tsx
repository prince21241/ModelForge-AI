import { BarChart3, FileSearch, GitCompare, LayoutDashboard } from "lucide-react";
import type { ViewId } from "../types";

const ITEMS: { id: ViewId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "extract", label: "Extract Invoice", icon: FileSearch },
  { id: "comparison", label: "Model Comparison", icon: GitCompare },
  { id: "benchmarks", label: "Benchmark Results", icon: BarChart3 },
];

type SidebarProps = {
  view: ViewId;
  onChange: (view: ViewId) => void;
};

export function Sidebar({ view, onChange }: SidebarProps) {
  return (
    <aside className="min-w-0 max-w-full bg-navy text-slate-100 md:min-h-screen">
      <div className="px-5 py-5">
        <p className="text-lg font-semibold tracking-tight">DistillOps AI</p>
        <p className="mt-1 text-xs text-slate-300">Invoice extraction</p>
      </div>
      <nav className="flex min-w-0 max-w-full gap-2 overflow-x-auto px-3 pb-4 md:block md:space-y-1 md:overflow-visible md:px-3" aria-label="Primary">
        {ITEMS.map((item) => {
          const Icon = item.icon;
          const active = view === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onChange(item.id)}
              className={`flex min-w-max items-center gap-2 rounded-lg border-l-2 px-3 py-2 text-left text-sm md:w-full ${
                active ? "border-teal-300 bg-white/15 text-white" : "border-transparent text-slate-300 hover:bg-white/10"
              }`}
              aria-current={active ? "page" : undefined}
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
