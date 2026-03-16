type MissionLogProps = {
  logs: string[];
  mode?: "ai" | "fallback" | null;
};

export default function MissionLog({ logs, mode = null }: MissionLogProps) {
  const modeLabel = mode === "fallback" ? "Fallback Mode" : "AI Mode";

  return (
    <section className="flex h-full min-h-0 flex-col rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-900">MissionLog</h2>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
            mode === "fallback"
              ? "bg-amber-100 text-amber-800"
              : "bg-emerald-100 text-emerald-800"
          }`}
        >
          {modeLabel}
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto rounded-lg bg-slate-900 p-3 font-mono text-xs text-emerald-300">
        {logs.length === 0 ? (
          <p className="text-slate-400">No mission events yet.</p>
        ) : (
          <ul className="space-y-2">
            {logs.map((line, idx) => (
              <li key={`${idx}-${line.slice(0, 12)}`}>{line}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
