type MissionLogProps = {
  logs: string[];
};

export default function MissionLog({ logs }: MissionLogProps) {
  return (
    <section className="h-full rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Mission Log</h2>
      <div className="h-[calc(100%-2rem)] overflow-y-auto rounded-lg bg-slate-900 p-3 font-mono text-xs text-emerald-300">
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
