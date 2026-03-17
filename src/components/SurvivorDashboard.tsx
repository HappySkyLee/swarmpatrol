type Survivor = {
  x: number;
  y: number;
  detected_round: number;
  detected_elapsed_minutes: number;
  status: "unconfirmed" | "confirmed";
  route_ready: boolean;
};

type SurvivorDashboardProps = {
  survivors: Survivor[];
};

export default function SurvivorDashboard({ survivors }: SurvivorDashboardProps) {
  const confirmedCount = survivors.filter((item) => item.status === "confirmed").length;
  const unconfirmedCount = survivors.filter((item) => item.status === "unconfirmed").length;

  return (
    <section className="h-full rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">Survivor Dashboard</h2>
        <div className="flex items-center gap-2 text-xs font-semibold">
          <span className="rounded-full bg-orange-100 px-2 py-0.5 text-orange-700">
            {unconfirmedCount} unconfirmed
          </span>
          <span className="rounded-full bg-red-100 px-2 py-0.5 text-red-700">
            {confirmedCount} confirmed
          </span>
        </div>
      </div>

      <div className="h-[calc(100%-2rem)] overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3">
        {survivors.length === 0 ? (
          <p className="text-sm text-slate-500">No unconfirmed or confirmed survivors yet.</p>
        ) : (
          <ul className="space-y-2">
            {survivors.map((survivor, index) => (
              <li
                key={`${index}-${survivor.x}-${survivor.y}`}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2"
              >
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold text-slate-900">
                    Coordinate ({survivor.x}, {survivor.y})
                  </p>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      survivor.status === "confirmed"
                        ? "bg-red-100 text-red-700"
                        : "bg-orange-100 text-orange-700"
                    }`}
                  >
                    {survivor.status === "confirmed" ? "Confirmed Survivor" : "Unconfirmed"}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-600">
                  Detected at round {survivor.detected_round} ({survivor.detected_elapsed_minutes} min)
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
