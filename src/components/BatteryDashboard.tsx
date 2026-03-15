type BatteryDrone = {
  drone_id: number;
  battery_percentage: number;
};

type BatteryDashboardProps = {
  drones: BatteryDrone[];
};

export default function BatteryDashboard({ drones }: BatteryDashboardProps) {
  return (
    <section className="flex h-full min-h-0 flex-col rounded-2xl border border-slate-200 bg-white/80 p-5 shadow-sm backdrop-blur">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">BatteryDashboard</h2>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {drones.length === 0 ? (
          <p className="text-sm text-slate-500">No telemetry loaded.</p>
        ) : (
          drones.map((drone) => {
            const pct = Math.max(0, Math.min(100, drone.battery_percentage));
            const color =
              pct > 60
                ? "bg-emerald-500"
                : pct > 30
                  ? "bg-amber-500"
                  : "bg-rose-500";

            return (
              <div key={drone.drone_id}>
                <div className="mb-1 flex items-center justify-between text-sm text-slate-700">
                  <span>Drone {drone.drone_id}</span>
                  <span>{pct}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-200">
                  <div
                    className={`h-2 rounded-full ${color}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
