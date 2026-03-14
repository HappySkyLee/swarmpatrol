"use client";

import { useMemo, useState } from "react";

import BatteryDashboard from "../components/BatteryDashboard";
import GridMap from "../components/GridMap";
import MissionLog from "../components/MissionLog";

type Telemetry = {
  drone_id: number;
  x: number;
  y: number;
  battery_percentage: number;
  mode: string;
};

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export default function Home() {
  const [missionStarted, setMissionStarted] = useState(false);
  const [missionLoading, setMissionLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry[]>([]);

  const statusLabel = useMemo(() => {
    if (missionLoading) return "Starting mission...";
    return missionStarted ? "Mission active" : "Mission idle";
  }, [missionLoading, missionStarted]);

  const fetchTelemetry = async () => {
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/drone_telemetry`);
      if (!response.ok) {
        throw new Error(`Telemetry request failed (${response.status})`);
      }
      const data = (await response.json()) as Telemetry[];
      setTelemetry(data);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Telemetry failed";
      setLogs((prev) => [`[error] ${message}`, ...prev]);
    }
  };

  const startMission = async () => {
    if (missionLoading) return;
    setMissionLoading(true);

    try {
      const response = await fetch(`${BACKEND_BASE_URL}/start_mission`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`Start mission failed (${response.status})`);
      }

      const payload = (await response.json()) as {
        message?: string;
        agent_output?: string;
      };

      setMissionStarted(true);
      setLogs((prev) => {
        const next = [
          `[system] ${payload.message ?? "Mission started."}`,
          `[agent] ${payload.agent_output ?? "Planning started."}`,
          ...prev,
        ];
        return next.slice(0, 50);
      });
      await fetchTelemetry();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown start failure";
      setLogs((prev) => [`[error] ${message}`, ...prev]);
    } finally {
      setMissionLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-100 via-sky-50 to-cyan-100 p-4 md:p-6">
      <div className="mx-auto flex w-full max-w-[1500px] items-center justify-between pb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 md:text-3xl">Swarm Mission Dashboard</h1>
          <p className="text-sm text-slate-600">Command Agent Base: (20,20)</p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-slate-700">{statusLabel}</span>
          <button
            type="button"
            onClick={startMission}
            disabled={missionLoading}
            className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {missionLoading ? "Starting..." : "Start Mission"}
          </button>
        </div>
      </div>

      <section className="grid min-h-[calc(100vh-7rem)] grid-cols-1 gap-4 lg:grid-cols-5">
        <div className="lg:col-span-3 lg:h-full">
          <GridMap missionStarted={missionStarted} />
        </div>

        <div className="grid gap-4 lg:col-span-2 lg:grid-rows-2">
          <div className="min-h-[260px] lg:min-h-0">
            <MissionLog logs={logs} />
          </div>
          <div className="min-h-[260px] lg:min-h-0">
            <BatteryDashboard drones={telemetry} />
          </div>
        </div>
      </section>
    </main>
  );
}
