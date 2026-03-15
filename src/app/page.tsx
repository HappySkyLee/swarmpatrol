"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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

type HealthResponse = {
  orchestrator_ready?: boolean;
  orchestrator_error?: string | null;
};

export default function Home() {
  const [missionStarted, setMissionStarted] = useState(false);
  const [missionLoading, setMissionLoading] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (missionLoading) return "Connecting...";
    return missionStarted ? "Mission active" : "Mission idle";
  }, [missionLoading, missionStarted]);

  const fetchTelemetry = async (logErrors = false) => {
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/drone_telemetry`);
      if (!response.ok) {
        throw new Error(`Telemetry request failed (${response.status})`);
      }
      const data = (await response.json()) as Telemetry[];
      setTelemetry(data);
    } catch (error) {
      if (logErrors) {
        const message = error instanceof Error ? error.message : "Telemetry failed";
        setLogs((prev) => [`[error] ${message}`, ...prev]);
      }
    }
  };

  useEffect(() => {
    void fetchTelemetry();
    const timer = window.setInterval(() => {
      void fetchTelemetry();
    }, 2000);

    return () => {
      window.clearInterval(timer);
    };
  }, []);

  const checkMissionReadiness = async (): Promise<{ ok: boolean; reason?: string }> => {
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/health`);
      if (!response.ok) {
        return {
          ok: false,
          reason: `Health check failed (${response.status}).`,
        };
      }

      const health = (await response.json()) as HealthResponse;
      if (health.orchestrator_ready === false) {
        return {
          ok: false,
          reason:
            health.orchestrator_error ??
            "Mission orchestrator is unavailable. Ensure backend dependencies and OPENAI_API_KEY are configured.",
        };
      }

      return { ok: true };
    } catch {
      return {
        ok: false,
        reason: "Cannot reach backend health endpoint. Ensure backend is running.",
      };
    }
  };

  const startMission = async () => {
    if (missionLoading || missionStarted) return;
    setMissionLoading(true);

    const addLog = (line: string) =>
      setLogs((prev) => [line, ...prev].slice(0, 100));

    const readiness = await checkMissionReadiness();
    if (!readiness.ok) {
      addLog(`[error] ${readiness.reason}`);
      setMissionLoading(false);
      setMissionStarted(false);
      return;
    }

    let connected = false;
    const es = new EventSource(`${BACKEND_BASE_URL}/mission_log`);
    eventSourceRef.current = es;

    es.onopen = () => {
      connected = true;
      setMissionStarted(true);
      setMissionLoading(false);
      addLog("[system] Mission stream connected.");
      void fetchTelemetry(true);
    };

    es.addEventListener("status", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as { message?: string };
        addLog(`[status] ${data.message ?? e.data}`);
      } catch { addLog(`[status] ${e.data as string}`); }
    });

    es.addEventListener("thought", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as { message?: string };
        addLog(`[thought] ${data.message ?? e.data}`);
      } catch { addLog(`[thought] ${e.data as string}`); }
    });

    es.addEventListener("observation", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as { tool?: string; observation?: unknown };
        const obs = JSON.stringify(data.observation ?? {}).slice(0, 120);
        addLog(`[obs:${data.tool ?? "?"}] ${obs}`);
      } catch { addLog(`[obs] ${e.data as string}`); }
    });

    es.addEventListener("final", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as { output?: string };
        addLog(`[result] ${data.output ?? "Mission complete."}`);
      } catch { addLog("[result] Mission complete."); }
      es.close();
      eventSourceRef.current = null;
      void fetchTelemetry(true);
    });

    es.onerror = () => {
      if (!connected) {
        addLog(
          "[error] Cannot connect to mission stream. Check /health for orchestrator_error details."
        );
        setMissionLoading(false);
        setMissionStarted(false);
      } else {
        addLog("[status] Mission stream closed.");
      }
      es.close();
      eventSourceRef.current = null;
    };
  };

  return (
    <main className="h-screen overflow-hidden bg-gradient-to-br from-slate-100 via-sky-50 to-cyan-100 p-3 md:p-4">
      <div className="mx-auto flex w-full max-w-[1400px] items-center justify-between pb-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 md:text-3xl">Swarm Mission Dashboard</h1>
          <p className="text-sm text-slate-600">Command Agent Base: (10,10)</p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-slate-700">{statusLabel}</span>
          <button
            type="button"
            onClick={() => {
              void startMission();
            }}
            disabled={missionLoading || missionStarted}
            className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {missionLoading ? "Connecting..." : missionStarted ? "Mission Active" : "Start Mission"}
          </button>
        </div>
      </div>

      <section className="mx-auto grid h-[calc(100vh-6.5rem)] w-full max-w-[1400px] min-h-0 grid-cols-1 gap-3 lg:grid-cols-5">
        <div className="min-h-0 lg:col-span-3 lg:h-full">
          <GridMap missionStarted={missionStarted} />
        </div>

        <div className="grid min-h-0 gap-3 lg:col-span-2 lg:grid-rows-2">
          <div className="min-h-0">
            <MissionLog logs={logs} />
          </div>
          <div className="min-h-0">
            <BatteryDashboard drones={telemetry} />
          </div>
        </div>
      </section>
    </main>
  );
}
