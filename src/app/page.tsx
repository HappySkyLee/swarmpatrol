"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import BatteryDashboard from "../components/BatteryDashboard";
import GridMap from "../components/GridMap";
import MissionLog from "../components/MissionLog";

type MissionMode = "ai" | "fallback" | null;

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
  const [missionStopping, setMissionStopping] = useState(false);
  const [missionRestarting, setMissionRestarting] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry[]>([]);
  const [missionMode, setMissionMode] = useState<MissionMode>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (missionLoading) return "Connecting...";
    if (missionRestarting) return "Restarting...";
    if (missionStopping) return "Stopping...";
    return missionStarted ? "Mission active" : "Mission idle";
  }, [missionLoading, missionRestarting, missionStarted, missionStopping]);

  const closeMissionStream = () => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  };

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

  const checkMissionReadiness = async (): Promise<{
    ok: boolean;
    reason?: string;
    warning?: string;
  }> => {
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
          ok: true,
          warning:
            health.orchestrator_error ??
            "Mission orchestrator is unavailable. Running simulation-only fallback mode.",
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

  const startMission = async (options?: { force?: boolean }) => {
    const forceStart = options?.force ?? false;
    if (missionLoading || missionStopping || missionRestarting || (missionStarted && !forceStart)) {
      return;
    }

    closeMissionStream();
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

    if (readiness.warning) {
      addLog(`[status] ${readiness.warning}`);
    }

    let connected = false;
    const es = new EventSource(`${BACKEND_BASE_URL}/mission_log`);
    eventSourceRef.current = es;

    es.onopen = () => {
      connected = true;
      setMissionStarted(true);
      setMissionLoading(false);
      setMissionMode("ai");
      addLog("[system] Mission stream connected.");
      void fetchTelemetry(true);
    };

    es.addEventListener("status", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as { message?: string; detail?: string };
        const message = data.message ?? (e.data as string);
        if (message.toLowerCase().includes("fallback")) {
          setMissionMode("fallback");
        }
        addLog(`[status] ${data.message ?? e.data}`);
        if (data.detail) {
          addLog(`[status] detail: ${data.detail}`);
        }
      } catch {
        addLog(`[status] ${e.data as string}`);
      }
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
      closeMissionStream();
      setMissionStarted(false);
      setMissionMode(null);
      void fetchTelemetry(true);
    });

    es.addEventListener("mission_error", (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data as string) as { message?: string; detail?: string };
        const detail = data.detail ? ` ${data.detail}` : "";
        addLog(`[error] ${data.message ?? "Mission stream failed."}${detail}`);
      } catch {
        addLog("[error] Mission stream failed.");
      }

      setMissionLoading(false);
      setMissionStarted(false);
      setMissionMode(null);
      closeMissionStream();
    });

    es.onerror = () => {
      if (!connected) {
        addLog(
          "[error] Cannot connect to mission stream. Check /health for orchestrator_error details."
        );
        setMissionLoading(false);
        setMissionStarted(false);
        setMissionMode(null);
      } else {
        addLog("[status] Mission stream closed.");
      }
      closeMissionStream();
    };
  };

  const stopMission = async (options?: { silent?: boolean }) => {
    const silent = options?.silent ?? false;
    if (missionStopping || missionRestarting) {
      return;
    }

    setMissionStopping(true);
    const addLog = (line: string) =>
      setLogs((prev) => [line, ...prev].slice(0, 100));

    try {
      const response = await fetch(`${BACKEND_BASE_URL}/stop_mission`, {
        method: "POST",
      });
      if (!response.ok && !silent) {
        addLog(`[error] Stop mission request failed (${response.status}).`);
      }
    } catch {
      if (!silent) {
        addLog("[error] Cannot reach backend stop_mission endpoint.");
      }
    } finally {
      closeMissionStream();
      setMissionLoading(false);
      setMissionStarted(false);
      setMissionMode(null);
      if (!silent) {
        addLog("[status] Mission stop requested.");
      }
      setMissionStopping(false);
      void fetchTelemetry(true);
    }
  };

  const restartMission = async () => {
    if (missionLoading || missionStopping || missionRestarting) {
      return;
    }

    setMissionRestarting(true);
    const addLog = (line: string) =>
      setLogs((prev) => [line, ...prev].slice(0, 100));

    await stopMission({ silent: true });

    try {
      const response = await fetch(`${BACKEND_BASE_URL}/restart_mission`, {
        method: "POST",
      });

      if (!response.ok) {
        addLog(`[error] Restart mission request failed (${response.status}).`);
        return;
      }

      setLogs(["[status] Mission state reset complete. Starting new mission stream..."]);
      await startMission({ force: true });
    } catch {
      addLog("[error] Cannot reach backend restart_mission endpoint.");
    } finally {
      setMissionRestarting(false);
    }
  };

  return (
    <main className="min-h-screen overflow-y-auto bg-gradient-to-br from-slate-100 via-sky-50 to-cyan-100 p-3 md:p-4">
      <div className="mx-auto flex w-full max-w-[1400px] items-center justify-between pb-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 md:text-3xl">Swarm Mission Dashboard</h1>
          <p className="text-sm text-slate-600">Command Agent Base: (20,20)</p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-slate-700">
            {statusLabel}
            {missionMode === "fallback" ? " • Fallback Mode" : ""}
          </span>
          <button
            type="button"
            onClick={() => {
              void startMission();
            }}
            disabled={missionLoading || missionStarted || missionStopping || missionRestarting}
            className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {missionLoading ? "Connecting..." : missionStarted ? "Mission Active" : "Start Mission"}
          </button>
          <button
            type="button"
            onClick={() => {
              void stopMission();
            }}
            disabled={missionStopping || missionRestarting || (!missionStarted && !missionLoading)}
            className="rounded-xl bg-rose-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-600 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {missionStopping ? "Stopping..." : "Stop Mission"}
          </button>
          <button
            type="button"
            onClick={() => {
              void restartMission();
            }}
            disabled={missionLoading || missionStopping || missionRestarting}
            className="rounded-xl bg-amber-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-amber-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {missionRestarting ? "Restarting..." : "Restart Mission"}
          </button>
        </div>
      </div>

      <section className="mx-auto grid h-[calc(100vh-6.5rem)] w-full max-w-[1400px] min-h-0 grid-cols-1 gap-3 lg:grid-cols-5">
        <div className="min-h-0 lg:col-span-3 lg:h-full">
          <GridMap missionStarted={missionStarted} />
        </div>

        <div className="grid min-h-0 gap-3 lg:col-span-2 lg:grid-rows-2">
          <div className="min-h-0">
            <MissionLog logs={logs} mode={missionMode} />
          </div>
          <div className="min-h-0">
            <BatteryDashboard drones={telemetry} />
          </div>
        </div>
      </section>
    </main>
  );
}
