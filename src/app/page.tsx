"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import BatteryDashboard from "../components/BatteryDashboard";
import GridMap from "../components/GridMap";
import MissionLog from "../components/MissionLog";
import SurvivorDashboard from "../components/SurvivorDashboard";

type Telemetry = {
  drone_id: number;
  x: number;
  y: number;
  battery_percentage: number;
  mode: string;
};

type Survivor = {
  x: number;
  y: number;
  detected_round: number;
  detected_elapsed_minutes: number;
  status: "suspect" | "confirmed";
  route_ready: boolean;
};

type SurvivorsResponse = {
  count: number;
  confirmed_count: number;
  suspect_count: number;
  survivors: Survivor[];
};

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

type HealthResponse = {
  orchestrator_ready?: boolean;
  orchestrator_error?: string | null;
  simulation_running?: boolean;
  elapsed_minutes?: number;
  mission_phase?: string;
  mission_completed?: boolean;
  completed_round?: number | null;
  completed_elapsed_minutes?: number | null;
};

export default function Home() {
  const [missionStarted, setMissionStarted] = useState(false);
  const [missionLoading, setMissionLoading] = useState(false);
  const [missionStopping, setMissionStopping] = useState(false);
  const [missionRestarting, setMissionRestarting] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry[]>([]);
  const [survivors, setSurvivors] = useState<Survivor[]>([]);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [missionCompleted, setMissionCompleted] = useState(false);
  const [completedRound, setCompletedRound] = useState<number | null>(null);
  const [completedMinutes, setCompletedMinutes] = useState<number | null>(null);
  const [missionElapsedSeconds, setMissionElapsedSeconds] = useState(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  const missionCompletedLoggedRef = useRef(false);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (missionLoading) return "Connecting...";
    if (missionRestarting) return "Restarting...";
    if (missionStopping) return "Stopping...";
    if (missionCompleted) return "Mission completed";
    if (simulationRunning) return "Simulation running";
    return missionStarted ? "Mission active" : "Mission idle";
  }, [
    missionLoading,
    missionRestarting,
    missionStarted,
    missionStopping,
    simulationRunning,
    missionCompleted,
  ]);

  const survivorCounts = useMemo(() => {
    const suspect = survivors.filter((item) => item.status === "suspect").length;
    const confirmed = survivors.filter((item) => item.status === "confirmed").length;
    return { suspect, confirmed, total: survivors.length };
  }, [survivors]);

  const formattedMissionTimer = useMemo(() => {
    const total = Math.max(0, Math.floor(missionElapsedSeconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    if (hours > 0) {
      return `${hours.toString().padStart(2, "0")}:${minutes
        .toString()
        .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
    }
    return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  }, [missionElapsedSeconds]);

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

  const fetchSurvivors = async (logErrors = false) => {
    try {
      const response = await fetch(`${BACKEND_BASE_URL}/survivors`);
      if (!response.ok) {
        throw new Error(`Survivor request failed (${response.status})`);
      }
      const data = (await response.json()) as SurvivorsResponse;
      setSurvivors(Array.isArray(data.survivors) ? data.survivors : []);
    } catch (error) {
      if (logErrors) {
        const message = error instanceof Error ? error.message : "Survivor fetch failed";
        setLogs((prev) => [`[error] ${message}`, ...prev]);
      }
    }
  };

  useEffect(() => {
    void fetchTelemetry();
    void fetchSurvivors();
    const timer = window.setInterval(() => {
      void fetchTelemetry();
      void fetchSurvivors();
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
      setSimulationRunning(Boolean(health.simulation_running));
      if (
        health.mission_phase === "searching" &&
        typeof health.elapsed_minutes === "number"
      ) {
        setMissionElapsedSeconds(Math.max(0, Math.floor(health.elapsed_minutes)) * 60);
      }
      setMissionCompleted(Boolean(health.mission_completed));
      setCompletedRound(
        typeof health.completed_round === "number" ? health.completed_round : null
      );
      setCompletedMinutes(
        typeof health.completed_elapsed_minutes === "number"
          ? health.completed_elapsed_minutes
          : null
      );
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

  useEffect(() => {
    let cancelled = false;

    const pollHealth = async () => {
      try {
        const response = await fetch(`${BACKEND_BASE_URL}/health`);
        if (!response.ok) {
          return;
        }
        const health = (await response.json()) as HealthResponse;
        if (!cancelled) {
          setSimulationRunning(Boolean(health.simulation_running));
          if (
            health.mission_phase === "searching" &&
            typeof health.elapsed_minutes === "number"
          ) {
            setMissionElapsedSeconds(Math.max(0, Math.floor(health.elapsed_minutes)) * 60);
          }
          setMissionCompleted(Boolean(health.mission_completed));
          setCompletedRound(
            typeof health.completed_round === "number" ? health.completed_round : null
          );
          setCompletedMinutes(
            typeof health.completed_elapsed_minutes === "number"
              ? health.completed_elapsed_minutes
              : null
          );
        }
      } catch {
        // Keep previous health state when backend is temporarily unavailable.
      }
    };

    void pollHealth();
    const timer = window.setInterval(() => {
      void pollHealth();
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!missionCompleted || missionCompletedLoggedRef.current) {
      return;
    }

    missionCompletedLoggedRef.current = true;
    closeMissionStream();
    setMissionLoading(false);
    setMissionStarted(false);
    setSimulationRunning(false);
    setLogs((prev) => ["[status] Mission complete: all drones returned to base.", ...prev].slice(0, 100));
    void fetchTelemetry(true);
    void fetchSurvivors(true);
  }, [missionCompleted]);

  const startMission = async (options?: { force?: boolean }) => {
    const forceStart = options?.force ?? false;
    if (missionLoading || missionStopping || missionRestarting || (missionStarted && !forceStart)) {
      return;
    }

    closeMissionStream();
    setMissionLoading(true);
    setMissionElapsedSeconds(0);

    if (missionCompleted) {
      missionCompletedLoggedRef.current = false;
      setMissionCompleted(false);
      setCompletedRound(null);
      setCompletedMinutes(null);
    }

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
      setSimulationRunning(true);
      setMissionLoading(false);
      addLog("[system] Mission stream connected.");
      void fetchTelemetry(true);
      void fetchSurvivors(true);
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
      closeMissionStream();
      setMissionStarted(false);
      void fetchTelemetry(true);
      void fetchSurvivors(true);
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
      closeMissionStream();
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
      setMissionElapsedSeconds(0);
      if (!silent) {
        addLog("[status] Mission stop requested.");
      }
      setSimulationRunning(false);
      setMissionStopping(false);
      void fetchTelemetry(true);
      void fetchSurvivors(true);
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
      setSurvivors([]);
      missionCompletedLoggedRef.current = false;
      setMissionCompleted(false);
      setCompletedRound(null);
      setCompletedMinutes(null);
      setMissionElapsedSeconds(0);
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
          <span className="text-sm font-medium text-slate-700">{statusLabel}</span>
          <span className="rounded-lg bg-slate-800 px-3 py-1 font-mono text-sm font-semibold text-white">
            {formattedMissionTimer}
          </span>
          <button
            type="button"
            onClick={() => {
              void startMission();
            }}
            disabled={missionLoading || missionStarted || missionStopping || missionRestarting || missionCompleted}
            className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {missionLoading ? "Connecting..." : missionStarted ? "Mission Active" : "Start Mission"}
          </button>
          <button
            type="button"
            onClick={() => {
              void stopMission();
            }}
            disabled={missionStopping || missionRestarting || missionCompleted}
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

      {missionCompleted ? (
        <section className="mx-auto mb-3 w-full max-w-[1400px] rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          <p className="font-semibold">
            Mission Complete: all drones returned to base and the simulation ended.
          </p>
          <p className="mt-1">
            Survivor results: {survivorCounts.total} detected coordinates ({survivorCounts.suspect} suspect, {survivorCounts.confirmed} confirmed)
            {completedRound !== null && completedMinutes !== null
              ? ` at round ${completedRound} (${completedMinutes} min).`
              : "."}
          </p>
        </section>
      ) : null}

      <section className="mx-auto grid h-[calc(100vh-6.5rem)] w-full max-w-[1400px] min-h-0 grid-cols-1 gap-3 lg:grid-cols-5">
        <div className="min-h-0 lg:col-span-3 lg:h-full">
          <GridMap missionStarted={missionStarted} />
        </div>

        <div className="grid min-h-0 gap-3 lg:col-span-2 lg:grid-rows-3">
          <div className="min-h-0">
            <MissionLog logs={logs} />
          </div>
          <div className="min-h-0">
            <BatteryDashboard drones={telemetry} />
          </div>
          <div className="min-h-0">
            <SurvivorDashboard survivors={survivors} />
          </div>
        </div>
      </section>
    </main>
  );
}
