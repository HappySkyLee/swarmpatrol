"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type GridMapProps = {
  missionStarted: boolean;
};

type GridStateResponse = {
  width: number;
  height: number;
  cell_status: string[][];
  hazard_types: string[][];
};

type DroneTelemetry = {
  drone_id: number;
  x: number;
  y: number;
  battery_percentage: number;
  mode: string;
};

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

const COMMAND_AGENT_BASE = { x: 20, y: 20 };

const defaultGrid = {
  width: 40,
  height: 40,
  cell_status: Array.from({ length: 40 }, () =>
    Array.from({ length: 40 }, () => "unvisited")
  ),
  hazard_types: Array.from({ length: 40 }, () =>
    Array.from({ length: 40 }, () => "none")
  ),
};

export default function GridMap({ missionStarted }: GridMapProps) {
  const [grid, setGrid] = useState<GridStateResponse>(defaultGrid);
  const [drones, setDrones] = useState<DroneTelemetry[]>([]);
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number } | null>(null);
  const [hoverTooltipPosition, setHoverTooltipPosition] = useState<{ x: number; y: number } | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);

  const flattenedCells = useMemo(() => {
    const cells: string[] = [];
    for (let y = 0; y < grid.height; y += 1) {
      for (let x = 0; x < grid.width; x += 1) {
        cells.push(grid.cell_status[y]?.[x] ?? "unvisited");
      }
    }
    return cells;
  }, [grid.cell_status, grid.height, grid.width]);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [gridRes, droneRes] = await Promise.all([
          fetch(`${BACKEND_BASE_URL}/grid_state`),
          fetch(`${BACKEND_BASE_URL}/drone_telemetry`),
        ]);

        if (!gridRes.ok || !droneRes.ok) {
          return;
        }

        const gridData = (await gridRes.json()) as GridStateResponse;
        const droneData = (await droneRes.json()) as DroneTelemetry[];

        if (!cancelled) {
          setGrid(gridData);
          setDrones(droneData);
        }
      } catch {
        // Keep last known map if backend is temporarily unavailable.
      }
    };

    load();
    const timer = window.setInterval(load, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const cellClass = (status: string, hazardType: string, x: number, y: number) => {
    if (x === COMMAND_AGENT_BASE.x && y === COMMAND_AGENT_BASE.y) {
      return "bg-emerald-500";
    }
    if (status === "unconfirmed") {
      return "animate-pulse bg-orange-400";
    }
    if (status === "survivor_found" || status === "survivor") {
      return "bg-red-500";
    }
    if (hazardType === "heavy_smoke") {
      return "bg-slate-200";
    }
    if (hazardType === "heavy_wind") {
      return "bg-sky-200";
    }
    if (status === "clear") {
      return "bg-slate-700";
    }
    return "bg-slate-300";
  };

  const updateHoverTooltipPosition = (clientX: number, clientY: number) => {
    if (!mapContainerRef.current) {
      return;
    }

    const rect = mapContainerRef.current.getBoundingClientRect();
    const TOOLTIP_W = 80;
    const TOOLTIP_H = 28;
    const OFFSET = 10;

    const relX = clientX - rect.left;
    const relY = clientY - rect.top;

    // Default: bottom-right of cursor
    let tx = relX + OFFSET;
    let ty = relY + OFFSET;

    // Flip to left when close to right edge
    if (tx + TOOLTIP_W > rect.width) {
      tx = relX - TOOLTIP_W - OFFSET;
    }

    // Flip to above when close to bottom edge
    if (ty + TOOLTIP_H > rect.height) {
      ty = relY - TOOLTIP_H - OFFSET;
    }

    setHoverTooltipPosition({ x: tx, y: ty });
  };

  return (
    <section className="h-full rounded-2xl border border-slate-200 bg-white/70 p-5 shadow-sm backdrop-blur">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">Grid Map</h2>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          <span className="flex items-center gap-1 text-xs text-slate-600">
            <span className="inline-block h-3 w-3 rounded-sm bg-orange-400" />
            Unconfirmed
          </span>
          <span className="flex items-center gap-1 text-xs text-slate-600">
            <span className="inline-block h-3 w-3 rounded-sm bg-red-500" />
            Survivor
          </span>
          <span className="flex items-center gap-1 text-xs text-slate-600">
            <span className="inline-block h-3 w-3 rounded-sm bg-slate-700" />
            Visited
          </span>
          <span className="flex items-center gap-1 text-xs text-slate-600">
            <span className="inline-block h-3 w-3 rounded-sm bg-sky-200" />
            Heavy Wind
          </span>
          <span className="flex items-center gap-1 text-xs text-slate-600">
            <span className="inline-block h-3 w-3 rounded-sm bg-slate-200" />
            Heavy Smoke
          </span>
        </div>
      </div>

      <div
        ref={mapContainerRef}
        className="relative h-[calc(100%-3rem)] overflow-hidden rounded-xl border border-slate-200 bg-slate-100 p-2"
      >
        <div
          className="grid h-full w-full gap-[1px]"
          onMouseLeave={() => {
            setHoveredCell(null);
            setHoverTooltipPosition(null);
          }}
          style={{
            width: "100%",
            height: "100%",
            gridTemplateColumns: `repeat(${grid.width}, 1fr)`,
            gridTemplateRows: `repeat(${grid.height}, 1fr)`,
          }}
        >
          {flattenedCells.map((status, index) => {
            const x = index % grid.width;
            const y = Math.floor(index / grid.width);
            const hazardType = grid.hazard_types[y]?.[x] ?? "none";
            return (
              <div
                key={`${index}-${status}`}
                className={cellClass(status, hazardType, x, y)}
                onMouseEnter={(event) => {
                  setHoveredCell({ x, y });
                  updateHoverTooltipPosition(event.clientX, event.clientY);
                }}
                onMouseMove={(event) => {
                  setHoveredCell({ x, y });
                  updateHoverTooltipPosition(event.clientX, event.clientY);
                }}
              />
            );
          })}
        </div>

        {hoveredCell && hoverTooltipPosition ? (
          <div
            className="pointer-events-none absolute z-20 rounded-md border border-slate-200 bg-white/95 px-2 py-1 text-xs font-medium text-slate-800 shadow-sm"
            style={{
              left: `${hoverTooltipPosition.x}px`,
              top: `${hoverTooltipPosition.y}px`,
            }}
          >
            ({hoveredCell.x}, {hoveredCell.y})
          </div>
        ) : null}

        {drones.map((drone) => {
          const left = (drone.x / Math.max(grid.width - 1, 1)) * 100;
          const top = (drone.y / Math.max(grid.height - 1, 1)) * 100;
          return (
            <div
              key={drone.drone_id}
              className="absolute z-10 -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${left}%`, top: `${top}%` }}
              title={`Drone ${drone.drone_id} (${drone.x}, ${drone.y})`}
            >
              <div className="flex h-5 w-5 items-center justify-center rounded-full border border-white bg-slate-900 text-[10px] font-bold text-white shadow">
                D
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
