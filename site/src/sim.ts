// Replays an exported episode: the exact warehouse state at any (fractional) tick.

import type { Cell, Episode, Option, Order, Truck } from "./data";

export interface RobotState {
  id: number;
  x: number; // column, fractional while moving
  y: number; // row
  heading: number; // radians
  kind: Option["k"];
  moving: boolean;
  battery: number; // 0..1
  carried: { tier: number; order: number }[];
  target: Cell;
  trail: { x: number; y: number }[];
}

export interface ItemState {
  order: Order;
  cell: Cell;
  index: number;
}

export interface TruckState {
  truck: Truck;
  /** 0 = parked, 0..1 driving away, -1..0 pulling in */
  motion: number;
  countdown: number; // ticks to scheduled departure
}

export interface Pop {
  kind: "fulfilled" | "lost" | "deferred";
  order: Order;
  age: number; // ticks since event
  dock: number;
}

export interface WorldState {
  t: number;
  robots: RobotState[];
  shelfItems: ItemState[];
  dockItems: { dock: number; tier: number; order: number }[];
  trucks: TruckState[];
  pops: Pop[];
  captured: number;
  arrived: number;
  fulfilled: number;
  lost: number;
  deferred: number;
  openOrders: Order[];
}

const DRIVE_TICKS = 10;
const POP_TICKS = 22;
const TRAIL_TICKS = 14;

export class Replay {
  readonly ep: Episode;
  private byRobot: Option[][];
  private picker = new Map<string, number>();
  private truckById = new Map<number, Truck>();
  private dockTrucks: Truck[][];

  constructor(ep: Episode) {
    this.ep = ep;
    this.byRobot = Array.from({ length: ep.robots }, () => []);
    for (const o of ep.options) {
      this.byRobot[o.r].push(o);
      if (o.k === "pick" && o.order !== undefined) this.picker.set(`${o.order}:${o.item}`, o.r);
    }
    for (const t of ep.trucks) this.truckById.set(t.id, t);
    const docks = ep.layout.docks.length;
    this.dockTrucks = Array.from({ length: docks }, (_, d) =>
      ep.trucks.filter((t) => t.dock === d).sort((a, b) => a.scheduled - b.scheduled),
    );
  }

  get horizon() {
    return this.ep.horizon;
  }

  truck(id: number) {
    return this.truckById.get(id)!;
  }

  /** Option a robot is executing at time t (last started option). */
  optionAt(r: number, t: number): Option | undefined {
    const opts = this.byRobot[r];
    let lo = 0;
    let hi = opts.length - 1;
    let found: Option | undefined;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (opts[mid].t0 <= t) {
        found = opts[mid];
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return found;
  }

  private nextOption(r: number, o: Option): Option | undefined {
    const opts = this.byRobot[r];
    return opts[opts.indexOf(o) + 1];
  }

  robotPos(r: number, t: number): { x: number; y: number; heading: number; moving: boolean } {
    const o = this.optionAt(r, t);
    if (!o) {
      const [y, x] = this.ep.layout.packing;
      return { x, y, heading: 0, moving: false };
    }
    const steps = o.path.length - 1;
    const p = Math.max(0, Math.min(steps, t - o.t0));
    const i = Math.min(steps - 1, Math.floor(p));
    if (steps <= 0) {
      const [y, x] = o.path[0];
      return { x, y, heading: 0, moving: false };
    }
    const f = p - i;
    const [y0, x0] = o.path[i];
    const [y1, x1] = o.path[i + 1];
    const eased = f; // constant speed: one cell per tick, as in the simulator
    return {
      x: x0 + (x1 - x0) * eased,
      y: y0 + (y1 - y0) * eased,
      heading: Math.atan2(y1 - y0, x1 - x0),
      moving: p < steps,
    };
  }

  state(t: number): WorldState {
    const ep = this.ep;
    const robots: RobotState[] = [];
    for (let r = 0; r < ep.robots; r++) {
      const pos = this.robotPos(r, t);
      const o = this.optionAt(r, t);
      const next = o ? this.nextOption(r, o) : undefined;
      let battery = ep.battery;
      if (o) {
        const b1 = next ? next.bat : o.bat;
        const span = (next ? next.t0 : o.t1) - o.t0 || 1;
        battery = o.bat + (b1 - o.bat) * Math.min(1, Math.max(0, (t - o.t0) / span));
      }
      const trail: { x: number; y: number }[] = [];
      for (let k = TRAIL_TICKS; k >= 1; k -= 1) {
        if (t - k < 0) continue;
        const p = this.robotPos(r, t - k);
        trail.push({ x: p.x, y: p.y });
      }
      const last = o?.path[o.path.length - 1] ?? ep.layout.packing;
      robots.push({
        id: r,
        x: pos.x,
        y: pos.y,
        heading: pos.heading,
        kind: o?.k ?? "wait",
        moving: pos.moving,
        battery: battery / ep.battery,
        carried: [],
        target: last,
        trail,
      });
    }

    const shelfItems: ItemState[] = [];
    const dockItems: WorldState["dockItems"] = [];
    const pops: Pop[] = [];
    const openOrders: Order[] = [];
    let captured = 0;
    let arrived = 0;
    let fulfilled = 0;
    let lost = 0;
    let deferred = 0;
    for (const order of ep.orders) {
      if (order.arrival > t) continue;
      arrived += order.value;
      const open = t < order.resolved;
      if (!open) {
        if (order.status === "fulfilled") {
          captured += order.captured;
          fulfilled += 1;
        } else lost += 1;
        const age = t - order.resolved;
        if (age < POP_TICKS) pops.push({ kind: order.status === "fulfilled" ? "fulfilled" : "lost", order, age, dock: order.dock });
        continue;
      }
      openOrders.push(order);
      if (order.deferrals > 0) {
        const first = this.truck(order.truck0);
        if (first && t >= first.actual) {
          deferred += 1;
          const age = t - first.actual;
          if (age < POP_TICKS) pops.push({ kind: "deferred", order, age, dock: order.dock });
        }
      }
      order.items.forEach((item, index) => {
        const picked = item.pick !== null && item.pick <= t;
        const loaded = item.load !== null && item.load <= t;
        if (loaded) dockItems.push({ dock: order.dock, tier: order.tier, order: order.id });
        else if (picked) {
          const r = this.picker.get(`${order.id}:${index}`) ?? 0;
          robots[r]?.carried.push({ tier: order.tier, order: order.id });
        } else shelfItems.push({ order, cell: item.cell, index });
      });
    }

    const trucks: TruckState[] = [];
    this.dockTrucks.forEach((list) => {
      // the truck at the dock: the first not yet gone (+ drive-away window)
      let current: Truck | undefined;
      let prevGone = -Infinity;
      for (const truck of list) {
        if (t < truck.actual + DRIVE_TICKS) {
          current = truck;
          break;
        }
        prevGone = truck.actual + DRIVE_TICKS;
      }
      if (!current) return;
      let motion = 0;
      if (t > current.actual) motion = Math.min(1, (t - current.actual) / DRIVE_TICKS);
      else if (t - prevGone < DRIVE_TICKS) motion = -1 + Math.max(0, (t - prevGone) / DRIVE_TICKS);
      trucks.push({ truck: current, motion, countdown: current.scheduled - t });
    });

    return { t, robots, shelfItems, dockItems, trucks, pops, captured, arrived, fulfilled, lost, deferred, openOrders };
  }

  /** Revenue captured so far as a share of all revenue that has arrived so far. */
  capturedPct(t: number): number {
    const s = this.state(t);
    return s.arrived > 0 ? (100 * s.captured) / s.arrived : 0;
  }

  /** Final captured share of all revenue, as the simulator reports it. */
  get finalPct(): number {
    return this.ep.stats.revenue_captured_pct;
  }
}

/** Breadth-first path on the layout grid between two cells (row, col). */
export function gridPath(grid: string[], from: Cell, to: Cell): Cell[] {
  const walkable = (r: number, c: number) => ".PCD".includes(grid[r]?.[c] ?? "#");
  const key = (r: number, c: number) => r * 1000 + c;
  const prev = new Map<number, number>();
  const queue: Cell[] = [from];
  prev.set(key(...from), -1);
  while (queue.length) {
    const [r, c] = queue.shift()!;
    if (r === to[0] && c === to[1]) break;
    for (const [dr, dc] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nr = r + dr;
      const nc = c + dc;
      if (walkable(nr, nc) && !prev.has(key(nr, nc))) {
        prev.set(key(nr, nc), key(r, c));
        queue.push([nr, nc]);
      }
    }
  }
  const out: Cell[] = [];
  let k = key(...to);
  if (!prev.has(k)) return [from];
  while (k !== -1) {
    out.push([Math.floor(k / 1000), k % 1000]);
    k = prev.get(k)!;
  }
  return out.reverse();
}
