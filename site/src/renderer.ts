// 2.5D canvas renderer for a warehouse replay.

import type { Cell, Layout } from "./data";
import { FLOOR, INK, INK2, MUTED, ROBOT, SHELF_FRONT, SHELF_TOP, SURFACE, TIER, WALL, rgba } from "./palette";
import type { RobotState, WorldState } from "./sim";

export interface Ghost {
  path: Cell[];
  score: number; // 0..1 normalised
  chosen: boolean;
  label: string;
}

export interface Overlay {
  attention?: Record<string, number>;
  ghosts?: Ghost[];
  /** Cells to spotlight; everything else is dimmed. */
  spotlight?: Cell[];
  spotlightRadius?: number;
  robotColors?: readonly string[];
  robotLabels?: boolean;
  /** Draw the countdown rings on trucks. */
  countdowns?: boolean;
  highlightTier?: number | null;
}

const TOP = 0.6; // cells of margin above the grid
const BOTTOM = 2.6; // room for trucks below the dock wall

export class WarehouseRenderer {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private layout: Layout;
  private rows: number;
  private cols: number;
  private s = 20; // cell size in css px
  private ox = 0;
  private oy = 0;
  private dpr = 1;
  private staticLayer: HTMLCanvasElement = document.createElement("canvas");
  private shelfSide = new Map<string, [number, number]>();
  private ro: ResizeObserver;

  constructor(canvas: HTMLCanvasElement, layout: Layout) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.layout = layout;
    this.rows = layout.grid.length;
    this.cols = layout.grid[0].length;
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < this.cols; c++) {
        if (!".PCD".includes(layout.grid[r][c])) continue;
        for (const [dr, dc] of [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ])
          if (layout.grid[r + dr]?.[c + dc] === "S") {
            this.shelfSide.set(`${r},${c}`, [dr, dc]);
            break;
          }
      }
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.resize();
  }

  destroy() {
    this.ro.disconnect();
  }

  private resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    const s = Math.min(rect.width / (this.cols + 0.4), rect.height / (this.rows + TOP + BOTTOM));
    this.s = s;
    this.ox = (rect.width - s * this.cols) / 2;
    this.oy = (rect.height - s * (this.rows + TOP + BOTTOM)) / 2 + s * TOP;
    this.drawStatic();
  }

  /** Centre of cell (row, col) in css px. */
  px(col: number, row: number): [number, number] {
    return [this.ox + (col + 0.5) * this.s, this.oy + (row + 0.5) * this.s];
  }

  private drawStatic() {
    const layer = this.staticLayer;
    layer.width = this.canvas.width;
    layer.height = this.canvas.height;
    const g = layer.getContext("2d")!;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const { s, ox, oy } = this;
    const grid = this.layout.grid;
    // floor slab with a soft drop shadow
    g.save();
    g.shadowColor = "rgba(0,0,0,0.65)";
    g.shadowBlur = s * 1.6;
    g.shadowOffsetY = s * 0.6;
    g.fillStyle = WALL;
    roundRect(g, ox - s * 0.1, oy - s * 0.1, s * this.cols + s * 0.2, s * this.rows + s * 0.2, s * 0.4);
    g.fill();
    g.restore();
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < this.cols; c++) {
        const ch = grid[r][c];
        const x = ox + c * s;
        const y = oy + r * s;
        if (ch === "#") continue;
        g.fillStyle = FLOOR;
        g.fillRect(x + 0.5, y + 0.5, s - 1, s - 1);
      }
    // subtle floor grid
    g.strokeStyle = "rgba(255,255,255,0.025)";
    g.lineWidth = 1;
    for (let r = 1; r < this.rows - 1; r++) {
      g.beginPath();
      g.moveTo(ox + s, oy + r * s);
      g.lineTo(ox + (this.cols - 1) * s, oy + r * s);
      g.stroke();
    }
    // shelves: raised blocks with a lit top and darker front face
    const lift = s * 0.32;
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < this.cols; c++) {
        if (grid[r][c] !== "S") continue;
        const x = ox + c * s + 1.5;
        const y = oy + r * s + 1.5;
        const w = s - 3;
        const below = grid[r + 1]?.[c] === "S";
        if (!below) {
          g.fillStyle = SHELF_FRONT;
          g.fillRect(x, y + w - lift * 0.2, w, lift);
        }
        g.fillStyle = SHELF_TOP;
        g.fillRect(x, y - lift * 0.6, w, w);
        g.fillStyle = "rgba(255,255,255,0.05)";
        g.fillRect(x, y - lift * 0.6, w, 1.5);
        // tiny bin dividers
        g.fillStyle = "rgba(0,0,0,0.22)";
        g.fillRect(x + w * 0.5 - 0.5, y - lift * 0.6 + 2, 1, w - 4);
      }
    // docks
    this.layout.docks.forEach(([r, c], d) => {
      const [x, y] = this.px(c, r);
      g.save();
      g.strokeStyle = rgba(INK2, 0.5);
      g.setLineDash([3, 3]);
      g.lineWidth = 1.2;
      roundRect(g, x - s * 0.46, y - s * 0.46, s * 0.92, s * 0.92, s * 0.18);
      g.stroke();
      g.restore();
      g.fillStyle = rgba(INK2, 0.75);
      g.font = `600 ${Math.max(8, s * 0.36)}px "JetBrains Mono", monospace`;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(`D${d + 1}`, x, y);
    });
    // charger
    if (this.layout.charger) {
      const [r, c] = this.layout.charger;
      const [x, y] = this.px(c, r);
      const grad = g.createRadialGradient(x, y, 0, x, y, s * 1.4);
      grad.addColorStop(0, "rgba(240,239,236,0.16)");
      grad.addColorStop(1, "rgba(240,239,236,0)");
      g.fillStyle = grad;
      g.fillRect(x - s * 1.5, y - s * 1.5, s * 3, s * 3);
      bolt(g, x, y, s * 0.42, INK);
    }
    // packing / start
    {
      const [r, c] = this.layout.packing;
      const [x, y] = this.px(c, r);
      g.strokeStyle = rgba(INK2, 0.35);
      g.lineWidth = 1;
      g.beginPath();
      g.arc(x, y, s * 0.32, 0, Math.PI * 2);
      g.stroke();
    }
  }

  draw(world: WorldState, overlay: Overlay = {}) {
    const g = this.ctx;
    const { s } = this;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    g.drawImage(this.staticLayer, 0, 0);
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const t = world.t;
    const colors = overlay.robotColors ?? ROBOT;

    this.drawTrucks(world, overlay.countdowns ?? true);

    // dock stacks: items loaded and waiting for their truck
    const perDock = new Map<number, number[]>();
    for (const it of world.dockItems) {
      const list = perDock.get(it.dock) ?? [];
      list.push(it.tier);
      perDock.set(it.dock, list);
    }
    perDock.forEach((tiers, d) => {
      const [r, c] = this.layout.docks[d];
      const [x, y] = this.px(c, r);
      tiers.slice(0, 12).forEach((tier, i) => {
        const cx = x - s * 0.3 + (i % 4) * s * 0.2;
        const cy = y - s * 0.62 - Math.floor(i / 4) * s * 0.2;
        g.fillStyle = TIER[tier];
        roundRect(g, cx - s * 0.08, cy - s * 0.08, s * 0.16, s * 0.16, s * 0.04);
        g.fill();
      });
    });

    // attention beams from the robot to the orders it is weighing
    if (overlay.attention && world.robots[0]) {
      const robot = world.robots[0];
      const [rx, ry] = this.px(robot.x, robot.y);
      const max = Math.max(...Object.values(overlay.attention), 1e-6);
      for (const item of world.shelfItems) {
        const w = overlay.attention[String(item.order.id)];
        if (!w) continue;
        const [ix, iy] = this.itemPos(item.cell, item.index);
        const k = w / max;
        const grad = g.createLinearGradient(rx, ry, ix, iy);
        grad.addColorStop(0, rgba(INK, 0.05));
        grad.addColorStop(1, rgba(TIER[item.order.tier], 0.15 + 0.7 * k));
        g.strokeStyle = grad;
        g.lineWidth = 0.6 + 3.2 * k;
        g.beginPath();
        g.moveTo(rx, ry);
        const mx = (rx + ix) / 2;
        const my = Math.min(ry, iy) - s * (0.6 + 1.2 * k);
        g.quadraticCurveTo(mx, my, ix, iy);
        g.stroke();
      }
    }

    // order items waiting on shelves
    for (const item of world.shelfItems) {
      const [x, y] = this.itemPos(item.cell, item.index);
      const o = item.order;
      const dim = overlay.highlightTier != null && overlay.highlightTier !== o.tier;
      const age = t - o.arrival;
      const appear = Math.min(1, age / 4);
      const radius = s * (0.14 + 0.07 * Math.min(1, Math.log1p(o.value) / 4.2)) * (0.4 + 0.6 * easeOutBack(appear));
      const color = TIER[o.tier];
      const pulse = 0.5 + 0.5 * Math.sin(t * 0.35 + o.id * 1.7);
      g.save();
      g.globalAlpha = dim ? 0.18 : 1;
      if (age < 6) {
        g.strokeStyle = rgba(color, 1 - age / 6);
        g.lineWidth = 1.5;
        g.beginPath();
        g.arc(x, y, radius + (age / 6) * s * 0.9, 0, Math.PI * 2);
        g.stroke();
      }
      g.shadowColor = color;
      g.shadowBlur = s * (0.35 + 0.35 * pulse);
      g.fillStyle = color;
      g.beginPath();
      g.arc(x, y, radius, 0, Math.PI * 2);
      g.fill();
      if (o.vip) {
        g.shadowBlur = 0;
        star(g, x, y - radius - s * 0.16, s * 0.12, INK);
      }
      g.restore();
    }

    // ghost futures (look-ahead candidates)
    if (overlay.ghosts) {
      for (const ghost of overlay.ghosts) {
        if (ghost.path.length < 2) continue;
        g.save();
        g.lineCap = "round";
        g.lineJoin = "round";
        g.setLineDash(ghost.chosen ? [] : [s * 0.15, s * 0.2]);
        g.strokeStyle = ghost.chosen ? rgba(INK, 0.95) : rgba(INK2, 0.15 + 0.45 * ghost.score);
        g.lineWidth = ghost.chosen ? 3 : 1.5;
        if (ghost.chosen) {
          g.shadowColor = INK;
          g.shadowBlur = s * 0.6;
        }
        g.beginPath();
        ghost.path.forEach(([r, c], i) => {
          const [x, y] = this.px(c, r);
          if (i === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        });
        g.stroke();
        const [er, ec] = ghost.path[ghost.path.length - 1];
        const [ex, ey] = this.px(ec, er);
        g.setLineDash([]);
        g.fillStyle = ghost.chosen ? INK : rgba(INK2, 0.5);
        g.beginPath();
        g.arc(ex, ey, ghost.chosen ? s * 0.16 : s * 0.1, 0, Math.PI * 2);
        g.fill();
        g.restore();
      }
    }

    // robots
    world.robots.forEach((robot) => this.drawRobot(robot, colors[robot.id % colors.length], overlay.robotLabels ?? false, t));

    // event pops, stacked so simultaneous events at one dock don't overlap
    const stack = new Map<number, number>();
    for (const pop of world.pops) {
      const [r, c] = this.layout.docks[pop.dock];
      const [x, y] = this.px(c, r);
      const slot = stack.get(pop.dock) ?? 0;
      stack.set(pop.dock, slot + 1);
      const k = pop.age / 22;
      const alpha = 1 - k * k;
      const rise = s * (0.9 + 2.2 * easeOutCubic(k) + slot * 0.75);
      g.save();
      g.globalAlpha = Math.max(0, alpha);
      g.textAlign = "center";
      g.font = `700 ${Math.max(10, s * 0.55)}px "JetBrains Mono", monospace`;
      if (pop.kind === "fulfilled") {
        g.fillStyle = INK;
        g.shadowColor = TIER[pop.order.tier];
        g.shadowBlur = 12;
        g.fillText(`+${pop.order.captured.toFixed(1)}`, x, y - rise);
      } else if (pop.kind === "lost") {
        g.fillStyle = TIER[0];
        g.fillText("✕ lost", x, y - rise);
      } else {
        g.fillStyle = INK2;
        g.fillText("½ deferred", x, y - rise);
      }
      g.restore();
    }

    if (overlay.spotlight && overlay.spotlight.length) this.drawSpotlight(overlay.spotlight, overlay.spotlightRadius ?? 1.4);
  }

  private itemPos(cell: Cell, index: number): [number, number] {
    const [r, c] = cell;
    const [x, y] = this.px(c, r);
    const side = this.shelfSide.get(`${r},${c}`) ?? [0, 0];
    const spread = (index - 1) * this.s * 0.18;
    return [x + side[1] * this.s * 0.3 + (side[1] === 0 ? spread : 0), y + side[0] * this.s * 0.3 + (side[0] === 0 ? spread : 0)];
  }

  private drawTrucks(world: WorldState, countdowns: boolean) {
    const g = this.ctx;
    const { s } = this;
    for (const ts of world.trucks) {
      const [r, c] = this.layout.docks[ts.truck.dock];
      const [x, y0] = this.px(c, r);
      const y = y0 + s * 1.55;
      const m = ts.motion;
      const dy = m > 0 ? easeInCubic(m) * s * 3 : m < 0 ? -m * s * 2 : 0;
      const alpha = m > 0 ? 1 - m : m < 0 ? 1 + m : 1;
      g.save();
      g.globalAlpha = Math.max(0, alpha);
      g.translate(x, y + dy);
      // trailer
      const w = s * 2.1;
      const h = s * 0.95;
      const grad = g.createLinearGradient(0, -h / 2, 0, h / 2);
      grad.addColorStop(0, "#2a3042");
      grad.addColorStop(1, "#1a1e2b");
      g.fillStyle = grad;
      roundRect(g, -w / 2, -h / 2, w, h, s * 0.16);
      g.fill();
      g.strokeStyle = "rgba(255,255,255,0.08)";
      g.lineWidth = 1;
      g.stroke();
      // cab
      g.fillStyle = "#33394d";
      roundRect(g, w / 2 - s * 0.1, -h * 0.38, s * 0.55, h * 0.76, s * 0.12);
      g.fill();
      g.fillStyle = rgba(INK2, 0.6);
      g.fillRect(w / 2 + s * 0.18, -h * 0.28, s * 0.16, h * 0.56);
      // countdown ring
      if (countdowns && !ts.truck.final) {
        const frac = Math.max(0, Math.min(1, ts.countdown / 180));
        const urgent = ts.countdown < 35;
        g.beginPath();
        g.strokeStyle = "rgba(255,255,255,0.08)";
        g.lineWidth = 2.5;
        g.arc(-w / 2 + s * 0.45, 0, s * 0.3, 0, Math.PI * 2);
        g.stroke();
        g.beginPath();
        g.strokeStyle = urgent ? TIER[0] : INK2;
        g.arc(-w / 2 + s * 0.45, 0, s * 0.3, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
        g.stroke();
      }
      g.fillStyle = rgba(INK, 0.85);
      g.font = `600 ${Math.max(8, s * 0.32)}px "JetBrains Mono", monospace`;
      g.textAlign = "center";
      g.textBaseline = "middle";
      const label = ts.truck.final ? "LAST" : ts.countdown > 0 ? `${Math.ceil(ts.countdown)}` : "GO";
      g.fillText(label, s * 0.25, 0);
      g.restore();
    }
  }

  private drawRobot(robot: RobotState, color: string, label: boolean, t: number) {
    const g = this.ctx;
    const { s } = this;
    // trail
    if (robot.trail.length > 1) {
      g.save();
      g.lineCap = "round";
      for (let i = 1; i < robot.trail.length; i++) {
        const a = robot.trail[i - 1];
        const b = robot.trail[i];
        const [ax, ay] = this.px(a.x, a.y);
        const [bx, by] = this.px(b.x, b.y);
        g.strokeStyle = rgba(color, (i / robot.trail.length) * 0.55);
        g.lineWidth = s * 0.22 * (i / robot.trail.length);
        g.beginPath();
        g.moveTo(ax, ay);
        g.lineTo(bx, by);
        g.stroke();
      }
      g.restore();
    }
    const [x, y] = this.px(robot.x, robot.y);
    const size = s * 0.66;
    const bob = robot.moving ? Math.sin(t * 2.4) * s * 0.03 : 0;
    g.save();
    g.translate(x, y + bob);
    // glow
    const glow = g.createRadialGradient(0, 0, 0, 0, 0, s * 1.3);
    glow.addColorStop(0, rgba(color, 0.45));
    glow.addColorStop(1, rgba(color, 0));
    g.fillStyle = glow;
    g.fillRect(-s * 1.3, -s * 1.3, s * 2.6, s * 2.6);
    // battery ring
    g.beginPath();
    g.strokeStyle = "rgba(255,255,255,0.1)";
    g.lineWidth = 2;
    g.arc(0, 0, size * 0.78, 0, Math.PI * 2);
    g.stroke();
    g.beginPath();
    g.strokeStyle = robot.battery < 0.25 ? TIER[0] : rgba(INK, 0.85);
    g.arc(0, 0, size * 0.78, -Math.PI / 2, -Math.PI / 2 + robot.battery * Math.PI * 2);
    g.stroke();
    // body
    g.fillStyle = SURFACE;
    roundRect(g, -size / 2, -size / 2, size, size, size * 0.28);
    g.fill();
    g.lineWidth = 2;
    g.strokeStyle = color;
    g.shadowColor = color;
    g.shadowBlur = s * 0.5;
    g.stroke();
    g.shadowBlur = 0;
    // "eye" showing heading
    g.fillStyle = color;
    const ex = Math.cos(robot.heading) * size * 0.2;
    const ey = Math.sin(robot.heading) * size * 0.2;
    g.beginPath();
    g.arc(ex, ey, size * 0.12, 0, Math.PI * 2);
    g.fill();
    // charging sparkle
    if (robot.kind === "charge" && !robot.moving) {
      bolt(g, 0, -size * 0.95, s * 0.22, color);
    }
    // carried items
    robot.carried.slice(0, 6).forEach((item, i) => {
      g.fillStyle = TIER[item.tier];
      g.beginPath();
      g.arc(-size * 0.32 + i * size * 0.13, size * 0.62, size * 0.09, 0, Math.PI * 2);
      g.fill();
    });
    if (label) {
      g.fillStyle = INK;
      g.font = `700 ${Math.max(8, s * 0.3)}px "JetBrains Mono", monospace`;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(`R${robot.id + 1}`, 0, -size * 1.05);
    }
    g.restore();
  }

  private drawSpotlight(cells: Cell[], radius: number) {
    const g = this.ctx;
    const { s } = this;
    const w = this.canvas.width / this.dpr;
    const h = this.canvas.height / this.dpr;
    g.save();
    g.fillStyle = "rgba(5,6,10,0.58)";
    g.beginPath();
    g.rect(0, 0, w, h);
    for (const [r, c] of cells) {
      const [x, y] = this.px(c, r);
      g.moveTo(x + s * radius, y);
      g.arc(x, y, s * radius, 0, Math.PI * 2, true);
    }
    g.fill("evenodd");
    g.strokeStyle = rgba(INK, 0.35);
    g.lineWidth = 1;
    for (const [r, c] of cells) {
      const [x, y] = this.px(c, r);
      g.beginPath();
      g.arc(x, y, s * radius, 0, Math.PI * 2);
      g.stroke();
    }
    g.restore();
  }

  get cellSize() {
    return this.s;
  }

  get mutedInk() {
    return MUTED;
  }
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

function bolt(g: CanvasRenderingContext2D, x: number, y: number, size: number, color: string) {
  g.save();
  g.translate(x, y);
  g.fillStyle = color;
  g.beginPath();
  g.moveTo(size * 0.15, -size);
  g.lineTo(-size * 0.55, size * 0.1);
  g.lineTo(-size * 0.05, size * 0.1);
  g.lineTo(-size * 0.2, size);
  g.lineTo(size * 0.55, -size * 0.15);
  g.lineTo(size * 0.05, -size * 0.15);
  g.closePath();
  g.fill();
  g.restore();
}

function star(g: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) {
  g.save();
  g.fillStyle = color;
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    g.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
  g.closePath();
  g.fill();
  g.restore();
}

export const easeOutCubic = (k: number) => 1 - Math.pow(1 - k, 3);
export const easeInCubic = (k: number) => k * k * k;
export const easeOutBack = (k: number) => {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(k - 1, 3) + c1 * Math.pow(k - 1, 2);
};
