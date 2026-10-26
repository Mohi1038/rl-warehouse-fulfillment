import { ScrollTrigger } from "gsap/ScrollTrigger";
import type { Cell, Episode } from "../data";
import { WarehouseRenderer, type Overlay } from "../renderer";
import { Replay } from "../sim";
import { reducedMotion, whileVisible } from "../util";

// scroll progress through the steps -> simulation tick
const ANCHORS: [number, number][] = [
  [0, 0],
  [0.18, 40],
  [0.38, 200],
  [0.58, 300],
  [0.76, 380],
  [1, 720],
];

function tickAt(p: number): number {
  for (let i = 1; i < ANCHORS.length; i++) {
    const [p0, t0] = ANCHORS[i - 1];
    const [p1, t1] = ANCHORS[i];
    if (p <= p1) return t0 + ((p - p0) / (p1 - p0)) * (t1 - t0);
  }
  return 720;
}

export function initWorld(ep: Episode) {
  const section = document.getElementById("world")!;
  const canvas = document.getElementById("world-canvas") as HTMLCanvasElement;
  const replay = new Replay(ep);
  const renderer = new WarehouseRenderer(canvas, ep.layout);
  const hud = {
    t: document.getElementById("hud-t")!,
    rev: document.getElementById("hud-rev")!,
    ful: document.getElementById("hud-ful")!,
    lost: document.getElementById("hud-lost")!,
  };
  document.getElementById("world-final")!.textContent = `${ep.stats.revenue_captured_pct.toFixed(1)}%`;

  let focus = "layout";
  let target = 0;
  let t = 0;
  const steps = [...section.querySelectorAll<HTMLElement>(".step")];
  steps.forEach((step) =>
    ScrollTrigger.create({
      trigger: step,
      start: "top 60%",
      end: "bottom 40%",
      onToggle: (self) => {
        step.classList.toggle("is-active", self.isActive);
        if (self.isActive) focus = step.dataset.focus ?? "all";
      },
    }),
  );
  ScrollTrigger.create({
    trigger: section.querySelector(".steps")!,
    start: "top top",
    end: "bottom bottom",
    onUpdate: (self) => (target = tickAt(self.progress)),
  });

  const { layout } = ep;
  const docks: Cell[] = layout.docks;
  const dockAndTruck: Cell[] = docks.map(([r, c]) => [r + 0.8, c + 0.4]);

  const frame = () => {
    t += (target - t) * (reducedMotion ? 1 : 0.12);
    const world = replay.state(t);
    const overlay: Overlay = {};
    if (focus === "layout") {
      overlay.spotlight = [...docks, ...(layout.charger ? [layout.charger] : []), layout.packing];
      overlay.spotlightRadius = 1.3;
    } else if (focus === "trucks") {
      overlay.spotlight = dockAndTruck;
      overlay.spotlightRadius = 1.75;
    } else if (focus === "robot") {
      const r = world.robots[0];
      overlay.spotlight = [[r.y, r.x]];
      overlay.spotlightRadius = 2.4;
    }
    renderer.draw(world, overlay);
    hud.t.textContent = String(Math.floor(t)).padStart(3, "0");
    hud.rev.textContent = `${(world.arrived ? (100 * world.captured) / world.arrived : 0).toFixed(1)}%`;
    hud.ful.textContent = String(world.fulfilled);
    hud.lost.textContent = String(world.lost);
  };
  whileVisible(canvas, frame);
}
