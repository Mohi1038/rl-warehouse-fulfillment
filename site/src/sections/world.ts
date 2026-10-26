import { ScrollTrigger } from "gsap/ScrollTrigger";
import type { Episode } from "../data";
import { Replay } from "../sim";
import { reducedMotion, whileVisible } from "../util";
import { Warehouse3D, type ViewName } from "../warehouse3d";

// scroll progress through the steps -> simulation tick
const ANCHORS: [number, number][] = [
  [0, 0],
  [0.18, 60],
  [0.38, 210],
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
  const scene = new Warehouse3D(canvas, ep);
  scene.setView("overview", true);
  const hud = {
    t: document.getElementById("hud-t")!,
    rev: document.getElementById("hud-rev")!,
    ful: document.getElementById("hud-ful")!,
    lost: document.getElementById("hud-lost")!,
  };
  document.getElementById("world-final")!.textContent = `${ep.stats.revenue_captured_pct.toFixed(1)}%`;

  let target = 0;
  let t = 0;
  section.querySelectorAll<HTMLElement>(".step").forEach((step) =>
    ScrollTrigger.create({
      trigger: step,
      start: "top 60%",
      end: "bottom 40%",
      onToggle: (self) => {
        step.classList.toggle("is-active", self.isActive);
        if (self.isActive) scene.setView((step.dataset.focus ?? "all") as ViewName);
      },
    }),
  );
  ScrollTrigger.create({
    trigger: section.querySelector(".steps")!,
    start: "top top",
    end: "bottom bottom",
    onUpdate: (self) => (target = tickAt(self.progress)),
  });

  whileVisible(canvas, (dt) => {
    t += (target - t) * (reducedMotion ? 1 : 0.1);
    // snap when close, so the end-of-shift trucks (which leave at exactly tick 720) are counted
    if (Math.abs(target - t) < 0.3) t = target;
    const world = replay.state(t);
    scene.frame(world, dt);
    hud.t.textContent = String(Math.floor(t)).padStart(3, "0");
    hud.rev.textContent = `${(world.arrived ? (100 * world.captured) / world.arrived : 0).toFixed(1)}%`;
    hud.ful.textContent = String(world.fulfilled);
    hud.lost.textContent = String(world.lost);
  });
}
