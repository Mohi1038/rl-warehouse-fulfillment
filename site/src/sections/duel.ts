import type { Episode } from "../data";
import { WarehouseRenderer } from "../renderer";
import { Replay } from "../sim";
import { reducedMotion, whileVisible } from "../util";

const DURATION = 56; // seconds for a full shift
const HOLD = 3; // seconds paused on the final score
const START = 100; // tick the replay starts from

export function initDuel(ppo: Episode, opponents: Record<string, Episode>) {
  const a = new Replay(ppo);
  const ra = new WarehouseRenderer(document.getElementById("duel-a-canvas") as HTMLCanvasElement, ppo.layout);
  const canvasB = document.getElementById("duel-b-canvas") as HTMLCanvasElement;
  const rb = new WarehouseRenderer(canvasB, ppo.layout);
  const replays: Record<string, Replay> = Object.fromEntries(Object.entries(opponents).map(([k, ep]) => [k, new Replay(ep)]));
  let b = replays.greedy;
  let elapsed = 0;
  const outA = document.getElementById("duel-a")!;
  const outB = document.getElementById("duel-b")!;
  const nameB = document.getElementById("duel-b-name")!;
  const bar = document.getElementById("duel-clock-bar")!;

  const pct = (r: Replay, t: number) => {
    if (t >= r.horizon) return r.finalPct;
    const s = r.state(t);
    return s.arrived ? (100 * s.captured) / s.arrived : 0;
  };
  const render = (t: number) => {
    const sa = a.state(t);
    const sb = b.state(t);
    ra.draw(sa, { countdowns: true });
    rb.draw(sb, { countdowns: true, robotColors: ["#7c7c79"] });
    outA.textContent = `${pct(a, t).toFixed(1)}%`;
    outB.textContent = `${pct(b, t).toFixed(1)}%`;
    bar.style.transform = `scaleX(${t / a.horizon})`;
  };

  document.querySelectorAll<HTMLButtonElement>("[data-opponent]").forEach((btn) =>
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-opponent]").forEach((x) => x.setAttribute("aria-selected", String(x === btn)));
      b = replays[btn.dataset.opponent!];
      nameB.textContent = btn.dataset.opponent === "edf" ? "EDF" : "Greedy";
      elapsed = 0;
    }),
  );
  document.querySelector("[data-replay]")!.addEventListener("click", () => (elapsed = 0));

  if (reducedMotion) {
    render(a.horizon);
    return;
  }
  whileVisible(
    canvasB,
    (dt) => {
      elapsed += dt;
      if (elapsed > DURATION + HOLD) elapsed = 0;
      // skip the quiet first ticks before the first orders arrive
      render(Math.min(a.horizon, START + (elapsed / DURATION) * (a.horizon - START)));
    },
    () => (elapsed = 0),
  );
}
