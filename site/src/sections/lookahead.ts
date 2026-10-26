import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { TIER_NAMES, type Cell, type Episode, type LookaheadDecision, type Summary } from "../data";
import { WarehouseRenderer, type Ghost } from "../renderer";
import { Replay, gridPath } from "../sim";
import { reducedMotion, whileVisible } from "../util";

function chooseDecisions(ep: Episode): LookaheadDecision[] {
  const all = (ep.lookahead ?? []).filter((d) => d.t > 40 && d.cands.length >= 3);
  const spread = (d: LookaheadDecision) => Math.max(...d.cands.map((c) => c.v)) - Math.min(...d.cands.map((c) => c.v));
  const agree = all.filter((d) => d.chosen === d.base);
  const first = agree[0] ?? all[0];
  const second = [...agree].filter((d) => d !== first && d.t > first.t).sort((a, b) => b.cands.length - a.cands.length || spread(b) - spread(a))[0] ?? first;
  const override = all
    .filter((d) => d.chosen !== d.base)
    .sort((a, b) => {
      const gap = (d: LookaheadDecision) => (d.cands.find((c) => c.a === d.chosen)?.v ?? 0) - (d.cands.find((c) => c.a === d.base)?.v ?? 0);
      return gap(b) - gap(a);
    })[0];
  return [first, second, override ?? second];
}

export function initLookahead(ep: Episode, summary: Summary) {
  const section = document.getElementById("lookahead")!;
  const canvas = document.getElementById("la-canvas") as HTMLCanvasElement;
  const panel = document.getElementById("la-panel")!;
  const replay = new Replay(ep);
  const renderer = new WarehouseRenderer(canvas, ep.layout);
  const decisions = chooseDecisions(ep);
  const byId = new Map(ep.orders.map((o) => [o.id, o]));
  const look = summary.lookahead;

  const view = { t: decisions[0].t - 30, reveal: 0 };
  let ghosts: (Ghost & { full: Cell[] })[] = [];

  const label = (a: string) => {
    const [kind, arg] = a.split(":");
    if (kind === "pick") {
      const o = byId.get(Number(arg));
      return `Pick #${arg}${o ? ` · ${TIER_NAMES[o.tier]}` : ""}`;
    }
    if (kind === "deliver") return `Deliver D${Number(arg) + 1}`;
    return kind === "charge" ? "Charge" : "Wait";
  };

  const target = (a: string, from: Cell, t: number): Cell => {
    const [kind, arg] = a.split(":");
    if (kind === "pick") {
      const o = byId.get(Number(arg));
      const items = (o?.items ?? []).filter((i) => i.pick === null || i.pick > t);
      let best: Cell = from;
      let bestLen = Infinity;
      for (const i of items) {
        const len = gridPath(ep.layout.grid, from, i.cell).length;
        if (len < bestLen) {
          bestLen = len;
          best = i.cell;
        }
      }
      return best;
    }
    if (kind === "deliver") return ep.layout.docks[Number(arg)];
    if (kind === "charge" && ep.layout.charger) return ep.layout.charger;
    return from;
  };

  const show = (index: number) => {
    let d = decisions[index];
    const total = d.cands.length;
    const decision = ep.options.find((o) => Math.abs(o.t0 - d.t) < 1e-6);
    const from: Cell = decision?.path[0] ?? ep.layout.packing;
    // the five best actions, plus the policy's own pick when it is not among them
    const ranked = [...d.cands].sort((a, b) => b.v - a.v);
    const shown = ranked.slice(0, 5);
    const base = ranked.find((c) => c.a === d.base);
    if (base && !shown.includes(base)) shown.push(base);
    const omitted = d.cands.length - shown.length;
    d = { ...d, cands: shown };
    const vs = d.cands.map((c) => c.v);
    const lo = Math.min(...vs);
    const hi = Math.max(...vs);
    ghosts = d.cands.map((c) => {
      const full = gridPath(ep.layout.grid, from, target(c.a, from, d.t));
      return { path: full, full, score: hi > lo ? (c.v - lo) / (hi - lo) : 1, chosen: index > 0 && c.a === d.chosen, label: c.a };
    });
    const rows = [...d.cands]
      .sort((a, b) => b.v - a.v)
      .map((c) => {
        const width = hi > lo ? 18 + 82 * ((c.v - lo) / (hi - lo)) : 100;
        const tags = [c.a === d.chosen && index > 0 ? "look-ahead" : "", c.a === d.base && index === 2 ? "PPO's pick" : ""].filter(Boolean).join(" · ");
        return `<div class="la-row${c.a === d.chosen && index > 0 ? " win" : ""}"><span>${label(c.a)}</span><div class="bar" style="width:${index === 0 ? 100 : width}%;${index === 0 ? "opacity:.25" : ""}"></div><span class="v">${index === 0 ? "?" : c.v.toFixed(1)}</span></div>${tags ? `<div class="la-note">↳ ${tags}</div>` : ""}`;
      })
      .join("");
    const head =
      index === 0 ? `Decision at tick ${Math.round(d.t)} · ${total} valid actions` : `Revenue by end of shift, mean of 32 sampled futures · tick ${Math.round(d.t)}`;
    const more = omitted > 0 ? `<div class="la-note">+ ${omitted} lower-scoring actions not shown</div>` : "";
    const foot =
      index === 2
        ? `<div class="la-note">100 held-out seeds: ${look.without.revenue_captured_pct.toFixed(1)}% → <b style="color:var(--ink)">${look.with.revenue_captured_pct.toFixed(1)}%</b> revenue</div>`
        : "";
    panel.innerHTML = `<h4>${head}</h4>${rows}${more}${foot}`;
    gsap.from(panel.querySelectorAll(".bar"), { scaleX: 0, duration: reducedMotion ? 0 : 0.9, ease: "expo.out", stagger: 0.06, delay: 0.6 });
    gsap.killTweensOf(view);
    view.reveal = 0;
    gsap.to(view, { t: d.t, duration: reducedMotion ? 0 : 1.2, ease: "power2.inOut" });
    gsap.to(view, { reveal: 1, duration: reducedMotion ? 0 : 1.1, ease: "power2.out", delay: reducedMotion ? 0 : 1.2 });
  };

  section.querySelectorAll<HTMLElement>(".step").forEach((step) =>
    ScrollTrigger.create({
      trigger: step,
      start: "top 60%",
      end: "bottom 40%",
      onToggle: (self) => {
        step.classList.toggle("is-active", self.isActive);
        if (self.isActive) show(Number(step.dataset.la));
      },
    }),
  );
  show(0);

  whileVisible(canvas, () => {
    const visible = ghosts.map((g) => ({ ...g, path: g.full.slice(0, Math.max(1, Math.ceil(g.full.length * view.reveal))) }));
    renderer.draw(replay.state(view.t), { ghosts: view.reveal > 0 ? visible : undefined, countdowns: true });
  });
}
