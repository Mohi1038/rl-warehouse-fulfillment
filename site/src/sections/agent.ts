import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { TIER_NAMES, type Episode, type Order } from "../data";
import { TIER } from "../palette";
import { gridPath } from "../sim";
import { reducedMotion } from "../util";

type Phase = "cards" | "tokens" | "attention" | "pointer";

interface Snapshot {
  t: number;
  weights: Record<string, number>;
  chosen: number | null;
  orders: Order[];
}

function pickSnapshot(ep: Episode): Snapshot {
  const picks = new Map(ep.options.filter((o) => o.k === "pick").map((o) => [o.t0, o.order!]));
  let best = ep.attention![0];
  let bestScore = -1;
  for (const a of ep.attention ?? []) {
    const n = Object.keys(a.w).length;
    const chosen = picks.get(a.t);
    const score = n + (chosen !== undefined ? 3 : 0) - (n > 9 ? (n - 9) * 2 : 0);
    if (score > bestScore) {
      bestScore = score;
      best = a;
    }
  }
  const byId = new Map(ep.orders.map((o) => [o.id, o]));
  const orders = Object.keys(best.w)
    .map((id) => byId.get(Number(id))!)
    .sort((a, b) => a.id - b.id);
  return { t: best.t, weights: best.w, chosen: picks.get(best.t) ?? null, orders };
}

export function initAgent(ep: Episode) {
  const section = document.getElementById("agent")!;
  const stage = document.getElementById("agent-stage")!;
  const cardsEl = document.getElementById("agent-cards")!;
  const beams = document.getElementById("agent-beams") as unknown as SVGSVGElement;
  const cls = document.getElementById("agent-cls")!;
  const actions = document.getElementById("agent-actions")!;
  const snap = pickSnapshot(ep);
  const truckById = new Map(ep.trucks.map((t) => [t.id, t]));
  const maxW = Math.max(...Object.values(snap.weights));
  const decision = ep.options.find((o) => o.t0 === snap.t);
  const robotCell = decision?.path[0] ?? ep.layout.packing;
  const load = ep.orders
    .flatMap((o) => o.items.filter((i) => i.pick !== null && i.pick <= snap.t && (i.load === null || i.load > snap.t) && o.resolved > snap.t))
    .reduce((sum, i) => sum + i.w, 0);
  const free = ep.capacity - load;

  const cards = snap.orders.map((o) => {
    const first = truckById.get(o.truck0)!;
    const truck = snap.t >= first.actual ? truckById.get(o.truck)! : first;
    const slack = Math.round(truck.scheduled - snap.t);
    const waiting = o.items.filter((i) => i.pick === null || i.pick > snap.t);
    const left = waiting.length;
    const age = Math.round(snap.t - o.arrival);
    const dist = Math.min(...waiting.map((i) => gridPath(ep.layout.grid, robotCell, i.cell).length - 1), 40);
    const fits = waiting.some((i) => i.w <= free) ? 1 : 0;
    const feats = [
      1,
      o.value / 50,
      o.tier === 0 ? 1 : 0,
      o.tier === 1 ? 1 : 0,
      o.tier === 2 ? 1 : 0,
      o.dock === 0 ? 1 : 0,
      o.dock === 1 ? 1 : 0,
      o.dock === 2 ? 1 : 0,
      Math.max(0, Math.min(1, slack / 200)),
      left / 4,
      (o.items.length - left) / 4,
      dist / 40,
      fits,
      Math.min(1, age / 200),
    ];
    const el = document.createElement("div");
    el.className = "ocard";
    el.style.setProperty("--c", TIER[o.tier]);
    el.innerHTML = `
      <div class="top"><span>#${o.id}${o.vip ? " ★" : ""}</span><span>${o.value.toFixed(1)}</span></div>
      <div>${TIER_NAMES[o.tier]} · D${o.dock + 1}</div>
      <div>slack ${slack}t · ${left} item${left === 1 ? "" : "s"} left</div>
      <div class="feats">${feats.map((f) => `<i style="--o:${0.15 + 0.85 * Math.max(0, Math.min(1, f))}"></i>`).join("")}</div>`;
    if (o.id === snap.chosen) el.dataset.chosen = "1";
    cardsEl.appendChild(el);
    return { el, order: o, w: snap.weights[String(o.id)] ?? 0 };
  });
  const chosenLabel = snap.chosen !== null ? `PICK #${snap.chosen}` : "WAIT";
  actions.innerHTML = [chosenLabel, "DELIVER", "CHARGE", "WAIT"]
    .map((a, i) => `<span${i === 0 ? ' style="color:var(--ink);border-color:var(--ink)"' : ""}>${a}</span>`)
    .join("");

  let phase: Phase = "cards";

  const layoutFor = (p: Phase) => {
    const W = stage.clientWidth;
    const H = stage.clientHeight;
    const cw = cards[0]?.el.offsetWidth ?? 168;
    const n = cards.length;
    if (p === "cards") {
      const cols = W < 600 ? 2 : 3;
      const gap = 14;
      const ch = cards[0]?.el.offsetHeight ?? 96;
      const rows = Math.ceil(n / cols);
      const gridW = cols * cw + (cols - 1) * gap;
      const gridH = rows * ch + (rows - 1) * gap;
      return cards.map((_, i) => ({
        x: (W - gridW) / 2 + (i % cols) * (cw + gap),
        y: (H - gridH) / 2 + Math.floor(i / cols) * (ch + gap),
        rotation: 0,
      }));
    }
    const rowH = Math.min(58, (H * 0.92) / n);
    return cards.map((_, i) => ({ x: Math.max(0, W * 0.04), y: (H - rowH * n) / 2 + i * rowH, rotation: 0, cw }));
  };

  const drawBeams = (show: boolean, pointer: boolean) => {
    const W = stage.clientWidth;
    const H = stage.clientHeight;
    beams.setAttribute("viewBox", `0 0 ${W} ${H}`);
    beams.innerHTML = "";
    if (!show) return;
    const cx = W * 0.66;
    const cy = H / 2;
    const ns = "http://www.w3.org/2000/svg";
    cards.forEach(({ el, w }) => {
      const x = (gsap.getProperty(el, "x") as number) + el.offsetWidth;
      const y = (gsap.getProperty(el, "y") as number) + el.offsetHeight / 2;
      const k = w / maxW;
      const path = document.createElementNS(ns, "path");
      const midX = (x + cx) / 2;
      path.setAttribute("d", `M${x},${y} C${midX},${y} ${midX},${cy} ${cx - 46},${cy}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#ededed");
      path.setAttribute("stroke-width", String(0.8 + 9 * k));
      path.setAttribute("stroke-opacity", String(0.12 + 0.8 * k));
      path.setAttribute("stroke-linecap", "round");
      beams.appendChild(path);
      const len = path.getTotalLength();
      gsap.fromTo(path, { strokeDasharray: len, strokeDashoffset: len }, { strokeDashoffset: 0, duration: reducedMotion ? 0 : 1.1, ease: "power2.inOut", delay: reducedMotion ? 0 : 0.2 + (1 - k) * 0.4 });
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", String(x + 10));
      label.setAttribute("y", String(y - 6));
      label.setAttribute("class", "chart-label");
      label.textContent = `${(w * 100).toFixed(0)}%`;
      beams.appendChild(label);
    });
    if (pointer) {
      const chosen = cards.find((c) => c.el.dataset.chosen);
      if (chosen) {
        // pointer: an arc from the [CLS] token over the beams to the order slot it selects
        const x = (gsap.getProperty(chosen.el, "x") as number) + chosen.el.offsetWidth;
        const y = gsap.getProperty(chosen.el, "y") as number;
        const top = Math.min(y, cy) - 70;
        const p = document.createElementNS(ns, "path");
        p.setAttribute("d", `M${cx},${cy - 48} C${cx},${top} ${x + 70},${top} ${x - 14},${y - 4}`);
        p.setAttribute("fill", "none");
        p.setAttribute("stroke", "#f0efec");
        p.setAttribute("stroke-width", "2.5");
        p.setAttribute("stroke-dasharray", "6 6");
        p.style.filter = "drop-shadow(0 0 8px rgba(240,239,236,0.8))";
        beams.appendChild(p);
        gsap.fromTo(p, { opacity: 0 }, { opacity: 1, duration: 0.6 });
        gsap.to(p, { strokeDashoffset: -48, duration: 1.2, repeat: -1, ease: "none" });
      }
    }
    gsap.set(cls, { left: cx - 46, top: cy - 46 });
  };

  let pendingBeams: gsap.core.Tween | null = null;
  const apply = (p: Phase, animate = true) => {
    phase = p;
    const pos = layoutFor(p);
    const dur = animate && !reducedMotion ? 1.5 : 0;
    // a fast scroll can fire several phases at once: each one cancels the last
    pendingBeams?.kill();
    gsap.killTweensOf([cls, actions, ...cards.map((c) => c.el), ...cards.map((c) => c.el.querySelector(".feats"))]);
    beams.innerHTML = "";
    cards.forEach(({ el }, i) => {
      el.classList.toggle("compact", p !== "cards");
      el.classList.toggle("chosen", p === "pointer" && !!el.dataset.chosen);
      gsap.to(el, {
        x: pos[i].x,
        y: pos[i].y,
        rotation: pos[i].rotation,
        opacity: p === "pointer" && !el.dataset.chosen ? 0.45 : 1,
        duration: dur,
        ease: "expo.inOut",
        delay: dur ? i * 0.025 : 0,
      });
      gsap.to(el.querySelector(".feats"), { opacity: p === "cards" ? 0 : 1, duration: dur * 0.6 });
    });
    gsap.to(cls, { opacity: p === "attention" || p === "pointer" ? 1 : 0, scale: p === "attention" || p === "pointer" ? 1 : 0.4, duration: dur * 0.8, ease: "back.out(2)" });
    gsap.to(actions, { opacity: p === "pointer" ? 1 : 0, y: p === "pointer" ? 0 : 20, duration: dur * 0.7 });
    pendingBeams = gsap.delayedCall(dur + cards.length * 0.025 + 0.05, () => drawBeams(p === "attention" || p === "pointer", p === "pointer"));
  };

  // initial scatter, entering from below
  const pos = layoutFor("cards");
  cards.forEach(({ el }, i) => gsap.set(el, { x: pos[i].x, y: pos[i].y + 60, rotation: pos[i].rotation, opacity: 0 }));
  ScrollTrigger.create({
    trigger: stage,
    start: "top 70%",
    once: true,
    onEnter: () => cards.forEach(({ el }, i) => gsap.to(el, { y: pos[i].y, opacity: 1, duration: reducedMotion ? 0 : 1, delay: reducedMotion ? 0 : i * 0.05, ease: "expo.out" })),
  });

  section.querySelectorAll<HTMLElement>(".step").forEach((step) =>
    ScrollTrigger.create({
      trigger: step,
      start: "top 60%",
      end: "bottom 40%",
      onToggle: (self) => {
        step.classList.toggle("is-active", self.isActive);
        const next = step.dataset.agent as Phase;
        if (self.isActive && next !== phase) apply(next);
      },
    }),
  );
  window.addEventListener("resize", () => apply(phase, false));
}
