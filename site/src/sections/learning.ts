import * as d3 from "d3";
import { gsap } from "gsap";
import type { Episode, Summary } from "../data";
import { MUTED } from "../palette";
import { hideTip, reducedMotion, showTip } from "../util";

const NEW = "#9085e9";
const GAMMA = 0.995;
const N = 64;

export function initLearning(ep: Episode, summary: Summary) {
  // --- discount weight of each decision in the real shift
  const fig = document.getElementById("discount-viz")!;
  const taus = ep.options.slice(0, N).map((o) => Math.max(1, o.t1 - o.t0));
  let elapsed = 0;
  const smdp = taus.map((tau) => {
    const w = Math.pow(GAMMA, elapsed);
    elapsed += tau;
    return w;
  });
  const perDecision = taus.map((_, n) => Math.pow(GAMMA, n));
  const horizon = smdp.findIndex((w) => w < 1 / Math.E);
  const rows: [string, number[], string][] = [
    ["γ^τ per tick (original)", smdp, MUTED],
    ["γ per decision (default)", perDecision, NEW],
  ];
  fig.innerHTML = rows
    .map(
      ([label, ws, c]) =>
        `<div class="row"><span>${label}</span><div class="dots">${ws
          .map((w, n) => `<i style="--c:${c};height:${(w * 100).toFixed(1)}%" data-n="${n}" data-w="${w.toFixed(3)}"></i>`)
          .join("")}</div></div>`,
    )
    .join("") +
    `<div class="row"><span></span><div class="scale"><span>decision 1</span><span>weight falls below 1/e after decision ${horizon + 1} (γ^τ)</span><span>decision ${N}</span></div></div>`;
  fig.querySelectorAll<HTMLElement>(".dots i").forEach((el) => {
    el.addEventListener("mousemove", (e) => showTip(`decision ${Number(el.dataset.n) + 1} · weight ${el.dataset.w}`, e.clientX, e.clientY));
    el.addEventListener("mouseleave", hideTip);
  });
  if (!reducedMotion)
    gsap.from(fig.querySelectorAll(".dots i"), {
      scaleY: 0,
      duration: 0.9,
      ease: "expo.out",
      stagger: { each: 0.012, from: "start" },
      scrollTrigger: { trigger: fig, start: "top 75%" },
    });

  // --- learning curves
  const chart = document.getElementById("curves-chart")!;
  const width = chart.clientWidth || 900;
  const height = 380;
  const m = { top: 24, right: 150, bottom: 42, left: 52 };
  const series: [string, string, [number, number][][]][] = [
    ["Per-decision γ (default)", NEW, summary.curves.ppo_attn],
    ["SMDP γ^τ (original)", MUTED, summary.curves.ppo_attn_smdp],
  ];
  const all = series.flatMap(([, , runs]) => runs.flat());
  const x = d3
    .scaleLinear()
    .domain([0, d3.max(all, (d) => d[0]) ?? 2e6])
    .range([m.left, width - m.right]);
  const y = d3
    .scaleLinear()
    .domain([Math.min(0, d3.min(all, (d) => d[1]) ?? 0), (d3.max(all, (d) => d[1]) ?? 160) * 1.05])
    .nice()
    .range([height - m.bottom, m.top]);
  const svg = d3.select(chart).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  svg
    .append("g")
    .attr("class", "grid")
    .attr("transform", `translate(${m.left},0)`)
    .call(
      d3
        .axisLeft(y)
        .ticks(5)
        .tickSize(-(width - m.left - m.right))
        .tickFormat(() => ""),
    )
    .call((g) => g.select(".domain").remove());
  svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(
      d3
        .axisBottom(x)
        .ticks(5)
        .tickFormat((d) => `${(+d / 1e6).toFixed(1)}M`)
        .tickSizeOuter(0),
    );
  svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(5))
    .call((g) => g.select(".domain").remove());
  svg.append("text").attr("class", "chart-label").attr("x", width - m.right).attr("y", height - 6).attr("text-anchor", "end").text("training decisions");
  svg.append("text").attr("class", "chart-label").attr("x", m.left).attr("y", 12).text("validation task return");
  const line = d3
    .line<[number, number]>()
    .x((d) => x(d[0]))
    .y((d) => y(d[1]))
    .curve(d3.curveMonotoneX);
  const means: { name: string; color: string; pts: [number, number][] }[] = [];
  const paths: SVGPathElement[] = [];
  series.forEach(([name, color, runs]) => {
    runs.forEach((run) => {
      const p = svg.append("path").datum(run).attr("d", line).attr("fill", "none").attr("stroke", color).attr("stroke-opacity", 0.22).attr("stroke-width", 1.2);
      paths.push(p.node()!);
    });
    // mean over seeds on the shortest common grid
    const len = Math.min(...runs.map((r) => r.length));
    const pts: [number, number][] = Array.from({ length: len }, (_, i) => [runs[0][i][0], d3.mean(runs, (r) => r[i][1]) ?? 0]);
    means.push({ name, color, pts });
    const p = svg.append("path").datum(pts).attr("d", line).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2.5);
    paths.push(p.node()!);
    const last = pts[pts.length - 1];
    svg
      .append("text")
      .attr("class", "chart-label")
      .attr("x", x(last[0]) + 8)
      .attr("y", y(last[1]) + 4)
      .text(name);
  });
  svg
    .append("text")
    .attr("class", "chart-label")
    .attr("x", m.left)
    .attr("y", height - m.bottom - 8)
    .attr("fill-opacity", 0.7)
    .text("thin: each of 5 seeds · thick: mean · validation sets differ (50 vs 20 episodes); held-out numbers below are comparable");

  // crosshair tooltip
  const cross = svg.append("line").attr("y1", m.top).attr("y2", height - m.bottom).attr("stroke", "rgba(240,239,236,0.3)").style("opacity", 0);
  svg
    .append("rect")
    .attr("x", m.left)
    .attr("y", m.top)
    .attr("width", width - m.left - m.right)
    .attr("height", height - m.top - m.bottom)
    .attr("fill", "transparent")
    .on("mousemove", (event: MouseEvent) => {
      const [mx] = d3.pointer(event);
      const xv = x.invert(mx);
      cross.attr("x1", mx).attr("x2", mx).style("opacity", 1);
      const rowsHtml = means
        .map((s) => {
          const i = d3.bisector((d: [number, number]) => d[0]).center(s.pts, xv);
          return `<span style="color:${s.color}">●</span> ${s.name}: ${s.pts[i][1].toFixed(1)}`;
        })
        .join("<br>");
      showTip(`${(xv / 1e6).toFixed(2)}M decisions<br>${rowsHtml}`, event.clientX, event.clientY);
    })
    .on("mouseleave", () => {
      cross.style("opacity", 0);
      hideTip();
    });

  if (!reducedMotion)
    paths.forEach((p, i) => {
      const len = p.getTotalLength();
      gsap.fromTo(
        p,
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, ease: "none", scrollTrigger: { trigger: chart, start: "top 80%", end: "bottom 50%", scrub: 0.8 }, delay: i * 0.01 },
      );
    });
}
