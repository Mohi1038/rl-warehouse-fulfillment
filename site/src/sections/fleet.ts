import * as d3 from "d3";
import { gsap } from "gsap";
import type { Episode, Summary } from "../data";
import { WarehouseRenderer } from "../renderer";
import { Replay } from "../sim";
import { hideTip, reducedMotion, showTip, whileVisible } from "../util";

const SERIES: [string, string, string][] = [
  ["ppo_attn", "PPO-Attention (shared)", "#9085e9"],
  ["greedy", "Greedy", "#c3c2b7"],
  ["edf", "EDF", "#5a5953"],
];

export function initFleet(ep: Episode, summary: Summary) {
  const canvas = document.getElementById("fleet-canvas") as HTMLCanvasElement;
  const replay = new Replay(ep);
  const renderer = new WarehouseRenderer(canvas, ep.layout);
  document.getElementById("fleet-pct")!.textContent = `${ep.stats.revenue_captured_pct.toFixed(1)}%`;
  const start = 100; // skip the quiet first ticks before orders arrive
  let t = start;
  if (reducedMotion) renderer.draw(replay.state(360), { robotLabels: true });
  else
    whileVisible(canvas, (dt) => {
      t += dt * 30;
      if (t > replay.horizon + 30) t = start;
      renderer.draw(replay.state(Math.min(t, replay.horizon)), { robotLabels: true });
    });

  // task return by fleet size
  const fig = document.getElementById("fleet-chart")!;
  const width = Math.min(720, fig.clientWidth || 720);
  const height = 240;
  const m = { top: 30, right: 10, bottom: 34, left: 46 };
  const ks = ["k1", "k2", "k3"];
  const x0 = d3
    .scaleBand()
    .domain(ks)
    .range([m.left, width - m.right])
    .padding(0.28);
  const x1 = d3
    .scaleBand()
    .domain(SERIES.map((s) => s[0]))
    .range([0, x0.bandwidth()])
    .padding(0.12);
  const values = ks.flatMap((k) => SERIES.map(([a]) => summary.fleet[k]?.[a] ?? 0));
  const y = d3
    .scaleLinear()
    .domain([Math.min(0, d3.min(values) ?? 0), (d3.max(values) ?? 1) * 1.1])
    .nice()
    .range([height - m.bottom, m.top]);
  const svg = d3.select(fig).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${y(0)})`)
    .call(
      d3
        .axisBottom(x0)
        .tickFormat((k) => `${k.slice(1)} robot${k === "k1" ? "" : "s"}`)
        .tickSizeOuter(0),
    );
  svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(4))
    .call((g) => g.select(".domain").remove());
  const bars = svg
    .append("g")
    .selectAll("g")
    .data(ks)
    .join("g")
    .attr("transform", (k) => `translate(${x0(k)},0)`)
    .selectAll("rect")
    .data((k) => SERIES.map(([a, name, c]) => ({ k, a, name, c, v: summary.fleet[k]?.[a] ?? 0 })))
    .join("rect")
    .attr("x", (d) => x1(d.a)!)
    .attr("width", x1.bandwidth())
    .attr("y", y(0))
    .attr("height", 0)
    .attr("rx", 4)
    .attr("fill", (d) => d.c)
    .on("mousemove", (e: MouseEvent, d) => showTip(`${d.name} · ${d.k.slice(1)} robot(s)<br>task return ${d.v.toFixed(1)}`, e.clientX, e.clientY))
    .on("mouseleave", hideTip);
  const legend = svg.append("g").attr("transform", `translate(${m.left},12)`);
  let offset = 0;
  SERIES.forEach(([, name, c]) => {
    const g = legend.append("g").attr("transform", `translate(${offset},0)`);
    offset += 34 + name.length * 7.4;
    g.append("rect").attr("width", 10).attr("height", 10).attr("y", -9).attr("rx", 2).attr("fill", c);
    g.append("text").attr("class", "chart-label").attr("x", 16).text(name);
  });
  const grow = () =>
    bars
      .transition()
      .duration(reducedMotion ? 0 : 1100)
      .delay((_, i) => (reducedMotion ? 0 : i * 70))
      .ease(d3.easeExpOut)
      .attr("y", (d) => Math.min(y(0), y(d.v)))
      .attr("height", (d) => Math.abs(y(d.v) - y(0)));
  gsap.to({}, { scrollTrigger: { trigger: fig, start: "top 80%", once: true, onEnter: grow } });
}
