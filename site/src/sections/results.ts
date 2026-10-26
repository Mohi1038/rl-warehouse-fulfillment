import * as d3 from "d3";
import { gsap } from "gsap";
import type { Summary } from "../data";
import { hideTip, reducedMotion, showTip } from "../util";

interface Row {
  name: string;
  revenue: number;
  taskReturn: number;
  express: number;
  ours: boolean;
  note?: string;
}

export function initResults(summary: Summary) {
  const nominal = summary.agents.nominal;
  const label: Record<string, string> = {
    ppo_attn: "PPO-Attention",
    ppo_attn_smdp: "PPO-Attention (original γ^τ)",
    dqn: "DQN",
    greedy: "Greedy",
    fifo: "FIFO",
    ppo_mlp: "PPO-MLP",
    edf: "EDF",
    edf_charge: "EDF + charge",
    random: "Random",
  };
  const rows: Row[] = Object.entries(nominal)
    .filter(([k]) => k in label)
    .map(([k, v]) => ({
      name: label[k],
      revenue: v.revenue_captured_pct,
      taskReturn: v.task_return,
      express: v.express_on_time_pct,
      ours: k === "ppo_attn",
    }));
  const la = summary.lookahead.with;
  rows.push({
    name: "PPO-Attention + look-ahead",
    revenue: la.revenue_captured_pct,
    taskReturn: la.task_return,
    express: la.express_on_time_pct,
    ours: true,
    note: "best checkpoint, 32 sampled futures per decision",
  });
  rows.sort((a, b) => b.revenue - a.revenue);

  const fig = document.getElementById("results-chart")!;
  const width = fig.clientWidth || 900;
  const rowH = 38;
  const m = { top: 10, right: 70, bottom: 30, left: Math.min(250, width * 0.36) };
  const height = m.top + m.bottom + rows.length * rowH;
  const x = d3
    .scaleLinear()
    .domain([0, 100])
    .range([m.left, width - m.right]);
  const y = d3
    .scaleBand()
    .domain(rows.map((r) => r.name))
    .range([m.top, height - m.bottom])
    .padding(0.32);
  const svg = d3.select(fig).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  svg
    .append("g")
    .attr("class", "grid")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(
      d3
        .axisBottom(x)
        .ticks(5)
        .tickSize(-(height - m.top - m.bottom))
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
        .tickFormat((d) => `${d}%`)
        .tickSizeOuter(0),
    );
  const g = svg
    .append("g")
    .selectAll("g")
    .data(rows)
    .join("g")
    .attr("transform", (r) => `translate(0,${y(r.name)})`);
  g.append("text")
    .attr("class", "chart-label")
    .attr("x", m.left - 12)
    .attr("y", y.bandwidth() / 2 + 4)
    .attr("text-anchor", "end")
    .attr("fill", (r) => (r.ours ? "#f0efec" : "#c3c2b7"))
    .text((r) => r.name);
  const bars = g
    .append("rect")
    .attr("x", m.left)
    .attr("height", y.bandwidth())
    .attr("rx", 4)
    .attr("width", 0)
    .attr("fill", (r) => (r.ours ? "#9085e9" : "rgba(195,194,183,0.32)"))
    .style("filter", (r) => (r.ours ? "drop-shadow(0 0 12px rgba(144,133,233,0.55))" : "none"));
  const values = g
    .append("text")
    .attr("class", "chart-value")
    .attr("x", m.left + 8)
    .attr("y", y.bandwidth() / 2 + 5)
    .text("");
  g.append("rect")
    .attr("x", 0)
    .attr("width", width)
    .attr("height", y.bandwidth())
    .attr("fill", "transparent")
    .on("mousemove", (e: MouseEvent, r) =>
      showTip(
        `${r.name}<br>revenue ${r.revenue.toFixed(1)}% · task return ${r.taskReturn.toFixed(1)}<br>express on time ${r.express.toFixed(1)}%${r.note ? `<br><span style="color:#898781">${r.note}</span>` : ""}`,
        e.clientX,
        e.clientY,
      ),
    )
    .on("mouseleave", hideTip);

  const grow = () => {
    bars
      .transition()
      .duration(reducedMotion ? 0 : 1400)
      .delay((_, i) => (reducedMotion ? 0 : i * 80))
      .ease(d3.easeExpOut)
      .attr("width", (r) => x(r.revenue) - m.left);
    values.each(function (r, i) {
      const el = this as SVGTextElement;
      const obj = { v: 0 };
      gsap.to(obj, {
        v: r.revenue,
        duration: reducedMotion ? 0 : 1.4,
        delay: reducedMotion ? 0 : i * 0.08,
        ease: "expo.out",
        onUpdate: () => {
          el.textContent = `${obj.v.toFixed(1)}%`;
          el.setAttribute("x", String(x(obj.v) + 8));
        },
      });
    });
  };
  gsap.to({}, { scrollTrigger: { trigger: fig, start: "top 80%", once: true, onEnter: grow } });

  document.getElementById("results-table")!.innerHTML = `<table><thead><tr><th>Agent</th><th>Revenue</th><th>Task return</th><th>Express on time</th></tr></thead><tbody>${rows
    .map((r) => `<tr><td>${r.name}</td><td>${r.revenue.toFixed(1)}%</td><td>${r.taskReturn.toFixed(1)}</td><td>${r.express.toFixed(1)}%</td></tr>`)
    .join("")}</tbody></table>`;
}
