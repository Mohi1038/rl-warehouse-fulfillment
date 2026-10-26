import * as d3 from "d3";
import { gsap } from "gsap";
import { TIER_NAMES, type Episode } from "../data";
import { TIER } from "../palette";
import { hideTip, reducedMotion, showTip } from "../util";

export function initArrivals(ep: Episode) {
  const fig = document.getElementById("arrivals-chart")!;
  const width = fig.clientWidth || 900;
  const height = 380;
  const m = { top: 24, right: 20, bottom: 40, left: 52 };
  const x = d3.scaleLinear().domain([0, ep.horizon]).range([m.left, width - m.right]);
  const maxRate = d3.max(ep.rate, (d) => d[1]) ?? 0.1;
  const y = d3
    .scaleLinear()
    .domain([0, maxRate * 1.15])
    .range([height - m.bottom, m.top]);
  const rateAt = (t: number) => {
    const i = Math.min(ep.rate.length - 1, Math.max(0, Math.round(t / 6)));
    return ep.rate[i][1];
  };
  const svg = d3.select(fig).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  const defs = svg.append("defs");
  const grad = defs.append("linearGradient").attr("id", "rate-grad").attr("x1", 0).attr("x2", 0).attr("y1", 0).attr("y2", 1);
  grad.append("stop").attr("offset", "0%").attr("stop-color", "#9085e9").attr("stop-opacity", 0.35);
  grad.append("stop").attr("offset", "100%").attr("stop-color", "#9085e9").attr("stop-opacity", 0);
  const clip = defs.append("clipPath").attr("id", "rate-clip").append("rect").attr("x", 0).attr("y", 0).attr("height", height).attr("width", 0);

  svg
    .append("g")
    .attr("class", "grid")
    .attr("transform", `translate(${m.left},0)`)
    .call(
      d3
        .axisLeft(y)
        .ticks(4)
        .tickSize(-(width - m.left - m.right))
        .tickFormat(() => ""),
    )
    .call((g) => g.select(".domain").remove());
  svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${height - m.bottom})`)
    .call(d3.axisBottom(x).ticks(8).tickSizeOuter(0));
  svg
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(${m.left},0)`)
    .call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".2f")))
    .call((g) => g.select(".domain").remove());
  svg.append("text").attr("class", "chart-label").attr("x", width - m.right).attr("y", height - 6).attr("text-anchor", "end").text("tick of the shift");
  svg.append("text").attr("class", "chart-label").attr("x", m.left).attr("y", 12).text("orders per tick, λ(t)");

  const body = svg.append("g").attr("clip-path", "url(#rate-clip)");
  const area = d3
    .area<[number, number]>()
    .x((d) => x(d[0]))
    .y0(height - m.bottom)
    .y1((d) => y(d[1]))
    .curve(d3.curveCatmullRom);
  const line = d3
    .line<[number, number]>()
    .x((d) => x(d[0]))
    .y((d) => y(d[1]))
    .curve(d3.curveCatmullRom);
  body.append("path").datum(ep.rate).attr("d", area).attr("fill", "url(#rate-grad)");
  body.append("path").datum(ep.rate).attr("d", line).attr("fill", "none").attr("stroke", "#9085e9").attr("stroke-width", 2);

  const r = d3
    .scaleSqrt()
    .domain([0, d3.max(ep.orders, (o) => o.value) ?? 1])
    .range([4, 13]);
  const dots = body
    .append("g")
    .selectAll("circle")
    .data(ep.orders)
    .join("circle")
    .attr("cx", (o) => x(o.arrival))
    .attr("cy", (o) => y(rateAt(o.arrival)))
    .attr("r", (o) => r(o.value))
    .attr("fill", (o) => TIER[o.tier])
    .attr("stroke", "#0b0d12")
    .attr("stroke-width", 2);
  // generous invisible hit targets
  body
    .append("g")
    .selectAll("circle")
    .data(ep.orders)
    .join("circle")
    .attr("cx", (o) => x(o.arrival))
    .attr("cy", (o) => y(rateAt(o.arrival)))
    .attr("r", 16)
    .attr("fill", "transparent")
    .style("cursor", "default")
    .on("mousemove", (event: MouseEvent, o) => {
      const fate = o.status === "fulfilled" ? (o.deferrals ? "fulfilled late (½ value)" : "fulfilled") : o.status === "lost" ? "lost" : "missed";
      showTip(
        `Order #${o.id} · ${TIER_NAMES[o.tier]}${o.vip ? " ★ VIP" : ""}<br>value ${o.value.toFixed(1)} · arrived t=${Math.round(o.arrival)}<br>${fate}`,
        event.clientX,
        event.clientY,
      );
    })
    .on("mouseleave", hideTip);

  // legend
  const legend = svg.append("g").attr("transform", `translate(${width - m.right - 300},${m.top})`);
  TIER_NAMES.forEach((name, i) => {
    const g = legend.append("g").attr("transform", `translate(${i * 100},0)`);
    g.append("circle").attr("r", 5).attr("cy", -4).attr("fill", TIER[i]);
    g.append("text").attr("class", "chart-label").attr("x", 10).text(name);
  });

  if (reducedMotion) {
    clip.attr("width", width);
    return;
  }
  const state = { w: 0 };
  gsap.to(state, {
    w: width,
    ease: "none",
    scrollTrigger: { trigger: fig, start: "top 80%", end: "bottom 45%", scrub: 0.6 },
    onUpdate: () => {
      clip.attr("width", state.w);
    },
  });
  dots.attr("r", 0);
  gsap.to(
    {},
    {
      scrollTrigger: {
        trigger: fig,
        start: "top 80%",
        end: "bottom 45%",
        scrub: 0.6,
        onUpdate: (self) => {
          const edge = x.invert(m.left + self.progress * (width - m.left));
          dots.attr("r", (o) => (o.arrival <= edge ? r(o.value) : 0));
        },
      },
    },
  );
}
