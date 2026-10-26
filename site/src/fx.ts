// Page-level motion: preloader, line reveals, cursor, magnetic links, marquee.

import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { reducedMotion } from "./util";

/** Split an element's text into masked lines (keeps inline elements such as <em>). */
export function splitLines(el: HTMLElement): HTMLElement[] {
  const units: HTMLElement[] = [];
  const walk = (node: Node, wrap: (span: HTMLElement) => HTMLElement) => {
    node.childNodes.forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        for (const word of (child.textContent ?? "").split(/\s+/).filter(Boolean)) {
          const span = document.createElement("span");
          span.textContent = word;
          units.push(wrap(span));
        }
      } else if (child instanceof HTMLBRElement) {
        const br = document.createElement("br");
        units.push(br as unknown as HTMLElement);
      } else if (child instanceof HTMLElement) {
        walk(child, (span) => {
          const clone = child.cloneNode(false) as HTMLElement;
          clone.appendChild(span);
          return wrap(clone);
        });
      }
    });
  };
  walk(el, (span) => span);
  el.innerHTML = "";
  units.forEach((u, i) => {
    el.appendChild(u);
    if (!(u instanceof HTMLBRElement) && i < units.length - 1) el.appendChild(document.createTextNode(" "));
  });
  // group by rendered line
  const lines: HTMLElement[][] = [];
  let top = -Infinity;
  for (const u of units) {
    if (u instanceof HTMLBRElement) {
      top = -Infinity;
      continue;
    }
    const t = u.offsetTop;
    if (t > top + 2) {
      lines.push([]);
      top = t;
    }
    lines[lines.length - 1].push(u);
  }
  el.innerHTML = "";
  return lines.map((line) => {
    const outer = document.createElement("span");
    outer.className = "line";
    const inner = document.createElement("span");
    inner.className = "line-inner";
    line.forEach((u, i) => {
      inner.appendChild(u);
      if (i < line.length - 1) inner.appendChild(document.createTextNode(" "));
    });
    outer.appendChild(inner);
    el.appendChild(outer);
    return inner;
  });
}

export function initReveals() {
  document.querySelectorAll<HTMLElement>("[data-lines]").forEach((el) => {
    const lines = splitLines(el);
    if (reducedMotion) return;
    const intro = el.closest("#hero");
    gsap.set(lines, { yPercent: 110 });
    const anim = () => gsap.to(lines, { yPercent: 0, duration: 1.4, ease: "expo.out", stagger: 0.09 });
    if (intro) document.addEventListener("intro", anim, { once: true });
    else ScrollTrigger.create({ trigger: el, start: "top 88%", once: true, onEnter: anim });
  });
  document.querySelectorAll<HTMLElement>("[data-fade]").forEach((el) => {
    if (reducedMotion) return;
    const intro = el.closest("#hero");
    gsap.set(el, { y: 30, opacity: 0 });
    const anim = () => gsap.to(el, { y: 0, opacity: 1, duration: 1.3, ease: "expo.out", delay: intro ? 0.45 : 0.1 });
    if (intro) document.addEventListener("intro", anim, { once: true });
    else ScrollTrigger.create({ trigger: el, start: "top 90%", once: true, onEnter: anim });
  });
}

/** Counter + curtain. ``progress`` resolves as data loads; resolves when the page is revealed. */
export function runLoader(progress: Promise<unknown>[]): Promise<void> {
  const loader = document.getElementById("loader")!;
  const num = document.getElementById("loader-num")!;
  const bar = document.getElementById("loader-bar")!;
  const state = { v: 0 };
  let done = 0;
  const target = () => (100 * done) / progress.length;
  const draw = () => {
    num.textContent = String(Math.round(state.v)).padStart(3, "0");
    bar.style.transform = `scaleX(${state.v / 100})`;
  };
  progress.forEach((p) =>
    p.then(() => {
      done += 1;
      gsap.to(state, { v: target(), duration: reducedMotion ? 0 : 0.8, ease: "power2.out", onUpdate: draw, overwrite: true });
    }),
  );
  return Promise.all(progress).then(
    () =>
      new Promise<void>((resolve) => {
        const tl = gsap.timeline({
          onComplete: () => {
            loader.remove();
            document.body.classList.remove("is-loading");
            resolve();
          },
        });
        if (reducedMotion) {
          tl.set(loader, { opacity: 0 });
          document.dispatchEvent(new Event("intro"));
          return;
        }
        tl.to(state, { v: 100, duration: 0.6, ease: "power2.out", onUpdate: draw })
          .to(".loader-count, .loader-top", { yPercent: -40, opacity: 0, duration: 0.7, ease: "expo.in" }, "+=0.15")
          .to(loader, { clipPath: "inset(0 0 100% 0)", duration: 1.2, ease: "expo.inOut" }, "-=0.25")
          .add(() => {
            document.dispatchEvent(new Event("intro"));
          }, "-=0.75");
      }),
  );
}

export function initCursor() {
  if (reducedMotion || !window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
  const cursor = document.querySelector<HTMLElement>(".cursor")!;
  const dot = cursor.querySelector<HTMLElement>(".cursor-dot")!;
  const ring = cursor.querySelector<HTMLElement>(".cursor-ring")!;
  const dx = gsap.quickTo(dot, "x", { duration: 0.08 });
  const dy = gsap.quickTo(dot, "y", { duration: 0.08 });
  const rx = gsap.quickTo(ring, "x", { duration: 0.45, ease: "power3" });
  const ry = gsap.quickTo(ring, "y", { duration: 0.45, ease: "power3" });
  window.addEventListener("pointermove", (e) => {
    cursor.classList.add("is-ready");
    dx(e.clientX);
    dy(e.clientY);
    rx(e.clientX);
    ry(e.clientY);
  });
  document.addEventListener("pointerover", (e) => {
    cursor.classList.toggle("is-hover", !!(e.target as HTMLElement).closest("a, button, summary, [data-magnetic]"));
  });
}

export function initMagnetic() {
  if (reducedMotion) return;
  document.querySelectorAll<HTMLElement>("[data-magnetic]").forEach((el) => {
    const x = gsap.quickTo(el, "x", { duration: 0.6, ease: "elastic.out(1, 0.4)" });
    const y = gsap.quickTo(el, "y", { duration: 0.6, ease: "elastic.out(1, 0.4)" });
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      x((e.clientX - (r.left + r.width / 2)) * 0.35);
      y((e.clientY - (r.top + r.height / 2)) * 0.35);
    });
    el.addEventListener("pointerleave", () => {
      x(0);
      y(0);
    });
  });
}

export function initMarquee() {
  const track = document.querySelector<HTMLElement>(".marquee-track");
  if (!track || reducedMotion) return;
  const loop = gsap.to(track, { xPercent: -50, duration: 38, ease: "none", repeat: -1 });
  // scrolling kicks the marquee faster (and backwards when scrolling up); the kick decays
  let boost = 1;
  let direction = 1;
  ScrollTrigger.create({
    onUpdate: (self) => {
      boost = Math.max(boost, 1 + Math.min(6, Math.abs(self.getVelocity()) / 400));
      direction = self.direction < 0 ? -1 : 1;
    },
  });
  gsap.ticker.add(() => {
    boost += (1 - boost) * 0.05;
    loop.timeScale(direction * boost);
  });
}

/** Story steps fade in as they rise to the middle of the screen and out as they leave (wide screens). */
export function initStepFades() {
  if (reducedMotion || !window.matchMedia("(min-width: 901px)").matches) return;
  document.querySelectorAll<HTMLElement>(".step").forEach((step) => {
    gsap.set(step, { opacity: 0 });
    ScrollTrigger.create({
      trigger: step,
      start: "top bottom",
      end: "bottom top",
      onUpdate: (self) => {
        const d = Math.abs(self.progress - 0.5) * 2; // 0 at centre, 1 at the edges
        gsap.set(step, { opacity: Math.max(0, Math.min(1, (1 - d) * 2.2)), y: (self.progress - 0.5) * -40 });
      },
    });
  });
}

export function initCounters() {
  document.querySelectorAll<HTMLElement>("[data-count]").forEach((el, i) => {
    const target = Number(el.dataset.count);
    const suffix = el.dataset.suffix ?? "";
    const obj = { v: 0 };
    const run = () =>
      gsap.to(obj, {
        v: target,
        duration: reducedMotion ? 0 : 2.4,
        delay: reducedMotion ? 0 : 0.8 + i * 0.12,
        ease: "expo.out",
        onUpdate: () => {
          el.textContent = `${obj.v.toFixed(1)}${suffix}`;
        },
      });
    document.addEventListener("intro", run, { once: true });
  });
}
