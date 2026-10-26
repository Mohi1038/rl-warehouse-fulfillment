// Shared helpers: visibility-gated animation loops, tooltip, motion preference.

export const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Calls ``frame(dt)`` every animation frame while ``el`` is on screen. */
export function whileVisible(el: Element, frame: (dt: number) => void, onShow?: () => void) {
  let raf = 0;
  let last = 0;
  const tick = (now: number) => {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    frame(dt);
    raf = requestAnimationFrame(tick);
  };
  const io = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting && !raf) {
        last = 0;
        onShow?.();
        raf = requestAnimationFrame(tick);
      } else if (!entry.isIntersecting && raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
    },
    { rootMargin: "120px" },
  );
  io.observe(el);
}

const tip = () => document.getElementById("tooltip")!;

export function showTip(html: string, x: number, y: number) {
  const el = tip();
  el.innerHTML = html;
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
  el.style.opacity = "1";
}

export function hideTip() {
  tip().style.opacity = "0";
}

export const fmt = (v: number, d = 1) => v.toFixed(d);
