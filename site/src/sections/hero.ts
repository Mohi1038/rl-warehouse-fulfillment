import { gsap } from "gsap";
import type { Episode } from "../data";
import { WarehouseRenderer } from "../renderer";
import { Replay } from "../sim";
import { reducedMotion, whileVisible } from "../util";

export function initHero(ep: Episode) {
  const canvas = document.getElementById("hero-canvas") as HTMLCanvasElement;
  const replay = new Replay(ep);
  const renderer = new WarehouseRenderer(canvas, ep.layout);
  let t = 120;
  const speed = 26; // ticks per second
  if (reducedMotion) renderer.draw(replay.state(360));
  else
    whileVisible(canvas, (dt) => {
      t += dt * speed;
      if (t > replay.horizon) t = 0;
      renderer.draw(replay.state(t), { countdowns: false });
    });

  // title: characters rise and unfold one by one
  const title = document.querySelector<HTMLElement>("[data-split]")!;
  title.innerHTML = [...(title.textContent ?? "")].map((ch) => `<span class="ch">${ch}</span>`).join("");
  const tl = gsap.timeline({ defaults: { ease: "expo.out" } });
  if (!reducedMotion) {
    tl.from(".hero-stage", { opacity: 0, scale: 1.25, duration: 2.4, ease: "power3.out" }, 0)
      .from(".eyebrow", { y: 20, opacity: 0, duration: 1 }, 0.3)
      .from(".title .ch", { yPercent: 110, rotateX: -80, opacity: 0, duration: 1.4, stagger: 0.06 }, 0.4)
      .from(".lede", { y: 30, opacity: 0, duration: 1.2 }, 0.9)
      .from(".hero-stats .stat", { y: 40, opacity: 0, duration: 1.1, stagger: 0.12 }, 1.1)
      .from(".footnote, .scroll-cue", { opacity: 0, duration: 1 }, 1.6);
  }
  document.querySelectorAll<HTMLElement>("[data-count]").forEach((el, i) => {
    const target = Number(el.dataset.count);
    const suffix = el.dataset.suffix ?? "";
    const obj = { v: 0 };
    gsap.to(obj, {
      v: target,
      duration: reducedMotion ? 0 : 2.2,
      delay: reducedMotion ? 0 : 1.3 + i * 0.15,
      ease: "power3.out",
      onUpdate: () => {
        el.textContent = `${obj.v.toFixed(1)}${suffix}`;
      },
    });
  });

  // parallax: the warehouse tilts flatter and recedes as you scroll away
  if (!reducedMotion)
    gsap.to(".hero-stage", {
      rotateX: 18,
      scale: 0.92,
      yPercent: -6,
      ease: "none",
      scrollTrigger: { trigger: "#hero", start: "top top", end: "bottom top", scrub: true },
    });
}
