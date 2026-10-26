import { gsap } from "gsap";
import type { Episode } from "../data";
import { Replay } from "../sim";
import { reducedMotion, whileVisible } from "../util";
import { Warehouse3D } from "../warehouse3d";

export function initHero(ep: Episode) {
  const canvas = document.getElementById("hero-canvas") as HTMLCanvasElement;
  const replay = new Replay(ep);
  const scene = new Warehouse3D(canvas, ep);
  scene.setView("hero", true);
  let t = 150;
  const speed = 9; // ticks per second
  whileVisible(canvas, (dt) => {
    if (!reducedMotion) {
      t += dt * speed;
      if (t > replay.horizon) t = 100;
    }
    scene.frame(replay.state(reducedMotion ? 360 : t), reducedMotion ? 0 : dt);
  });

  if (reducedMotion) return;
  // the scene drifts and dims as the hero scrolls away
  gsap.to(canvas, {
    opacity: 0.25,
    scale: 1.08,
    ease: "none",
    scrollTrigger: { trigger: "#hero", start: "top top", end: "bottom top", scrub: true },
  });
  gsap.to(".hero-inner", {
    yPercent: -18,
    opacity: 0,
    ease: "none",
    scrollTrigger: { trigger: "#hero", start: "30% top", end: "bottom top", scrub: true },
  });
}
