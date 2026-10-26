import "./style.css";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { loadEpisode, loadSummary } from "./data";
import { initAgent } from "./sections/agent";
import { initArrivals } from "./sections/arrivals";
import { initDuel } from "./sections/duel";
import { initFleet } from "./sections/fleet";
import { initHero } from "./sections/hero";
import { initLearning } from "./sections/learning";
import { initLookahead } from "./sections/lookahead";
import { initResults } from "./sections/results";
import { initWorld } from "./sections/world";
import { reducedMotion } from "./util";

gsap.registerPlugin(ScrollTrigger);

function smoothScroll() {
  if (reducedMotion) return;
  const lenis = new Lenis({ lerp: 0.09, wheelMultiplier: 1 });
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add((time) => lenis.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);
  document.querySelector(".rail")!.addEventListener("click", (e) => {
    const a = (e.target as HTMLElement).closest("a");
    if (!a) return;
    e.preventDefault();
    lenis.scrollTo(a.getAttribute("href")!, { duration: 1.6 });
  });
}

function chrome() {
  const rail = document.querySelector(".rail")!;
  const sections = [...document.querySelectorAll<HTMLElement>("[data-label]")];
  rail.innerHTML = sections.map((s) => `<a href="#${s.id}" aria-label="${s.dataset.label}"><span>${s.dataset.label}</span></a>`).join("");
  const links = [...rail.querySelectorAll("a")];
  sections.forEach((s, i) =>
    ScrollTrigger.create({
      trigger: s,
      start: "top 50%",
      end: "bottom 50%",
      onToggle: (self) => self.isActive && links.forEach((l, j) => l.classList.toggle("active", j === i)),
    }),
  );
  gsap.to(".progress span", {
    scaleX: 1,
    ease: "none",
    scrollTrigger: { trigger: document.body, start: "top top", end: "bottom bottom", scrub: 0.3 },
  });
  if (reducedMotion) return;
  document.querySelectorAll<HTMLElement>("main [data-reveal]").forEach((el) =>
    gsap.from(el, {
      y: 50,
      opacity: 0,
      duration: 1.2,
      ease: "expo.out",
      scrollTrigger: { trigger: el, start: "top 88%" },
    }),
  );
  document.querySelectorAll<HTMLElement>(".limits > div, .callouts > div").forEach((el, i) =>
    gsap.from(el, {
      y: 40,
      opacity: 0,
      rotateX: -15,
      duration: 1,
      delay: (i % 4) * 0.08,
      ease: "expo.out",
      scrollTrigger: { trigger: el, start: "top 90%" },
    }),
  );
}

async function main() {
  const [summary, ppo, greedy, edf, fleet, lookahead] = await Promise.all([
    loadSummary(),
    loadEpisode("ppo"),
    loadEpisode("greedy"),
    loadEpisode("edf"),
    loadEpisode("fleet"),
    loadEpisode("lookahead"),
  ]);
  smoothScroll();
  initHero(ppo);
  initWorld(ppo);
  initArrivals(ppo);
  initAgent(ppo);
  initLearning(ppo, summary);
  initDuel(ppo, { greedy, edf });
  initLookahead(lookahead, summary);
  initFleet(fleet, summary);
  initResults(summary);
  chrome();
  ScrollTrigger.refresh();
}

main().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("beforeend", `<p style="padding:24px">Could not load the episode data: ${String(err)}</p>`);
});
