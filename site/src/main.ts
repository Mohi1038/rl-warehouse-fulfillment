import "./style.css";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { loadEpisode, loadSummary } from "./data";
import { initCounters, initCursor, initMagnetic, initMarquee, initReveals, initStepFades, runLoader } from "./fx";
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
  const lenis = new Lenis({ lerp: 0.085 });
  lenis.on("scroll", ScrollTrigger.update);
  gsap.ticker.add((time) => lenis.raf(time * 1000));
  gsap.ticker.lagSmoothing(0);
  document.addEventListener("click", (e) => {
    const a = (e.target as HTMLElement).closest<HTMLAnchorElement>('a[href^="#"]');
    if (!a) return;
    e.preventDefault();
    lenis.scrollTo(a.getAttribute("href")!, { duration: 1.8 });
  });
}

/** Chapter rail and the bottom progress line. */
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
}

async function main() {
  window.scrollTo(0, 0);
  const loads = [loadSummary(), loadEpisode("ppo"), loadEpisode("greedy"), loadEpisode("edf"), loadEpisode("fleet"), loadEpisode("lookahead")] as const;
  const fonts = document.fonts.ready;
  const revealed = runLoader([...loads, fonts]);
  const [summary, ppo, greedy, edf, fleet, lookahead] = await Promise.all(loads);
  await fonts;
  smoothScroll();
  initReveals();
  initStepFades();
  initCounters();
  initCursor();
  initMagnetic();
  initMarquee();
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
  await revealed;
  ScrollTrigger.refresh();
}

main().catch((err) => {
  console.error(err);
  document.getElementById("loader")?.remove();
  document.body.classList.remove("is-loading");
  document.body.insertAdjacentHTML("beforeend", `<p style="padding:24px">Could not load the replay data: ${String(err)}</p>`);
});
