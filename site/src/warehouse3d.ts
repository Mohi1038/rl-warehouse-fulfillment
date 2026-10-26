// WebGL warehouse: the real layout in 3D, driven by the same replay state as the 2D renderer.

import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Episode } from "./data";
import { ROBOT, TIER } from "./palette";
import type { WorldState } from "./sim";

export type ViewName = "hero" | "overview" | "orders" | "trucks" | "robot" | "all" | "fleet";

const MAX_ORBS = 96;
const MAX_CUBES = 48;
const TRAIL = 28;

export class Warehouse3D {
  private renderer: THREE.WebGLRenderer;
  private composer: EffectComposer;
  private bloom: UnrealBloomPass;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
  private camPos = new THREE.Vector3(0, 26, 22);
  private camLook = new THREE.Vector3();
  private goalPos = new THREE.Vector3(0, 26, 22);
  private goalLook = new THREE.Vector3();
  private view: ViewName = "hero";
  private orbit = 0;
  private mouse = new THREE.Vector2();
  private rows: number;
  private cols: number;
  private robots: { group: THREE.Group; light: THREE.PointLight; trail: THREE.Line; core: THREE.Mesh }[] = [];
  private orbs: THREE.InstancedMesh;
  private halos: THREE.InstancedMesh;
  private pillars: THREE.InstancedMesh;
  private cubes: THREE.InstancedMesh;
  private trucks = new Map<number, THREE.Group>();
  private truckMats = new Map<number, THREE.MeshStandardMaterial>();
  private dummy = new THREE.Object3D();
  private color = new THREE.Color();
  private ro: ResizeObserver;
  private shelfSide = new Map<string, [number, number]>();
  private ep: Episode;
  private time = 0;

  constructor(private canvas: HTMLCanvasElement, ep: Episode) {
    this.ep = ep;
    const grid = ep.layout.grid;
    this.rows = grid.length;
    this.cols = grid[0].length;
    const small = window.innerWidth < 760;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: !small, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(small ? 1.25 : 2, window.devicePixelRatio));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = !small;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.background = new THREE.Color("#050505");
    this.scene.fog = new THREE.FogExp2("#050505", 0.028);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 1.05, 0.55, 0.18);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.buildStatic();
    for (let r = 0; r < ep.robots; r++) this.robots.push(this.buildRobot(r));

    const orbGeo = new THREE.SphereGeometry(1, 24, 16);
    this.orbs = new THREE.InstancedMesh(orbGeo, new THREE.MeshBasicMaterial({ toneMapped: false }), MAX_ORBS);
    this.halos = new THREE.InstancedMesh(
      orbGeo,
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }),
      MAX_ORBS,
    );
    this.pillars = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(0.03, 0.03, 1, 6, 1, true),
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }),
      MAX_ORBS,
    );
    this.cubes = new THREE.InstancedMesh(new THREE.BoxGeometry(0.16, 0.16, 0.16), new THREE.MeshBasicMaterial({ toneMapped: false }), MAX_CUBES);
    for (const m of [this.orbs, this.halos, this.pillars, this.cubes]) {
      m.count = 0;
      m.frustumCulled = false;
      this.scene.add(m);
    }

    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect();
      this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, ((e.clientY - r.top) / r.height) * 2 - 1);
    });
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.resize();
    this.setView("hero", true);
  }

  /** World position of a cell centre (row, col). */
  private at(row: number, col: number, y = 0): THREE.Vector3 {
    return new THREE.Vector3(col - this.cols / 2 + 0.5, y, row - this.rows / 2 + 0.5);
  }

  private resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.resolution.set(w / 2, h / 2);
    this.camera.aspect = w / h;
    // keep the whole floor in frame on tall screens
    this.camera.fov = w / h < 1 ? 48 : 32;
    this.camera.updateProjectionMatrix();
  }

  private buildStatic() {
    const { grid, docks, charger } = this.ep.layout;
    const scene = this.scene;
    scene.add(new THREE.AmbientLight("#8a8fa8", 0.35));
    const key = new THREE.DirectionalLight("#dfe3ff", 1.1);
    key.position.set(-10, 18, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const sc = key.shadow.camera;
    sc.left = -16;
    sc.right = 16;
    sc.top = 14;
    sc.bottom = -14;
    key.shadow.bias = -0.0005;
    scene.add(key);
    const rim = new THREE.DirectionalLight("#9085e9", 0.5);
    rim.position.set(12, 6, -10);
    scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(this.cols + 30, this.rows + 30),
      new THREE.MeshStandardMaterial({ color: "#0a0a0b", roughness: 0.82, metalness: 0.25 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // floor grid over walkable cells only
    const lines: number[] = [];
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < this.cols; c++) {
        if (grid[r][c] === "#") continue;
        const p = this.at(r, c, 0.005);
        lines.push(p.x - 0.5, p.y, p.z - 0.5, p.x + 0.5, p.y, p.z - 0.5, p.x - 0.5, p.y, p.z - 0.5, p.x - 0.5, p.y, p.z + 0.5);
      }
    const gridGeo = new THREE.BufferGeometry();
    gridGeo.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
    scene.add(new THREE.LineSegments(gridGeo, new THREE.LineBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.045 })));

    // perimeter
    const w = this.cols - 2;
    const h = this.rows - 2;
    const border = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-w / 2, 0.01, -h / 2),
        new THREE.Vector3(w / 2, 0.01, -h / 2),
        new THREE.Vector3(w / 2, 0.01, h / 2),
        new THREE.Vector3(-w / 2, 0.01, h / 2),
      ]),
      new THREE.LineBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.35 }),
    );
    scene.add(border);

    // shelves: dark blocks with hairline edges
    const boxes: THREE.BufferGeometry[] = [];
    const edges: THREE.BufferGeometry[] = [];
    const H = 1.15;
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < this.cols; c++) {
        if (grid[r][c] !== "S") continue;
        const p = this.at(r, c, H / 2);
        const box = new THREE.BoxGeometry(0.84, H, 0.84);
        box.translate(p.x, p.y, p.z);
        boxes.push(box);
        const e = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.84, H, 0.84));
        e.translate(p.x, p.y, p.z);
        edges.push(e);
      }
    const shelves = new THREE.Mesh(mergeGeometries(boxes), new THREE.MeshStandardMaterial({ color: "#101012", roughness: 0.55, metalness: 0.5 }));
    shelves.castShadow = true;
    shelves.receiveShadow = true;
    scene.add(shelves);
    scene.add(new THREE.LineSegments(mergeGeometries(edges), new THREE.LineBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.16 })));

    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < this.cols; c++) {
        if (!".PCD".includes(grid[r][c])) continue;
        for (const [dr, dc] of [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ])
          if (grid[r + dr]?.[c + dc] === "S") {
            this.shelfSide.set(`${r},${c}`, [dr, dc]);
            break;
          }
      }

    // docks: glowing bays, with trucks parked beyond the wall
    docks.forEach(([r, c], d) => {
      const p = this.at(r, c);
      const bay = new THREE.Mesh(
        new THREE.PlaneGeometry(0.92, 0.92),
        new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.08, toneMapped: false }),
      );
      bay.rotation.x = -Math.PI / 2;
      bay.position.set(p.x, 0.012, p.z);
      scene.add(bay);
      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.PlaneGeometry(0.92, 0.92)),
        new THREE.LineBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.7 }),
      );
      outline.rotation.x = -Math.PI / 2;
      outline.position.set(p.x, 0.014, p.z);
      scene.add(outline);
      void d;
    });

    for (const truck of this.ep.trucks) {
      const g = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color: "#16161a", roughness: 0.4, metalness: 0.6, emissive: "#000000" });
      const trailer = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.95, 0.95), mat);
      trailer.position.y = 0.55;
      trailer.castShadow = true;
      g.add(trailer);
      const cab = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.8, 0.85), mat);
      cab.position.set(1.4, 0.48, 0);
      cab.castShadow = true;
      g.add(cab);
      const e1 = new THREE.LineSegments(new THREE.EdgesGeometry(trailer.geometry), new THREE.LineBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.25 }));
      e1.position.copy(trailer.position);
      g.add(e1);
      const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.12, 0.6), new THREE.MeshBasicMaterial({ color: "#8a8a8a", toneMapped: false }));
      lamp.position.set(1.7, 0.5, 0);
      g.add(lamp);
      g.visible = false;
      scene.add(g);
      this.trucks.set(truck.id, g);
      this.truckMats.set(truck.id, mat);
    }

    if (charger) {
      const p = this.at(charger[0], charger[1]);
      const pad = new THREE.Mesh(new THREE.CircleGeometry(0.42, 6), new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.22, toneMapped: false }));
      pad.rotation.x = -Math.PI / 2;
      pad.position.set(p.x, 0.015, p.z);
      scene.add(pad);
      const glow = new THREE.PointLight("#ffffff", 2.5, 3.5);
      glow.position.set(p.x, 0.6, p.z);
      scene.add(glow);
    }
  }

  private buildRobot(r: number) {
    const color = new THREE.Color(ROBOT[r % ROBOT.length]);
    const group = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.62, 0.3, 0.62),
      new THREE.MeshStandardMaterial({ color: "#0d0d10", roughness: 0.3, metalness: 0.8 }),
    );
    body.position.y = 0.2;
    body.castShadow = true;
    group.add(body);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(body.geometry), new THREE.LineBasicMaterial({ color }));
    edges.position.copy(body.position);
    group.add(edges);
    const core = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.06, 0.34), new THREE.MeshBasicMaterial({ color: color.clone().multiplyScalar(3), toneMapped: false }));
    core.position.y = 0.38;
    group.add(core);
    const eye = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.3), new THREE.MeshBasicMaterial({ color: "#ffffff", toneMapped: false }));
    eye.position.set(0.32, 0.22, 0);
    group.add(eye);
    const light = new THREE.PointLight(color, 9, 6, 1.6);
    light.position.y = 0.9;
    group.add(light);
    this.scene.add(group);

    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
    const colors = new Float32Array(TRAIL * 3);
    for (let i = 0; i < TRAIL; i++) {
      const k = (i / (TRAIL - 1)) * 1.6;
      colors.set([color.r * k, color.g * k, color.b * k], i * 3);
    }
    trailGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, toneMapped: false }));
    trail.frustumCulled = false;
    this.scene.add(trail);
    return { group, light, trail, core };
  }

  setView(view: ViewName, instant = false) {
    this.view = view;
    if (!instant) return;
    this.updateGoal();
    this.camPos.copy(this.goalPos);
    this.camLook.copy(this.goalLook);
  }

  private updateGoal(world?: WorldState) {
    // The floor spans x ∈ [-9.5, 9.5], z ∈ [-6.5, 6.5], trucks park near z ≈ 8.
    // On wide screens the story text covers the left ~40%, so scroll views aim left of centre
    // to sit the warehouse in the right-hand part of the frame.
    const tall = this.camera.aspect < 1;
    const k = tall ? 1.6 : 1;
    const off = tall || this.view === "fleet" || this.view === "hero" ? 0 : -3.6;
    switch (this.view) {
      case "hero": {
        const R = 21 * k;
        this.goalPos.set(Math.sin(this.orbit) * R, 13 * k, Math.cos(this.orbit) * R);
        this.goalLook.set(0, 0, 0.5);
        break;
      }
      case "overview":
        this.goalPos.set(off, 31 * k, 11 * k);
        this.goalLook.set(off, 0, 0.8);
        break;
      case "orders":
        this.goalPos.set(off - 10, 15 * k, 15 * k);
        this.goalLook.set(off + 1, 0, -0.5);
        break;
      case "trucks":
        this.goalPos.set(off, 7.5 * k, 21 * k);
        this.goalLook.set(off, 0.4, 5.2);
        break;
      case "robot": {
        const r = world?.robots[0];
        const p = r ? this.at(r.y, r.x) : new THREE.Vector3();
        this.goalPos.set(p.x + 4 + off * 0.4, 6.5 * k, p.z + 6.5 * k);
        this.goalLook.set(p.x + off * 0.4, 0, p.z);
        break;
      }
      case "fleet":
        this.goalPos.set(0, 20 * k, 17.5 * k);
        this.goalLook.set(0, 0, 1.4);
        break;
      case "all":
        this.goalPos.set(off, 21 * k, 19 * k);
        this.goalLook.set(off, 0, 1.2);
        break;
    }
    // on tall screens the story text sits in the lower half: lift the scene into the upper half
    if (tall && this.view !== "hero" && this.view !== "fleet") this.goalLook.z += 5;
  }

  /** Advance one frame: apply ``world`` and render. */
  frame(world: WorldState, dt: number) {
    this.time += dt;
    if (this.view === "hero") this.orbit += dt * 0.06;
    this.updateGoal(world);
    const ease = 1 - Math.pow(0.0015, dt);
    this.camPos.lerp(this.goalPos, ease);
    this.camLook.lerp(this.goalLook, ease);
    this.camera.position.copy(this.camPos).add(new THREE.Vector3(this.mouse.x * 0.8, -this.mouse.y * 0.5, 0));
    this.camera.lookAt(this.camLook);

    // robots
    world.robots.forEach((rs, i) => {
      const r = this.robots[i];
      if (!r) return;
      const p = this.at(rs.y, rs.x);
      r.group.position.set(p.x, rs.moving ? Math.abs(Math.sin(this.time * 9)) * 0.04 : 0, p.z);
      r.group.rotation.y = -rs.heading;
      const flicker = rs.kind === "charge" && !rs.moving ? 0.6 + 0.4 * Math.sin(this.time * 10) : 1;
      r.light.intensity = 9 * flicker * (0.5 + 0.5 * rs.battery);
      const pos = r.trail.geometry.getAttribute("position") as THREE.BufferAttribute;
      const pts = [...rs.trail, { x: rs.x, y: rs.y }];
      for (let k = 0; k < TRAIL; k++) {
        const src = pts[Math.max(0, pts.length - TRAIL + k)] ?? pts[0];
        const q = this.at(src.y, src.x, 0.06);
        pos.setXYZ(k, q.x, q.y, q.z);
      }
      pos.needsUpdate = true;
    });

    // order orbs on the shelves, pillars on fresh arrivals
    let n = 0;
    let pillars = 0;
    for (const item of world.shelfItems) {
      if (n >= MAX_ORBS) break;
      const o = item.order;
      const [r, c] = item.cell;
      const side = this.shelfSide.get(`${r},${c}`) ?? [0, 0];
      const p = this.at(r + side[0] * 0.32, c + side[1] * 0.32 + (item.index - 1) * 0.16 * (side[1] === 0 ? 1 : 0), 0.0);
      const age = world.t - o.arrival;
      const appear = Math.min(1, age / 5);
      const size = (0.09 + 0.06 * Math.min(1, Math.log1p(o.value) / 4.2)) * (0.3 + 0.7 * appear);
      const y = 0.55 + Math.sin(this.time * 2 + o.id) * 0.05 + (1 - appear) * 2.5;
      this.dummy.position.set(p.x, y, p.z);
      this.dummy.scale.setScalar(size);
      this.dummy.rotation.set(0, 0, 0);
      this.dummy.updateMatrix();
      this.orbs.setMatrixAt(n, this.dummy.matrix);
      this.color.set(TIER[o.tier]).multiplyScalar(o.vip ? 4 : 2.2);
      this.orbs.setColorAt(n, this.color);
      this.dummy.scale.setScalar(size * 2.6);
      this.dummy.updateMatrix();
      this.halos.setMatrixAt(n, this.dummy.matrix);
      this.halos.setColorAt(n, this.color);
      if (age < 8 && pillars < MAX_ORBS) {
        const hgt = 7 * (1 - age / 8);
        this.dummy.position.set(p.x, y + hgt / 2, p.z);
        this.dummy.scale.set(1 + (1 - age / 8) * 2, hgt, 1 + (1 - age / 8) * 2);
        this.dummy.updateMatrix();
        this.pillars.setMatrixAt(pillars, this.dummy.matrix);
        this.pillars.setColorAt(pillars, this.color);
        pillars++;
      }
      n++;
    }
    this.orbs.count = n;
    this.halos.count = n;
    this.pillars.count = pillars;
    for (const m of [this.orbs, this.halos, this.pillars]) {
      m.instanceMatrix.needsUpdate = true;
      if (m.instanceColor) m.instanceColor.needsUpdate = true;
    }

    // carried items ride on robots; loaded items wait at the docks
    let cubes = 0;
    world.robots.forEach((rs) =>
      rs.carried.forEach((it, i) => {
        if (cubes >= MAX_CUBES) return;
        const p = this.at(rs.y, rs.x);
        this.dummy.position.set(p.x - 0.15 + (i % 3) * 0.17, 0.48 + Math.floor(i / 3) * 0.17, p.z);
        this.dummy.scale.setScalar(1);
        this.dummy.updateMatrix();
        this.cubes.setMatrixAt(cubes, this.dummy.matrix);
        this.cubes.setColorAt(cubes, this.color.set(TIER[it.tier]).multiplyScalar(2));
        cubes++;
      }),
    );
    const perDock = new Map<number, number>();
    for (const it of world.dockItems) {
      if (cubes >= MAX_CUBES) break;
      const k = perDock.get(it.dock) ?? 0;
      perDock.set(it.dock, k + 1);
      const [r, c] = this.ep.layout.docks[it.dock];
      const p = this.at(r, c);
      this.dummy.position.set(p.x - 0.3 + (k % 4) * 0.2, 0.09 + Math.floor(k / 4) * 0.18, p.z - 0.25);
      this.dummy.updateMatrix();
      this.cubes.setMatrixAt(cubes, this.dummy.matrix);
      this.cubes.setColorAt(cubes, this.color.set(TIER[it.tier]).multiplyScalar(2));
      cubes++;
    }
    this.cubes.count = cubes;
    this.cubes.instanceMatrix.needsUpdate = true;
    if (this.cubes.instanceColor) this.cubes.instanceColor.needsUpdate = true;

    // trucks: parked beyond the dock wall, red pulse when departure is close, then drive off
    this.trucks.forEach((g) => (g.visible = false));
    for (const ts of world.trucks) {
      const g = this.trucks.get(ts.truck.id)!;
      const [r, c] = this.ep.layout.docks[ts.truck.dock];
      const p = this.at(r + 2.1, c + 0.3);
      const m = ts.motion;
      const dx = m > 0 ? m * m * m * 9 : m < 0 ? m * 6 : 0;
      g.position.set(p.x + dx, 0, p.z);
      g.visible = true;
      const mat = this.truckMats.get(ts.truck.id)!;
      const urgent = !ts.truck.final && ts.countdown < 35 && ts.countdown > 0;
      mat.emissive.set(urgent ? "#d95926" : "#000000");
      mat.emissiveIntensity = urgent ? 0.12 + 0.12 * Math.sin(this.time * 8) : 0;
    }

    this.composer.render();
  }

  get currentView() {
    return this.view;
  }

  destroy() {
    this.ro.disconnect();
    this.renderer.dispose();
  }
}
