# DockRush — Technical Project Plan

**Deadline-Aware Warehouse Order Fulfillment with Attention-Based PPO**

> A robot (later a fleet) dynamically picks orders that arrive stochastically, deciding *which order to work on next* under truck-departure deadlines, carrying capacity, and battery limits. Learned with a permutation-invariant attention policy trained by PPO in an event-driven semi-MDP simulator.

**Hard constraint:** the decision-maker must be RL. Rule-based policies (FIFO/EDF/Greedy) exist **only as evaluation baselines**. BFS distances are environment physics (travel time), not the agent's decision method.

---

## Table of Contents
1. [Architecture](#1-architecture)
2. [Environment](#2-environment-warehouse_env)
3. [RL Formulation](#3-rl-formulation)
4. [Agents](#4-agents-agents)
5. [Training System](#5-training-system)
6. [Evaluation & Experiments](#6-evaluation--experiments)
7. [Bonus: Multi-Robot](#7-bonus-multi-robot)
8. [Engineering Quality](#8-engineering-quality)
9. [Repository Layout](#9-repository-layout)
10. [Milestones & Checklist](#10-milestones--checklist)
11. [Team Split & Git Workflow](#11-team-split--git-workflow)
12. [Risks & Mitigations](#12-risks--mitigations)

---

## 1. Architecture

```
┌────────────────────────── Event-driven simulator ──────────────────────────┐
│  OrderGenerator (NHPP) ──► OrderPool ──► Observation builder ──► Agent      │
│  TruckScheduler (jitter) ─► Docks          ▲                       │        │
│  Layout + dist matrix ────► Robot(s) ◄─────┴──── macro-action ◄────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
          │ (obs, action_mask, reward, τ = option duration)
          ▼
   Custom PPO (masking + SMDP discount + attention)   ← MAIN
   Double/Dueling DQN + PER (masked)                  ← RL baseline
   PPO-MLP                                            ← architecture ablation
   Tabular Q-learning (tiny env)                      ← sanity check
   Random / FIFO / EDF / Greedy                       ← evaluation-only baselines
```

**Key design choice:** the simulator is **event-driven**. The agent acts only when the robot becomes free; between decisions the clock advances by τ ticks while orders arrive and trucks depart. This makes the problem a **Semi-MDP** with temporally-extended actions (options).

---

## 2. Environment (`warehouse_env/`)

### 2.1 Layout — `layout.py`
- Grid ~21×15 loaded from text files in `warehouse_env/layouts/` (swappable):
  ```
  #####################
  #P..S.S..S.S..S.S..C#    P=packing, S=shelf, C=charger, D=dock, #=wall
  #...S.S..S.S..S.S...#
  #...S.S..S.S..S.S...#
  #D.....D.....D......#
  #####################
  ```
- ~60 shelf slots, each with an access cell.
- **All-pairs BFS distance matrix** computed once at init → `dist[a, b]` as `np.int16`.
- 3–4 layouts: 2 for training, 2 held out for OOD testing.

### 2.2 Time
- 1 tick = 1 cell of movement. Episode = one shift, **T = 720 ticks**.
- Pick = 2 ticks/item, dock load = 1 tick/item.

### 2.3 Orders — `orders.py`
- Arrivals: **non-homogeneous Poisson process**, sampled by thinning (Lewis–Shedler):
  `λ(t) = λ_base · (1 + 0.8·sin(2πt/T − π/2)) + spike(t)` → morning rush / lull / evening peak.

| Field | Distribution |
|---|---|
| items | 1–4 shelf IDs, Zipf-weighted toward "hot" shelves |
| tier | express 20% / standard 60% / bulk 20% |
| value | lognormal, μ by tier (express ≈ 3× bulk) |
| weight | sum of item weights (each 1–3) |
| truck | next departure of the matching dock with ≥ min slack |

- **VIP shocks:** with prob. p, a very high-value express order with a tight deadline.

### 2.4 Trucks — `trucks.py`
- 3 docks, departures every 120–180 ticks.
- **Jitter:** actual departure = scheduled + N(0, σ=8), clipped. Agent observes only the *scheduled* time → must learn a safety margin.
- On departure:
  - fully loaded assigned orders → **fulfilled**
  - express not loaded → **lost** (penalty)
  - standard/bulk not loaded → **reassigned** to next truck at 0.5× value

### 2.5 Robot
- Capacity **C = 6** weight units.
- Battery **B = 400**: −1/tick moving, −0.2/tick idle, +4/tick charging.
- Battery = 0 → stranded: large penalty + episode truncated.

### 2.6 Action space (macro-actions)
`Discrete(N_max + D + 2)` with N_max = 16, D = 3 → **21 actions**.

| Action | Semantics | Duration τ |
|---|---|---|
| `PICK(i)`, i ∈ [0,16) | travel to nearest unpicked item of order *i*, pick it | dist + 2 |
| `DELIVER(d)`, d ∈ [0,3) | travel to dock *d*, load all carried items for *d* | dist + n_items |
| `CHARGE` | go to charger, charge until full or a new order arrives | variable |
| `WAIT` | idle | 5 |

**Action mask** — invalid if:
- `PICK(i)`: empty slot, all items picked, or next item exceeds remaining capacity
- `DELIVER(d)`: nothing carried for dock *d*
- `CHARGE`: battery > 95%
- any move that would leave insufficient battery to reach the charger (hard safety mask)
- `WAIT` is always valid → mask never all-false.

### 2.7 Observation (`gymnasium.spaces.Dict`)
```python
obs = {
  "global": float32[16],
    # t/T, sin/cos(time phase), battery/B, load/C, robot x,y (norm),
    # carried-per-dock (3), time-to-scheduled-departure per dock (3),
    # backlog_count, λ_estimate
  "orders": float32[16, 14],
    # per slot: valid, value_norm, tier one-hot(3), dock one-hot(3),
    # slack=(departure − t − est_finish)/T, items_remaining, items_carried,
    # dist_to_nearest_item/maxdist, fits_capacity, age/T
  "action_mask": bool[21],
}
```
- More than 16 open orders → extras sit in a hidden **backlog** (only the count is observed); they enter slots as slots free.
- `est_finish` is a simple travel estimate used only as an input feature.

### 2.8 Reward — `reward.py`
Per option of duration τ:

$$
r = \sum_{o\in\text{fulfilled}} v_o\,\phi_{\text{tier}}
- \lambda_m \sum_{o\in\text{lost}} v_o
- \lambda_d \sum_{o\in\text{deferred}} 0.5\,v_o
- \lambda_c\,\tau
- \lambda_b\,\mathbb{1}[\text{stranded}]
+ \big(\gamma^{\tau}\Phi(s')-\Phi(s)\big)
$$

- **Potential-based shaping:** Φ(s) = α · Σ value of carried items whose truck is still reachable. Policy-invariant (Ng et al., 1999).
- Defaults: φ = {express 1.5, standard 1.0, bulk 0.7}, λ_m = 1.0, λ_d = 0.3, λ_c = 0.01, λ_b = 50, α = 0.3.
- Every term returned in `info["reward_terms"]` and logged separately.

### 2.9 Env API & performance
- `step(a) → obs, reward, terminated, truncated, info` with `info = {"tau", "reward_terms", "events"}`.
- Must pass `gymnasium.utils.env_checker.check_env`.
- Target **≥ 5k decisions/s** single-core (numpy, no Python objects in hot loops; profile with `cProfile`).
- Fully deterministic under a seed.

### 2.10 Renderer — `render.py` (pygame)
- Grid, shelves, robot coloured by battery, docks with truck countdown bars.
- Side panel: order list with slack colour bar (green → red).
- **Live attention overlay** highlighting orders the agent attends to.
- `render_mode="human"` and `"rgb_array"` → GIF/MP4 via `imageio`.

---

## 3. RL Formulation

| Component | Definition |
|---|---|
| **S** | Dict obs: global (16) + order set (16×14) + mask (21) ≈ 240 dims |
| **A** | Discrete(21) macro-actions with state-dependent masking |
| **R** | Revenue − missed/deferred penalties − op. cost − battery penalty + PBRS |
| **P** | Event-driven stochastic dynamics (NHPP arrivals, truck jitter, shocks) |
| **γ** | 0.995 per tick, applied as γ^τ per option (SMDP) |

---

## 4. Agents (`agents/`)

### 4.1 MAIN — PPO + Attention (`ppo_attn.py`, `networks.py`)
Written from scratch (CleanRL-style, ~500 lines) for full control over masking, SMDP discounting, and Dict observations.

**Network**
```
global(16) ──MLP──► g ∈ R^64 ──► [CLS] token
orders(16×14) ──MLP──► tokens ∈ R^{16×64}
[CLS] + tokens ──► TransformerEncoder(2 layers, 4 heads, d=64, key_padding_mask = ~valid)
   ├─ Pointer head:  logit_i = MLP([h_i, h_cls])          → 16 PICK logits
   ├─ Other head:    MLP(h_cls)                            → 5 logits (DELIVER×3, CHARGE, WAIT)
   └─ Value head:    MLP([h_cls, mean(h_valid)])           → V(s)
concat → 21 logits → masked_fill(~mask, −1e9) → Categorical
```
- Permutation-invariant over orders → can test with N_max = 24/32 without retraining.
- ~60k params; trains on CPU / laptop GPU.

**SMDP-correct GAE** (store τ_t per transition):

$$
\delta_t = r_t + \gamma^{\tau_t} V(s_{t+1})(1-d_t) - V(s_t),\qquad
\hat A_t = \delta_t + (\gamma\lambda)^{\tau_t}\hat A_{t+1}
$$

**Hyperparameters (starting point)**

| Param | Value |
|---|---|
| parallel envs | 16 (`AsyncVectorEnv`) |
| rollout | 256 decisions/env → 4096/update |
| epochs × minibatch | 4 × 1024 |
| lr | 3e-4, linear decay |
| clip ε | 0.2 |
| entropy coef | 0.02 → 0.001 annealed |
| value coef / grad clip | 0.5 / 0.5 |
| γ / GAE λ | 0.995 (per tick) / 0.95 |
| budget | 10–20M ticks (≈ 2–5M decisions) |

**Implementation details:** advantage normalisation, value clipping, orthogonal init, running reward-std normalisation, KL early stop (> 0.02).

### 4.2 RL baseline — DQN (`dqn.py`)
- Double DQN + **dueling** head on the **same attention encoder** (fair comparison).
- **Prioritized replay** (sum-tree), **n-step = 3** with SMDP discount γ^{Στ}.
- Masking in both action selection and target max.
- ε-greedy 1.0 → 0.05 over first 30% of training (sampling only valid actions).

### 4.3 Architecture ablation — PPO-MLP (`ppo_mlp.py`)
- Same PPO, flatten + MLP encoder; slots sorted by slack (MLP is not permutation invariant).

### 4.4 Sanity check — Tabular Q-learning (`q_learning.py`)
- Tiny env: 7×5 grid, ≤ 3 orders, capacity 2, 1 dock, no battery.
- State = (cell, carried bitmask, order statuses, time bucket) → dict Q-table.
- Must converge and match PPO on the tiny env → validates env + reward.

### 4.5 Evaluation-only baselines (`baselines.py`)
Random-valid · FIFO · EDF (earliest deadline) · Greedy (value / (dist+1)) · EDF + charge-at-25% rule.

---

## 5. Training System

### 5.1 Curriculum — `curriculum.py`
Auto-promote when on-time rate > 80% over last 50 episodes.

| Level | λ mult | Deadline slack | Battery | Shocks |
|---|---|---|---|---|
| 0 | 0.4 | ×2.0 | off | off |
| 1 | 0.7 | ×1.5 | on | off |
| 2 | 1.0 | ×1.0 | on | on |
| 3 | 1.3 | ×0.8 | on | on + blocked aisles |

### 5.2 Domain randomisation
Per episode: λ_base ∈ [0.6, 1.4]×, capacity ∈ {4…8}, jitter σ ∈ [0, 12], random training layout. Capacity and λ estimate are in the observation.

### 5.3 Infrastructure
- YAML configs (`configs/*.yaml`); a run = config + seed.
- Seed numpy / torch / env; deterministic torch where possible.
- TensorBoard (or W&B): losses, entropy, KL, explained variance, every reward term, on-time %, trucks missed, battery failures, curriculum level.
- Periodic eval (every 50 updates, 20 fixed episodes) → save `models/best_<algo>.pt`.

**Commands**
```bash
python train.py --config configs/ppo_attn.yaml --seed 0
python evaluate.py --model models/best_ppo_attn.pt --episodes 100 --render
bash experiments/run_all.sh          # all algos × 5 seeds + all figures
```

---

## 6. Evaluation & Experiments

| # | Experiment | Output |
|---|---|---|
| E1 | Main benchmark: all algos × 5 seeds on **100 held-out eval seeds** | results table |
| E2 | Learning curves (mean ± 95% CI) | plot |
| E3 | `rliable`: IQM + stratified bootstrap CIs, performance profiles, P(PPO-attn > EDF) | plots |
| E4 | Generalisation grid: λ ∈ {0.5…2.0} × capacity {3…10}, reward relative to EDF | heatmap |
| E5 | OOD layouts (2 unseen) | table |
| E6 | Scaling: train N_max=16, test 24/32 (attention only) | plot |
| E7 | Ablations: −mask, −PBRS, −curriculum, −SMDP discount, −attention, −battery | bar chart |
| E8 | Behaviour analysis: attention vs slack/value, action distribution over time-of-day, items-per-trip (batching) | plots |

**Metrics:** total reward, revenue captured %, on-time %, express on-time %, trucks missed, mean slack at delivery, battery failures, travel ticks per order.

**Headline goal:** PPO-Attention beats EDF and Greedy with statistically significant margin.

---

## 7. Bonus: Multi-Robot (`multi_env.py`)
- K = 2–3 robots, **shared policy** + robot-ID embedding.
- **Asynchronous decisions:** whichever robot frees up next queries the policy (natural in the event queue).
- Other robots added as tokens (position, target, load).
- Claimed orders masked for other robots (prevents duplicate work).
- Optional MAPPO-style centralized critic.
- Experiment: 1 vs 2 vs 3 robots; analyse emergent zone-splitting.

---

## 8. Engineering Quality
- **pytest invariants** (`tests/`):
  - capacity never exceeded
  - battery never negative
  - mask never all-false
  - Σ reward_terms == reward
  - no order fulfilled twice
  - same seed → identical trajectory
  - BFS distance matrix symmetric, triangle inequality holds
- **GitHub Actions CI:** tests + 1-minute PPO smoke run on every push.
- Pinned `requirements.txt` (+ optional `Dockerfile`).
- Pretrained checkpoints committed → `evaluate.py` works out-of-the-box.
- Type hints, `ruff` + `black`.

**Stack:** Python 3.11 · gymnasium · PyTorch · numpy · pygame · imageio · matplotlib/seaborn · rliable · tensorboard · pytest · (sb3-contrib MaskablePPO as optional cross-check).

---

## 9. Repository Layout
```
rl-warehouse-fulfillment/
├── README.md
├── PLAN.md
├── requirements.txt
├── configs/                 # ppo_attn.yaml, ppo_mlp.yaml, dqn.yaml, tiny_q.yaml, ...
├── warehouse_env/
│   ├── __init__.py          # gymnasium.register(...)
│   ├── env.py               # WarehouseEnv (event-driven SMDP)
│   ├── layout.py            # grid parsing + BFS distance matrix
│   ├── layouts/             # *.txt layout files
│   ├── orders.py            # NHPP arrivals, order generator
│   ├── trucks.py            # schedules, jitter, departures
│   ├── reward.py            # reward terms + PBRS
│   ├── render.py            # pygame visualiser
│   ├── tiny_env.py          # small env for tabular Q
│   └── multi_env.py         # bonus multi-robot
├── agents/
│   ├── networks.py          # attention encoder, pointer head, MLP encoder
│   ├── ppo_attn.py
│   ├── ppo_mlp.py
│   ├── dqn.py               # double + dueling + PER + n-step
│   ├── q_learning.py
│   └── baselines.py         # FIFO/EDF/Greedy — evaluation only
├── train.py
├── evaluate.py
├── curriculum.py
├── experiments/
│   ├── run_all.sh
│   └── ablations.yaml
├── plots/make_figures.py
├── models/                  # trained checkpoints
├── assets/                  # demo GIFs
└── tests/
```

---

## 10. Milestones & Checklist

| # | Milestone | Done when |
|---|---|---|
| M1 | Layout + BFS + order/truck generators | unit tests pass; arrival histogram matches λ(t) |
| M2 | Event-driven env + mask + reward terms | `check_env` passes; random agent runs; ≥ 5k steps/s |
| M3 | Renderer + GIF export | random-agent GIF recorded |
| M4 | Baselines + eval harness | baseline table on 100 seeds |
| M5 | Tiny env + tabular Q | converges to near-optimal |
| M6 | PPO-MLP (masked, SMDP GAE) | beats Random, approaches Greedy |
| M7 | Attention encoder + pointer head | **beats EDF & Greedy significantly** |
| M8 | DQN (dueling + PER + masking) | learning curves logged |
| M9 | Curriculum + domain randomisation | improves OOD heatmap |
| M10 | Full sweep + ablations | `run_all.sh` regenerates every figure |
| M11 | Multi-robot (bonus) | 2 robots > 1 robot, coordination visible |

### Checklist
- [x] M1 — layout parser, BFS matrix, NHPP order generator, truck scheduler
- [x] M2 — `WarehouseEnv` with Dict obs, action mask, reward terms, `check_env`
- [x] M2 — pytest invariants + CI workflow
- [x] M3 — pygame renderer, `rgb_array` GIF export
- [x] M4 — FIFO / EDF / Greedy / Random baselines, `evaluate.py`
- [x] M5 — tiny env + tabular Q-learning
- [x] M6 — PPO core (masking, SMDP GAE, vector envs, logging, checkpoints)
- [x] M7 — attention encoder + pointer head, hyperparameter tuning
- [x] M8 — DQN with dueling, PER, n-step
- [x] M9 — curriculum + domain randomisation
- [x] M10 — 5-seed sweep, rliable stats, generalisation, scaling, ablations, behaviour plots
- [x] M11 — multi-robot extension
- [x] Final — commit pretrained checkpoints, demo GIF, and README reproduction steps

### Suggested timeline (~3 weeks)
| Days | Milestones |
|---|---|
| 1–4 | M1–M3 |
| 5–6 | M4–M5 |
| 7–10 | M6, M8 |
| 11–14 | M7, M9 |
| 15–17 | M10 |
| 17–19 | M11 (bonus) |
| 19–21 | polish, checkpoints, README |

---

## 11. Team Split & Git Workflow
- **Member A — Environment:** layout, orders, trucks, env, reward, tests, renderer.
- **Member B — Agents:** networks, PPO, DQN, tabular Q, training loop, curriculum.
- **Member C — Experiments:** baselines, eval harness, sweeps, rliable stats, plots, multi-robot.

**Workflow**
- Feature branches (`feat/env-core`, `feat/ppo-attn`, ...) → PR → review → merge to `main`.
- Small, frequent, descriptive commits from **every** member (commit history is graded).
- Tag releases at milestones (`v0.1-env`, `v0.2-ppo`, `v1.0-final`).

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| PPO doesn't beat EDF | tune reward weights & α; longer training; curriculum; check mask/feature bugs with tiny env |
| Env too slow | vectorise with numpy, precompute distances, AsyncVectorEnv, profile hot loops |
| Reward hacking (e.g. endless WAIT) | per-tick op. cost, log reward terms, watch GIFs regularly |
| Unstable training | KL early stop, grad clip, reward normalisation, lower lr |
| Scope creep | M1–M10 are core; M11 only after M10 is done |
