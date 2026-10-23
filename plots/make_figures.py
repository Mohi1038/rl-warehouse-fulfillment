"""Regenerate every figure and the results tables from ``results/*.json`` and ``runs/*/s*/eval.csv``.

python plots/make_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Colour follows the agent, never its rank (reference categorical palette, fixed order).
COLORS = {
    "ppo_attn": "#2a78d6",
    "ppo_mlp": "#eb6834",
    "dqn": "#1baf7a",
    "edf": "#eda100",
    "greedy": "#e87ba4",
    "fifo": "#008300",
    "edf_charge": "#6250d6",
    "random": "#898781",
    "abl_no_dr": "#eb6834",
}
LABELS = {
    "ppo_attn": "PPO-Attention",
    "ppo_mlp": "PPO-MLP",
    "dqn": "DQN",
    "edf": "EDF",
    "greedy": "Greedy",
    "fifo": "FIFO",
    "edf_charge": "EDF+charge",
    "random": "Random",
}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("rb", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "font.family": "sans-serif", "lines.linewidth": 2,
})  # fmt: skip


def label(name: str) -> str:
    return LABELS.get(name, name)


def ci95(values) -> float:
    values = np.asarray(values, dtype=float)
    return 0.0 if len(values) < 2 else 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}")


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def seed_means(entries: list[dict], key: str = "task_return") -> list[float]:
    return [e["summary"][key] for e in entries]


# ---------------------------------------------------------------- E1 / E5 tables + bars
def main_benchmark(scen: dict, lines: list[str]) -> None:
    keys = [("task_return", "Task return"), ("revenue_captured_pct", "Revenue %"), ("on_time_pct", "On-time %"),
            ("express_on_time_pct", "Express on-time %"), ("trucks_missed", "Trucks missed"),
            ("mean_slack_at_delivery", "Slack at delivery"), ("battery_failures", "Battery failures"),
            ("travel_ticks_per_order", "Travel / order"), ("items_per_trip", "Items / trip")]  # fmt: skip
    for scenario, block in scen.items():
        lines.append(f"\n### {scenario}\n")
        lines.append("| Agent | runs | " + " | ".join(k[1] for k in keys) + " |")
        lines.append("|---|---|" + "---|" * len(keys))
        for name, entries in block["agents"].items():
            if not entries:
                continue
            cells = [f"{np.mean(seed_means(entries, k)):.1f} ± {ci95(seed_means(entries, k)):.1f}" if len(entries) > 1
                     else f"{np.mean(seed_means(entries, k)):.1f}" for k, _ in keys]  # fmt: skip
            lines.append(f"| {label(name)} | {len(entries)} | " + " | ".join(cells) + " |")

    order = sorted(
        (n for n, e in scen["nominal"]["agents"].items() if e), key=lambda n: np.mean(seed_means(scen["nominal"]["agents"][n]))
    )
    fig, axes = plt.subplots(1, len(scen), figsize=(3.0 * len(scen), 3.6), sharey=True)
    for ax, (scenario, block) in zip(np.atleast_1d(axes), scen.items()):
        means = [np.mean(seed_means(block["agents"][n])) for n in order]
        errs = [ci95(seed_means(block["agents"][n])) for n in order]
        ax.barh(
            range(len(order)),
            means,
            xerr=errs,
            color=[COLORS[n] for n in order],
            height=0.62,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2},
        )
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(scenario, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("task return")
    np.atleast_1d(axes)[0].set_yticks(range(len(order)), [label(n) for n in order])
    fig.suptitle(
        "E1/E5 — task return on 100 held-out seeds (mean ± 95% CI over training seeds)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "e1_e5_benchmark.png")


# ---------------------------------------------------------------- E2 learning curves
def learning_curves(runs: Path, baselines: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for algo in ("ppo_attn", "ppo_mlp", "dqn"):
        curves = []
        # x = update × 4096: PPO updates on 16 × 256 decisions and DQN logs every 4096 decisions
        for path in sorted((runs / algo).glob("s*/eval.csv")):
            rows = list(csv.DictReader(path.open()))
            curves.append((np.array([float(r["update"]) * 4096 for r in rows]), np.array([float(r["task_return"]) for r in rows]),
                           np.array([float(r["on_time_pct"]) for r in rows])))  # fmt: skip
        if not curves:
            continue
        grid = np.linspace(max(c[0][0] for c in curves), min(c[0][-1] for c in curves), 60)
        for ax, idx in zip(axes, (1, 2)):
            ys = np.stack([np.interp(grid, c[0], c[idx]) for c in curves])
            mean, err = ys.mean(0), (1.96 * ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else np.zeros_like(grid))
            ax.plot(grid / 1e6, mean, color=COLORS[algo], label=f"{label(algo)} (n={len(ys)})")
            ax.fill_between(grid / 1e6, mean - err, mean + err, color=COLORS[algo], alpha=0.18, linewidth=0)
    for ax, key in zip(axes, ("task_return", "on_time_pct")):
        for name in ("edf", "greedy"):
            if name in baselines:
                ax.axhline(baselines[name][key], color=COLORS[name], linestyle="--", linewidth=1.2, label=label(name))
        ax.set_xlabel("decisions (millions)")
    axes[0].set_ylabel("validation task return")
    axes[1].set_ylabel("validation on-time %")
    axes[0].set_title("E2 — learning curves (mean ± 95% CI)", loc="left")
    fig.tight_layout()
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    save(fig, "e2_learning_curves.png")


# ---------------------------------------------------------------- E3 rliable
def rliable_stats(scen: dict, lines: list[str]) -> None:
    from rliable import library as rly
    from rliable import metrics

    tasks = list(scen)
    agents = [n for n in scen["nominal"]["agents"] if scen["nominal"]["agents"][n] and n != "random"]

    def norm(task, values):
        lo = np.mean(seed_means(scen[task]["agents"]["random"]))
        hi = np.mean(seed_means(scen[task]["agents"]["edf"]))
        return (np.asarray(values) - lo) / (hi - lo)

    scores = {}
    for name in agents:
        runs = min(len(scen[t]["agents"][name]) for t in tasks)
        scores[name] = np.stack([norm(t, seed_means(scen[t]["agents"][name])[:runs]) for t in tasks], axis=1)

    def agg(x):
        return np.array([metrics.aggregate_iqm(x), metrics.aggregate_mean(x), metrics.aggregate_optimality_gap(x, 1.0)])

    point, intervals = rly.get_interval_estimates(scores, agg, reps=2000)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})
    order = sorted(agents, key=lambda n: point[n][0])
    for ax, (index, title) in zip(axes[:2], ((0, "IQM"), (1, "Mean"))):
        for row, name in enumerate(order):
            lo, hi = intervals[name][0][index], intervals[name][1][index]
            ax.barh(row, hi - lo, left=lo, height=0.55, color=COLORS[name], alpha=0.85)
            ax.plot([point[name][index]] * 2, [row - 0.3, row + 0.3], color=INK, linewidth=1.5)
        ax.axvline(1.0, color=COLORS["edf"], linestyle="--", linewidth=1)
        ax.set_yticks(range(len(order)), [label(n) for n in order] if index == 0 else [])
        ax.set_title(f"{title} EDF-normalised score", loc="left")
        ax.grid(axis="y", visible=False)
    taus = np.linspace(-1.0, 4.0, 101)
    profiles, profile_cis = rly.create_performance_profile(scores, taus, reps=500)
    for name in order:
        axes[2].plot(taus, profiles[name], color=COLORS[name], label=label(name))
        axes[2].fill_between(taus, profile_cis[name][0], profile_cis[name][1], color=COLORS[name], alpha=0.15, linewidth=0)
    axes[2].set_xlabel("EDF-normalised score τ")
    axes[2].set_ylabel("fraction of runs × tasks > τ")
    axes[2].set_title("Performance profile", loc="left")
    axes[2].legend(fontsize=7)
    fig.suptitle(
        "E3 — rliable: stratified bootstrap 95% CIs over 5 scenarios (random = 0, EDF = 1)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "e3_rliable.png")

    lines.append("\n### rliable aggregates (EDF-normalised; random = 0, EDF = 1)\n")
    lines.append("| Agent | IQM [95% CI] | Mean [95% CI] |")
    lines.append("|---|---|---|")
    for name in reversed(order):
        p, lo, hi = point[name], intervals[name][0], intervals[name][1]
        lines.append(f"| {label(name)} | {p[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}] | {p[1]:.2f} [{lo[1]:.2f}, {hi[1]:.2f}] |")
    lines.append("\n| Comparison | P(X > Y) [95% CI] |")
    lines.append("|---|---|")
    pairs = {f"{a},{b}": (scores[a], scores[b]) for a in ("ppo_attn",) for b in agents if b != a}
    prob, prob_ci = rly.get_interval_estimates(pairs, metrics.probability_of_improvement, reps=2000)
    for key in pairs:
        a, b = key.split(",")
        lines.append(
            f"| P({label(a)} > {label(b)}) | {np.ravel(prob[key])[0]:.2f} [{np.ravel(prob_ci[key])[0]:.2f}, {np.ravel(prob_ci[key])[1]:.2f}] |"
        )


# ---------------------------------------------------------------- E4 generalisation heatmap
def generalisation(gen: dict) -> None:
    lambdas, caps = gen["lambda_mult"], gen["capacity"]
    panels = [n for n in ("ppo_attn", "abl_no_dr") if np.any(gen["task_return"].get(n, 0))]
    edf = np.array(gen["task_return"]["edf"])
    diffs = {n: np.array(gen["task_return"][n]) - edf for n in panels}
    limit = max(np.abs(d).max() for d in diffs.values())
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.0), squeeze=False, constrained_layout=True)
    titles = {"ppo_attn": "with domain randomisation", "abl_no_dr": "without domain randomisation"}
    for ax, name in zip(axes[0], panels):
        image = ax.imshow(diffs[name], cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit), aspect="auto", origin="lower")
        for i in range(len(lambdas)):
            for j in range(len(caps)):
                ax.text(j, i, f"{diffs[name][i, j]:+.0f}", ha="center", va="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(caps)), caps)
        ax.set_yticks(range(len(lambdas)), [f"{x:g}×" for x in lambdas])
        ax.set_xlabel("robot capacity (trained on 4–8)")
        if name == panels[0]:
            ax.set_ylabel("arrival-rate multiplier λ")
        ax.set_title(titles[name], loc="left", fontsize=10)
        ax.grid(False)
        # randomisation range during training: capacity 4–8 (columns 1–5), λ 0.6–1.4× (rows 0.75–1.25)
        ax.add_patch(plt.Rectangle((0.5, 0.5), 5, 3, fill=False, edgecolor=INK, linewidth=1.2, linestyle=":"))
    fig.colorbar(image, ax=axes[0].tolist(), label="task return − EDF", shrink=0.85)
    fig.suptitle(
        "E4 — generalisation: task return − EDF (blue = better; dotted box = randomisation range)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    save(fig, "e4_generalisation.png")


# ---------------------------------------------------------------- E6 scaling
def scaling(data: dict) -> None:
    fig, axes = plt.subplots(1, len(data), figsize=(4.6 * len(data), 3.4), sharey=False)
    for ax, (key, block) in zip(np.atleast_1d(axes), data.items()):
        slots = sorted(block, key=int)
        x = [int(s) for s in slots]
        ppo = [np.mean(block[s]["ppo_attn"]) for s in slots]
        err = [ci95(block[s]["ppo_attn"]) for s in slots]
        ax.errorbar(
            x, ppo, yerr=err, color=COLORS["ppo_attn"], marker="o", markersize=6, capsize=3, label="PPO-Attention (trained N=16)"
        )
        for name in ("edf", "greedy"):
            ax.plot(
                x,
                [block[s][name] for s in slots],
                color=COLORS[name],
                marker="s",
                markersize=5,
                linestyle="--",
                linewidth=1.5,
                label=label(name),
            )
        if "ppo_mlp" in block[slots[0]] and block[slots[0]]["ppo_mlp"]:
            ax.scatter(
                [x[0]],
                [np.mean(block[slots[0]]["ppo_mlp"])],
                color=COLORS["ppo_mlp"],
                marker="D",
                s=40,
                zorder=3,
                label="PPO-MLP (N=16 only)",
            )
        ax.set_xticks(x)
        ax.set_xlabel("order slots N_max at test time")
        ax.set_ylabel("task return")
        ax.set_title(f"arrival rate {key.split('_')[1]}×", loc="left")
    handles, labels_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("E6 — zero-shot scaling of the permutation-invariant policy", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "e6_scaling.png")


# ---------------------------------------------------------------- E7 ablations
def ablations(data: dict, lines: list[str]) -> None:
    names = [k for k in data if not k.endswith("__env")]
    order = ["full"] + sorted((n for n in names if n != "full"), key=lambda n: -np.mean(data[n]))
    descriptions = {"full": "full agent", "no_attention": "− attention (MLP)", "no_mask": "− action mask", "no_pbrs": "− PBRS shaping",
                    "no_curriculum": "− curriculum", "no_smdp": "− SMDP discount", "no_battery": "− battery (easier env)",
                    "no_dr": "− domain randomisation", "long_horizon": "SMDP, γ = 0.9995 per tick"}  # fmt: skip
    means = [np.mean(data[n]) for n in order]
    errs = [ci95(data[n]) for n in order]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [COLORS["ppo_attn"] if n == "full" else ("#86b6ef" if n != "no_battery" else MUTED) for n in order]
    ax.barh(
        range(len(order)), means, xerr=errs, color=colors, height=0.6, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2}
    )
    for row, m in enumerate(means):
        ax.text(m + (errs[row] if errs[row] else 0) + 3, row, f"{m:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(order)), [descriptions.get(n, n) for n in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("task return on 100 held-out seeds (mean ± 95% CI over 3 seeds)")
    ax.set_title("E7 — ablations of PPO-Attention", loc="left")
    fig.tight_layout()
    save(fig, "e7_ablations.png")
    lines.append("\n### Ablations (task return, nominal scenario)\n")
    lines.append("| Variant | seeds | task return |")
    lines.append("|---|---|---|")
    for n in order:
        lines.append(f"| {descriptions.get(n, n)} | {len(data[n])} | {np.mean(data[n]):.1f} ± {ci95(data[n]):.1f} |")


# ---------------------------------------------------------------- E8 behaviour
def behaviour(data: dict) -> None:
    pairs = np.array(data["attention_pairs"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    if len(pairs):
        rel = pairs[:, 0] * pairs[:, 5]  # attention share among valid orders ÷ uniform share 1/n
        slack_ticks = pairs[:, 1] * 720
        bins = np.linspace(-150, 300, 19)
        centers, mean_rel = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (slack_ticks >= lo) & (slack_ticks < hi)
            if sel.sum() > 30:
                centers.append((lo + hi) / 2)
                mean_rel.append(rel[sel].mean())
        ax = axes[0]
        ax.plot(centers, mean_rel, color=COLORS["ppo_attn"], marker="o", markersize=5)
        ax.axhline(1.0, color=AXIS, linewidth=1)
        ax.text(centers[0], 1.0, " uniform", color=MUTED, fontsize=7, ha="left", va="bottom")
        ax.set_xlabel("order slack (ticks; < 0 = likely to miss its truck)")
        ax.set_ylabel("[CLS] attention ÷ uniform share")
        ax.set_title("Attention vs slack", loc="left")
    kinds = data["action_kinds"]
    tod = np.array(data["actions_by_time"]["ppo_attn"])
    share = tod / tod.sum(1, keepdims=True).clip(min=1)
    x = (np.arange(12) + 0.5) / 12 * 720
    kind_colors = {"pick": "#2a78d6", "deliver": "#eb6834", "charge": "#1baf7a", "wait": "#c3c2b7"}
    axes[1].stackplot(x, share.T * 100, colors=[kind_colors[k] for k in kinds], labels=kinds, edgecolor=SURFACE, linewidth=1)
    axes[1].set_xlim(0, 720)
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("time of shift (ticks)")
    axes[1].set_ylabel("share of PPO-Attention decisions %")
    axes[1].set_title("Actions over the shift", loc="left")
    axes[1].legend(loc="lower left", fontsize=7, ncol=4)
    width = 0.27
    for offset, name in zip((-width, 0, width), ("ppo_attn", "edf", "greedy")):
        counts = np.array(data["items_per_trip"][name][1:], dtype=float)
        axes[2].bar(
            np.arange(1, len(counts) + 1) + offset,
            100 * counts / counts.sum().clip(min=1),
            width=width * 0.92,
            color=COLORS[name],
            label=label(name),
        )
    axes[2].set_xlabel("items loaded per delivery trip (batching)")
    axes[2].set_ylabel("% of trips")
    axes[2].set_title("Batching", loc="left")
    axes[2].legend(fontsize=7)
    axes[2].grid(axis="x", visible=False)
    fig.suptitle("E8 — behaviour of PPO-Attention on the nominal scenario", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "e8_behaviour.png")


# ---------------------------------------------------------------- multi-robot and tiny env
def multi(data: dict, lines: list[str]) -> None:
    ks = sorted(data)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    x = np.arange(len(ks))
    width = 0.27
    for offset, name in zip((-width, 0, width), ("ppo_attn", "edf", "greedy")):
        if name == "ppo_attn":
            vals = [np.mean(seed_means(data[k][name])) for k in ks]
            errs = [ci95(seed_means(data[k][name])) for k in ks]
        else:
            vals = [data[k][name]["summary"]["task_return"] for k in ks]
            errs = [0] * len(ks)
        axes[0].bar(
            x + offset,
            vals,
            width * 0.92,
            yerr=errs,
            color=COLORS[name],
            label=label(name),
            error_kw={"ecolor": INK2, "capsize": 2},
        )
    axes[0].set_xticks(x, [f"{k[1:]} robot{'s' if k != 'k1' else ''}" for k in ks])
    axes[0].set_ylabel("task return")
    axes[0].set_title("Fleet size", loc="left")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="x", visible=False)
    biggest = ks[-1]
    columns = np.array(data[biggest]["pick_columns"])
    share = columns / columns.sum(1, keepdims=True).clip(min=1)
    image = axes[1].imshow(share, cmap=SEQUENTIAL, aspect="auto", vmin=0)
    axes[1].set_yticks(range(len(share)), [f"robot {i}" for i in range(len(share))])
    axes[1].set_xlabel("warehouse column of pick location (west → east)")
    axes[1].set_title(
        f"Emergent zoning with {biggest[1:]} robots: share of each robot's picks per column", loc="left", fontsize=10
    )
    axes[1].grid(False)
    fig.colorbar(image, ax=axes[1], label="share of picks")
    fig.suptitle(
        "M11 — multi-robot: shared PPO-Attention policy with robot tokens", x=0.01, ha="left", fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, "m11_multi_robot.png")
    lines.append("\n### Multi-robot (task return, nominal arrival rate)\n")
    lines.append("| Robots | PPO-Attention | EDF | Greedy | PPO on-time % |")
    lines.append("|---|---|---|---|---|")
    for k in ks:
        ppo = seed_means(data[k]["ppo_attn"])
        on_time = seed_means(data[k]["ppo_attn"], "on_time_pct")
        lines.append(f"| {k[1:]} | {np.mean(ppo):.1f} ± {ci95(ppo):.1f} | {data[k]['edf']['summary']['task_return']:.1f} | "
                     f"{data[k]['greedy']['summary']['task_return']:.1f} | {np.mean(on_time):.1f} |")  # fmt: skip


def tiny(data: dict, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for curve in data["q_curves"]:
        curve = np.array(curve)
        ax.plot(curve[:, 0] / 1000, curve[:, 1], color=COLORS["dqn"], alpha=0.9, linewidth=1.5)
    ax.plot([], [], color=COLORS["dqn"], label="tabular Q-learning (per seed)")
    ax.axhline(data["hindsight_optimal"], color=INK, linewidth=1.2, label="hindsight optimum")
    if data["ppo_attn"]:
        ax.axhline(np.mean(data["ppo_attn"]), color=COLORS["ppo_attn"], linestyle="--", linewidth=1.5, label="PPO-Attention")
    for name in ("edf", "random"):
        ax.axhline(data["baselines"][name], color=COLORS[name], linestyle=":", linewidth=1.2, label=label(name))
    ax.set_xlabel("training episodes (thousands)")
    ax.set_ylabel("greedy task return (200 seeds)")
    ax.set_title("M5 — tiny-env sanity check", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "m5_tiny_env.png")
    opt = data["hindsight_optimal"]
    lines.append("\n### Tiny environment (task return, 200 seeds)\n")
    lines.append("| Policy | task return | % of hindsight optimum |")
    lines.append("|---|---|---|")
    rows = [("Hindsight optimum", opt), ("Tabular Q-learning", np.mean(data["q_learning"]))]
    if data["ppo_attn"]:
        rows.append(("PPO-Attention", np.mean(data["ppo_attn"])))
    rows += [(label(k), v) for k, v in data["baselines"].items()]
    for name, value in rows:
        lines.append(f"| {name} | {value:.2f} | {100 * value / opt:.0f}% |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    args = parser.parse_args()
    runs = ROOT / args.runs
    lines = ["# Results\n", "Generated by `plots/make_figures.py` from `results/*.json`. ± is a 95% CI over training seeds.\n"]
    scen = load("scenarios")
    if scen:
        lines.append("\n## E1 / E5 — benchmark across scenarios\n")
        main_benchmark(scen, lines)
        baselines = {n: scen["nominal"]["agents"][n][0]["summary"] for n in ("edf", "greedy")}
        learning_curves(runs, baselines)
        lines.append("\n## E3 — statistical comparison\n")
        rliable_stats(scen, lines)
    if gen := load("generalisation"):
        generalisation(gen)
    if sc := load("scaling"):
        scaling(sc)
    if abl := load("ablations"):
        lines.append("\n## E7 — ablations\n")
        ablations(abl, lines)
    if beh := load("behaviour"):
        behaviour(beh)
    if mr := load("multi"):
        lines.append("\n## M11 — multi-robot\n")
        multi(mr, lines)
    if tn := load("tiny"):
        lines.append("\n## M5 — tiny environment\n")
        tiny(tn, lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tables.md").write_text("\n".join(lines) + "\n")
    print("  results/tables.md")


if __name__ == "__main__":
    main()
