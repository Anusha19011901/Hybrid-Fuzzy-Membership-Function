"""Phase 3: Binary Brain Storm Optimization over the 13 clinical attributes.

Fitness = macro F1 of logistic regression under an inner 3-fold stratified CV
on the training split only. BBSO is run for ten seeds; the fitness cache is
shared across seeds because fitness is a deterministic function of the mask.

Writes:
    results/03_bbso_runs.csv              one row per seed
    results/03_selection_frequency.csv    fraction of seeds selecting each attribute
    results/03_subset_vs_all_cv.csv       5-fold CV, three models x {all 13, BBSO subset}
    results/03_subset_vs_all_test.csv     held-out test, same grid
    results/03_per_site_subset.csv        per-hospital diagnostic for the subset models
    results/03_runtime.json               wall-clock timings and cache statistics
    figures/03_selection_frequency.png
    figures/03_convergence.png
    figures/03_subset_vs_all.png

Run from the repository root:  python scripts/phase3_bbso.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfm import config, data, evaluation, plotting  # noqa: E402
from hfm.bbso import BBSO, BBSOConfig, MaskFitness, consensus_subset, selection_frequency  # noqa: E402
from hfm.models import make_baselines, make_pipeline  # noqa: E402

SEEDS = list(range(10))
MODELS = ["logistic_regression", "random_forest", "mlp"]
PRETTY = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "mlp": "MLP (sklearn)"}


def plot_frequency(freq: pd.Series, chosen: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    y = np.arange(len(freq))
    colors = [plotting.RISK_COLORS["Moderate"] if a in chosen else plotting.BLUE_RAMP[3] for a in freq.index]
    ax.barh(y, freq.values, height=0.62, color=colors)
    for i, v in enumerate(freq.values):
        ax.text(v + 0.015, i, f"{v:.0%}", va="center", fontsize=8.5, color=plotting.INK_2)
    ax.set_yticks(y, [f"{a}  ·  {config.ATTRIBUTE_LABELS[a]}" for a in freq.index], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(f"Share of {len(SEEDS)} BBSO seeds whose best mask includes the attribute")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Attribute selection frequency across seeds")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plotting.RISK_COLORS["Moderate"]),
               plt.Rectangle((0, 0), 1, 1, color=plotting.BLUE_RAMP[3])]
    ax.legend(handles, ["in the chosen subset", "not chosen"], loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(histories: dict[int, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for seed, h in histories.items():
        ax.plot(range(1, len(h) + 1), h, color=plotting.CATEGORICAL[0], alpha=0.45, lw=1.4)
        ax.scatter([len(h)], [h[-1]], s=14, color=plotting.CATEGORICAL[0], zorder=3)
    ax.set_xlabel("Iteration (each line is one seed; dot = where it stopped)")
    ax.set_ylabel("Best inner-CV macro F1 so far")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.set_title("BBSO convergence, 10 seeds")
    fig.savefig(path)
    plt.close(fig)


def plot_subset_vs_all(folds: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(MODELS))
    h = 0.34
    for k, (variant, label, color) in enumerate(
        [("all", "All 13 attributes", plotting.CATEGORICAL[0]), ("subset", "BBSO subset", plotting.CATEGORICAL[1])]
    ):
        off = (k - 0.5) * h
        for i, m in enumerate(MODELS):
            vals = folds[(folds.model == m) & (folds.variant == variant)].macro_f1.values
            ax.barh(i + off, vals.mean(), height=h - 0.04, color=color, label=label if i == 0 else None)
            ax.scatter(vals, np.full(len(vals), i + off), s=16, color=plotting.INK,
                       edgecolor=plotting.SURFACE, linewidth=0.8, zorder=3)
            ax.text(vals.max() + 0.015, i + off, f"{vals.mean():.2f}", va="center", fontsize=8.5, color=plotting.INK_2)
    ax.set_yticks(y, [PRETTY[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 (bar = mean of 5 folds, dots = folds)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title("All attributes vs. the BBSO subset, 5-fold CV on the training split")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plotting.use_style()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    timings = {}

    df = data.load_dataset()
    train, test = data.make_split(df)
    X_train, y_train = data.split_xy(train)
    X_test, y_test = data.split_xy(test)

    # ---- BBSO over ten seeds, shared fitness cache ----------------------- #
    t0 = time.time()
    fitness = MaskFitness(X_train, y_train, n_splits=3, seed=config.SEED)
    timings["fitness_setup_s"] = round(time.time() - t0, 2)
    cfg = BBSOConfig()
    all_on = fitness(np.ones(fitness.n_features, dtype=bool))

    t0 = time.time()
    results, rows = [], []
    for seed in SEEDS:
        ts = time.time()
        res = BBSO(fitness, fitness.n_features, cfg, seed=seed).run()
        results.append(res)
        chosen = fitness.selected(res.best_mask)
        rows.append({"seed": seed, "best_macro_f1": res.best_fitness, "n_iter": res.n_iter,
                     "stopped_early": res.stopped_early, "n_selected": len(chosen),
                     "selected": " ".join(chosen), "seconds": round(time.time() - ts, 1)})
        print(f"seed {seed}: macro-F1 {res.best_fitness:.4f}  iters {res.n_iter:2d}  "
              f"{len(chosen):2d} attrs  {rows[-1]['seconds']:5.1f}s  [{rows[-1]['selected']}]")
    timings["bbso_total_s"] = round(time.time() - t0, 1)
    timings["unique_masks_evaluated"] = fitness.n_evals
    timings["fitness_calls"] = sum(cfg.n_pop * (1 + r.n_iter) for r in results)

    runs = pd.DataFrame(rows)
    runs.to_csv(config.RESULTS_DIR / "03_bbso_runs.csv", index=False)
    freq = selection_frequency(results, fitness.attributes)
    freq.to_frame().to_csv(config.RESULTS_DIR / "03_selection_frequency.csv")
    subset, how = consensus_subset(results, fitness.attributes)
    subset_fit = fitness(np.array([a in subset for a in fitness.attributes]))

    # ---- Evaluate subset vs all 13 with the phase 2 protocol ------------- #
    t0 = time.time()
    fold_rows, test_rows, site_rows = [], [], []
    for name in MODELS:
        for variant, attrs in (("all", None), ("subset", subset)):
            model = make_baselines()[name]
            res = evaluation.cross_validate(f"{name}/{variant}", make_pipeline(model, attributes=attrs), X_train, y_train)
            fold_rows.append(res.fold_metrics.assign(model=name, variant=variant))
            fitted = make_pipeline(make_baselines()[name], attributes=attrs).fit(X_train, y_train)
            pred = fitted.predict(X_test)
            test_rows.append({"model": name, "variant": variant, **evaluation.score(y_test, pred),
                              **{f"f1_{c}": v for c, v in evaluation.per_class_f1(y_test, pred).items()}})
            if variant == "subset":
                site_rows.append(evaluation.per_site(y_train, res.oof_pred, train[config.SITE_COL]).assign(model=name))
    timings["evaluation_s"] = round(time.time() - t0, 1)

    folds = pd.concat(fold_rows)
    cv_table = folds.groupby(["model", "variant"])[evaluation.METRICS].agg(["mean", "std"]).round(4)
    cv_table.to_csv(config.RESULTS_DIR / "03_subset_vs_all_cv.csv")
    test_table = pd.DataFrame(test_rows).set_index(["model", "variant"])
    test_table.to_csv(config.RESULTS_DIR / "03_subset_vs_all_test.csv")
    pd.concat(site_rows).reset_index().to_csv(config.RESULTS_DIR / "03_per_site_subset.csv", index=False)
    timings["chosen_subset"] = subset
    timings["chosen_by"] = how
    (config.RESULTS_DIR / "03_runtime.json").write_text(json.dumps(timings, indent=2))

    plot_frequency(freq, subset, config.FIGURES_DIR / "03_selection_frequency.png")
    plot_convergence({r.seed: r.history for r in results}, config.FIGURES_DIR / "03_convergence.png")
    plot_subset_vs_all(folds, config.FIGURES_DIR / "03_subset_vs_all.png")

    # ---- Report ----------------------------------------------------------- #
    print(f"\ninner-CV macro-F1 with all 13 attributes: {all_on:.4f}")
    print(f"chosen subset ({how}): {subset}  inner-CV macro-F1 {subset_fit:.4f}")
    print("\nselection frequency:")
    print(freq.round(2).to_string())
    print("\n5-fold CV macro-F1, all vs subset:")
    print(cv_table["macro_f1"].to_string())
    print("\nheld-out test:")
    print(test_table[evaluation.METRICS].round(3).to_string())
    print("\ntimings:", json.dumps({k: v for k, v in timings.items() if k not in ("chosen_subset", "chosen_by")}))


if __name__ == "__main__":
    main()
