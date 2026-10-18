"""Phase 2: baseline classifiers under stratified CV and on the held-out split.

Writes:
    results/02_baselines_cv.csv         mean/std of each metric over 5 folds
    results/02_baselines_folds.csv      every fold of every model
    results/02_baselines_test.csv       held-out test metrics (fit on all training rows)
    results/02_per_site_oof.csv         per-hospital metrics on out-of-fold training predictions
    results/02_per_site_test.csv        per-hospital metrics on the test split
    figures/02_baseline_cv.png
    figures/02_confusion_matrices.png
    figures/02_per_site.png

Run from the repository root:  python scripts/phase2_baselines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfm import config, data, evaluation, plotting  # noqa: E402
from hfm.models import make_baselines, make_pipeline  # noqa: E402

PRETTY = {
    "majority_class": "Majority class",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "mlp": "MLP (sklearn)",
}


def plot_cv(folds: pd.DataFrame, path: Path) -> None:
    """Mean macro-F1 per model as thin bars, with the five fold values as dots."""
    order = [m for m in PRETTY if m in folds["model"].unique()]
    means = folds.groupby("model")["macro_f1"].mean().loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(order))
    ax.barh(y, means.values, height=0.5, color=plotting.CATEGORICAL[0])
    for i, m in enumerate(order):
        vals = folds.loc[folds["model"] == m, "macro_f1"].values
        ax.scatter(vals, np.full_like(vals, i, dtype=float), s=22, color=plotting.INK,
                   edgecolor=plotting.SURFACE, linewidth=1, zorder=3)
        ax.text(vals.max() + 0.02, i, f"{means[m]:.2f}", va="center", fontsize=9, color=plotting.INK_2)
    ax.set_yticks(y, [PRETTY[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 (bar = mean of 5 folds, dots = folds)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Baseline models, 5-fold stratified CV on the training split")
    fig.savefig(path)
    plt.close(fig)


def plot_confusions(cms: dict[str, pd.DataFrame], path: Path) -> None:
    names = [m for m in PRETTY if m in cms]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.1), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), names):
        cm = cms[m]
        norm = cm.div(cm.sum(axis=1), axis=0)
        ax.imshow(norm.values, cmap=plotting.BLUE_CMAP, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm.values[i, j]}", ha="center", va="center", fontsize=9,
                        color=plotting.SURFACE if norm.values[i, j] > 0.55 else plotting.INK_2)
        ax.set_xticks(range(3), config.RISK_LEVELS, fontsize=8.5)
        ax.set_yticks(range(3), config.RISK_LEVELS, fontsize=8.5)
        ax.set_xlabel("predicted")
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_title(PRETTY[m], fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("true")
    fig.suptitle("Held-out test split (184 patients); counts, shaded by row share",
                 x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path)
    plt.close(fig)


def plot_per_site(per_site: pd.DataFrame, path: Path) -> None:
    """Accuracy per hospital, one dot per model; a diagnostic view only."""
    models = [m for m in PRETTY if m in per_site["model"].unique()]
    sites = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    offsets = np.linspace(-0.21, 0.21, len(models))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for k, m in enumerate(models):
        sub = per_site[per_site["model"] == m].set_index("site").loc[sites]
        ax.scatter(np.arange(len(sites)) + offsets[k], sub["accuracy"].values, s=46,
                   color=plotting.CATEGORICAL[k], label=PRETTY[m],
                   edgecolor=plotting.SURFACE, linewidth=1, zorder=3)
    for x in np.arange(1, len(sites)) - 0.5:
        ax.axvline(x, color=plotting.GRID, lw=0.8, zorder=1)
    n = per_site[per_site["model"] == models[0]].set_index("site").loc[sites, "n"]
    ax.set_xticks(range(len(sites)), [f"{s}\n(n={int(n[s])})" for s in sites])
    ax.set_xlim(-0.5, len(sites) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy (out-of-fold, training rows)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Accuracy by source hospital (diagnostic only; hospital is not a feature)")
    ax.legend(loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, -0.26))
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plotting.use_style()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)

    df = data.load_dataset()
    train, test = data.make_split(df)
    X_train, y_train = data.split_xy(train)
    X_test, y_test = data.split_xy(test)

    baselines = make_baselines()
    cv_results, fold_rows, test_rows, site_oof, site_test, cms = [], [], [], [], [], {}
    for name, model in baselines.items():
        t0 = time.time()
        pipe = make_pipeline(model)
        res = evaluation.cross_validate(name, pipe, X_train, y_train)
        cv_results.append(res)
        fold_rows.append(res.fold_metrics.assign(model=name))

        fitted = make_pipeline(model).fit(X_train, y_train)
        pred = fitted.predict(X_test)
        test_rows.append({"model": name, **evaluation.score(y_test, pred),
                          **{f"f1_{c}": v for c, v in evaluation.per_class_f1(y_test, pred).items()}})
        cms[name] = evaluation.confusion_frame(y_test, pred)
        site_oof.append(evaluation.per_site(y_train, res.oof_pred, train[config.SITE_COL]).assign(model=name))
        site_test.append(evaluation.per_site(y_test, pred, test[config.SITE_COL]).assign(model=name))
        extra = f"  iters={fitted.named_steps['model'].n_iter_}" if name == "mlp" else ""
        print(f"{name:20s} CV macro-F1 {res.fold_metrics.macro_f1.mean():.3f} ± "
              f"{res.fold_metrics.macro_f1.std():.3f}   test macro-F1 {test_rows[-1]['macro_f1']:.3f}"
              f"   ({time.time() - t0:.1f}s){extra}")

    cv_table = evaluation.summarize(cv_results)
    cv_table.to_csv(config.RESULTS_DIR / "02_baselines_cv.csv")
    pd.concat(fold_rows).to_csv(config.RESULTS_DIR / "02_baselines_folds.csv", index=False)
    test_table = pd.DataFrame(test_rows).set_index("model")
    test_table.to_csv(config.RESULTS_DIR / "02_baselines_test.csv")
    site_oof_df = pd.concat(site_oof).reset_index()
    site_oof_df.to_csv(config.RESULTS_DIR / "02_per_site_oof.csv", index=False)
    site_test_df = pd.concat(site_test).reset_index()
    site_test_df.to_csv(config.RESULTS_DIR / "02_per_site_test.csv", index=False)

    plot_cv(pd.concat(fold_rows), config.FIGURES_DIR / "02_baseline_cv.png")
    plot_confusions(cms, config.FIGURES_DIR / "02_confusion_matrices.png")
    plot_per_site(site_oof_df, config.FIGURES_DIR / "02_per_site.png")

    print("\n5-fold CV on training split (mean ± std):")
    print(cv_table.round(3).to_string())
    print("\nHeld-out test split:")
    print(test_table.round(3).to_string())
    print("\nPer hospital, out-of-fold training predictions:")
    print(site_oof_df.pivot(index="site", columns="model", values="accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
