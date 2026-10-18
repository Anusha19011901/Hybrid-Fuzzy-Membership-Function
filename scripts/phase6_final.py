"""Phase 6: consolidated evaluation of every model under one protocol.

Refits the four baselines (all 13 attributes), the main hybrid fuzzy model
(single-antecedent rules) and its pairwise variant, then reports:

* 5-fold stratified CV on the training split (mean ± sd),
* held-out test metrics with 2000-sample bootstrap 95% CIs,
* paired bootstrap differences against the best baseline,
* whether each model's CV mean lies inside its test CI,
* per-hospital metrics on out-of-fold predictions for every model.

Writes results/06_*.csv|json and figures/06_*.png.
Run from the repository root:  python scripts/phase6_final.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfm import config, data, evaluation, plotting  # noqa: E402
from hfm.fuzzy_model import CLASSES, HybridFuzzyClassifier  # noqa: E402
from hfm.models import make_baselines, make_pipeline  # noqa: E402

PRETTY = {
    "majority_class": "Majority class",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "mlp": "MLP (sklearn)",
    "hybrid_fuzzy": "Hybrid fuzzy (main, 31 rules)",
    "hybrid_fuzzy_pairs": "Hybrid fuzzy + pairwise rules",
}
FUZZY = {"hybrid_fuzzy", "hybrid_fuzzy_pairs"}
REFERENCE = "random_forest"
SITES = ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]


def selected_attributes() -> list[str]:
    return json.loads(config.SELECTED_ATTRIBUTES_JSON.read_text())["chosen_subset"]


def plot_test_ci(summary: pd.DataFrame, path: Path) -> None:
    order = [m for m in PRETTY if m in summary.index]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=True)
    for ax, metric, label in zip(axes, ["macro_f1", "accuracy"], ["Macro F1", "Accuracy"]):
        y = np.arange(len(order))
        for i, m in enumerate(order):
            r = summary.loc[m]
            col = plotting.CATEGORICAL[1] if m in FUZZY else plotting.CATEGORICAL[0]
            ax.plot([r[f"test_{metric}_lo"], r[f"test_{metric}_hi"]], [i, i], color=col, lw=2.2, solid_capstyle="round")
            ax.scatter([r[f"test_{metric}"]], [i], s=46, color=col, edgecolor=plotting.SURFACE, linewidth=1, zorder=3)
            ax.scatter([r[f"cv_{metric}_mean"]], [i], s=46, marker="D", facecolor=plotting.SURFACE,
                       edgecolor=plotting.INK_2, linewidth=1.2, zorder=4)
        ax.set_yticks(y, [PRETTY[m] for m in order])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"{label} on the 184-patient test split")
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
    axes[0].set_title("Test estimate with bootstrap 95% CI (line); diamond = 5-fold CV mean", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_per_site(site: pd.DataFrame, path: Path) -> None:
    models = [m for m in PRETTY if m in site["model"].unique()]
    offsets = np.linspace(-0.3, 0.3, len(models))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for k, m in enumerate(models):
        sub = site[site["model"] == m].set_index("site").loc[SITES]
        ax.scatter(np.arange(len(SITES)) + offsets[k], sub["macro_f1"].values, s=42,
                   color=plotting.CATEGORICAL[k], label=PRETTY[m], edgecolor=plotting.SURFACE, linewidth=1, zorder=3)
    for x in np.arange(1, len(SITES)) - 0.5:
        ax.axvline(x, color=plotting.GRID, lw=0.8, zorder=1)
    n = site[site["model"] == models[0]].set_index("site").loc[SITES, "n"]
    ax.set_xticks(range(len(SITES)), [f"{s}\n(n={int(n[s])})" for s in SITES])
    ax.set_xlim(-0.5, len(SITES) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1 over classes present at the site\n(out-of-fold, training rows)")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Per-hospital diagnostic for every model (hospital is never a feature)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.3), fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def plot_rules(rules: pd.DataFrame, path: Path, title: str) -> None:
    rules = rules.copy()
    rules["label"] = "IF " + rules["attribute"] + " is " + rules["term"]
    fig, ax = plt.subplots(figsize=(8.0, 0.26 * len(rules) + 1.4))
    ax.imshow(rules[CLASSES].to_numpy(), cmap=plotting.BLUE_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rules)):
        for j, c in enumerate(CLASSES):
            v = rules[c].iloc[i]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=plotting.SURFACE if v > 0.55 else plotting.INK_2)
        ax.text(len(CLASSES) - 0.4, i, f"n≈{rules['support'].iloc[i]:.0f}", va="center", fontsize=7, color=plotting.MUTED)
    prev = None
    for i, a in enumerate(rules["attribute"]):
        if prev is not None and a != prev:
            ax.axhline(i - 0.5, color=plotting.SURFACE, lw=2)
        prev = a
    ax.set_xticks(range(len(CLASSES)), [f"THEN risk is {c}" for c in CLASSES], fontsize=8.5)
    ax.set_yticks(range(len(rules)), rules["label"], fontsize=7.5)
    ax.set_xlim(-0.5, len(CLASSES) + 0.3)
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plotting.use_style()
    t_all = time.time()
    df = data.load_dataset()
    train, test = data.make_split(df)
    X_train, y_train = data.split_xy(train)
    X_test, y_test = data.split_xy(test)
    attrs = selected_attributes()

    estimators = {name: make_pipeline(model) for name, model in make_baselines().items()}
    estimators["hybrid_fuzzy"] = HybridFuzzyClassifier(attributes=attrs)
    estimators["hybrid_fuzzy_pairs"] = HybridFuzzyClassifier(attributes=attrs, pair_top_k=3)

    preds, rows, folds, sites, oof_class = {}, [], [], [], []
    fitted = {}
    for name, est in estimators.items():
        t0 = time.time()
        res = evaluation.cross_validate(name, est, X_train, y_train)
        folds.append(res.fold_metrics.assign(model=name))
        sites.append(evaluation.per_site(y_train, res.oof_pred, train[config.SITE_COL]).assign(model=name))
        oof_f1 = f1_score(y_train, res.oof_pred, labels=CLASSES, average=None, zero_division=0)
        oof_class.append({"model": name, **{f"oof_f1_{c}": v for c, v in zip(CLASSES, oof_f1)}})
        from sklearn.base import clone
        model = clone(est).fit(X_train, y_train)
        fitted[name] = model
        preds[name] = model.predict(X_test)
        ci = evaluation.bootstrap_ci(y_test, preds[name])
        test_f1 = f1_score(y_test, preds[name], labels=CLASSES, average=None, zero_division=0)
        s = res.summary
        rows.append({
            "model": name,
            "cv_macro_f1_mean": s["macro_f1_mean"], "cv_macro_f1_sd": s["macro_f1_std"],
            "cv_accuracy_mean": s["accuracy_mean"], "cv_accuracy_sd": s["accuracy_std"],
            "test_macro_f1": ci.loc["macro_f1", "estimate"], "test_macro_f1_lo": ci.loc["macro_f1", "ci_low"],
            "test_macro_f1_hi": ci.loc["macro_f1", "ci_high"],
            "test_accuracy": ci.loc["accuracy", "estimate"], "test_accuracy_lo": ci.loc["accuracy", "ci_low"],
            "test_accuracy_hi": ci.loc["accuracy", "ci_high"],
            **{f"test_f1_{c}": v for c, v in zip(CLASSES, test_f1)},
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{name:20s} CV macro-F1 {s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}  "
              f"test {ci.loc['macro_f1', 'estimate']:.3f} [{ci.loc['macro_f1', 'ci_low']:.3f}, {ci.loc['macro_f1', 'ci_high']:.3f}]"
              f"  ({rows[-1]['seconds']}s)")

    summary = pd.DataFrame(rows).set_index("model")
    summary["cv_mean_inside_test_ci"] = (summary["cv_macro_f1_mean"] >= summary["test_macro_f1_lo"]) & \
                                        (summary["cv_macro_f1_mean"] <= summary["test_macro_f1_hi"])
    summary = summary.join(pd.DataFrame(oof_class).set_index("model"))
    summary.to_csv(config.RESULTS_DIR / "06_summary.csv")
    pd.concat(folds).to_csv(config.RESULTS_DIR / "06_folds.csv", index=False)
    site = pd.concat(sites).reset_index()
    site.to_csv(config.RESULTS_DIR / "06_per_site.csv", index=False)

    diffs = []
    for name in estimators:
        if name == REFERENCE:
            continue
        d = evaluation.paired_bootstrap_diff(y_test, preds[name], preds[REFERENCE])
        for metric, r in d.iterrows():
            diffs.append({"model": name, "vs": REFERENCE, "metric": metric, **r.to_dict()})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(config.RESULTS_DIR / "06_paired_diff_vs_rf.csv", index=False)

    # fuzzy specifics: rules of both variants, cut-points, risk-axis spread
    fz_main, fz_pairs = fitted["hybrid_fuzzy"], fitted["hybrid_fuzzy_pairs"]
    fz_pairs.rule_table().to_csv(config.RESULTS_DIR / "06_rules_pairs.csv", index=False)
    spread = pd.DataFrame({
        "model": ["hybrid_fuzzy", "hybrid_fuzzy_pairs"],
        "n_rules": [len(fz_main.rule_table()), len(fz_pairs.rule_table())],
        "pairs": [str(fz_main.pairs_), str(fz_pairs.pairs_)],
        "oof_risk_sd": [fz_main.oof_risk_.std(), fz_pairs.oof_risk_.std()],
        "oof_risk_iqr": [np.subtract(*np.quantile(fz_main.oof_risk_, [0.75, 0.25])),
                         np.subtract(*np.quantile(fz_pairs.oof_risk_, [0.75, 0.25]))],
        "cutpoints": [str(tuple(round(c, 3) for c in fz_main.cutpoints_)),
                      str(tuple(round(c, 3) for c in fz_pairs.cutpoints_))],
    }).set_index("model")
    spread.to_csv(config.RESULTS_DIR / "06_fuzzy_variants.csv")
    (config.RESULTS_DIR / "06_model_choice.json").write_text(json.dumps({
        "main_fuzzy_model": "hybrid_fuzzy",
        "rule": "higher 5-fold CV macro F1 on the training split",
        "cv_macro_f1": {"hybrid_fuzzy": summary.loc["hybrid_fuzzy", "cv_macro_f1_mean"],
                        "hybrid_fuzzy_pairs": summary.loc["hybrid_fuzzy_pairs", "cv_macro_f1_mean"]},
    }, indent=2))

    plot_test_ci(summary, config.FIGURES_DIR / "06_test_ci.png")
    plot_per_site(site, config.FIGURES_DIR / "06_per_site_all.png")
    plot_rules(fz_pairs.rule_table(), config.FIGURES_DIR / "06_rules_pairs.png",
               "Pairwise variant: 31 single-antecedent rules + two-antecedent rules over the top-3 attributes")

    pd.set_option("display.width", 160)
    cols = ["cv_macro_f1_mean", "cv_macro_f1_sd", "test_macro_f1", "test_macro_f1_lo", "test_macro_f1_hi",
            "test_accuracy", "test_accuracy_lo", "test_accuracy_hi", "cv_mean_inside_test_ci", "oof_f1_Moderate", "test_f1_Moderate"]
    print("\n" + summary[cols].round(3).to_string())
    print("\npaired bootstrap difference vs random forest (test split):")
    print(diffs.round(3).to_string(index=False))
    print("\nfuzzy variants:"); print(spread.round(3).to_string())
    print("\nper hospital (OOF macro F1):")
    print(site.pivot(index="site", columns="model", values="macro_f1").round(3).to_string())
    print(f"\ntotal runtime {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
