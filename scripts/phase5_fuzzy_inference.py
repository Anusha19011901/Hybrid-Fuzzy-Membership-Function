"""Phase 5: fuzzy inference, weighted average, centroid defuzzification.

Evaluates the HybridFuzzyClassifier on the BBSO subset with the phase 2
protocol (5-fold stratified CV on the training split, held-out test split,
per-hospital diagnostic) and produces the explainability outputs.

Writes:
    results/05_rules.csv                  31 single-antecedent rules with consequents, support, weight
    results/05_attribute_weights.csv      inner-CV macro F1 and weight per attribute
    results/05_fuzzy_cv.csv / _folds.csv  5-fold CV metrics
    results/05_fuzzy_test.csv             held-out test metrics (+ per-class F1)
    results/05_per_site.csv               per-hospital metrics on out-of-fold predictions
    results/05_cutpoints.json             learned cut-points on the risk axis
    results/05_explanations.json          the two example explanations (numbers only)
    figures/05_rules.png                  rule consequents as a heatmap
    figures/05_attribute_weights.png
    figures/05_fuzzy_vs_baselines.png
    figures/05_risk_score_distribution.png
    figures/05_explain_patient_complete.png / _missing.png

Run from the repository root:  python scripts/phase5_fuzzy_inference.py
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
from hfm.fuzzy_model import CLASSES, HybridFuzzyClassifier  # noqa: E402

PRETTY = {"majority_class": "Majority class", "logistic_regression": "Logistic regression",
          "random_forest": "Random forest", "mlp": "MLP (sklearn)", "hybrid_fuzzy": "Hybrid fuzzy (this work)"}


def selected_attributes() -> list[str]:
    path = config.SELECTED_ATTRIBUTES_JSON
    return json.loads(path.read_text())["chosen_subset"] if path.exists() else list(config.FEATURE_COLS)


def fmt_value(attr: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not measured"
    if attr in config.CONTINUOUS_COLS:
        unit = config.ATTRIBUTE_UNITS.get(attr, "")
        return f"{v:g} {unit}".strip()
    if attr in config.DISCRETE_COLS:
        return f"{int(round(float(v)))}"
    return str(v)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_rules(rules: pd.DataFrame, path: Path) -> None:
    rules = rules.copy()
    rules["label"] = rules["attribute"] + " is " + rules["term"]
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(rules) + 1.4))
    im = ax.imshow(rules[CLASSES].to_numpy(), cmap=plotting.BLUE_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rules)):
        for j, c in enumerate(CLASSES):
            v = rules[c].iloc[i]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=plotting.SURFACE if v > 0.55 else plotting.INK_2)
        ax.text(len(CLASSES) - 0.4, i, f"n≈{rules['support'].iloc[i]:.0f}", va="center", fontsize=7.5, color=plotting.MUTED)
    ax.set_xticks(range(len(CLASSES)), [f"THEN risk is {c}" for c in CLASSES], fontsize=8.5)
    ax.set_yticks(range(len(rules)), ["IF " + l for l in rules["label"]], fontsize=8.5)
    # Separate attributes with hairlines.
    prev = None
    for i, a in enumerate(rules["attribute"]):
        if prev is not None and a != prev:
            ax.axhline(i - 0.5, color=plotting.SURFACE, lw=2)
        prev = a
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, len(CLASSES) + 0.3)
    ax.set_title("The learned rule base: class share of the patients in each term (training split)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def plot_weights(skill: pd.DataFrame, floor: float, path: Path) -> None:
    sk = skill.sort_values("weight", ascending=False)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(sk))
    ax.barh(y, sk["inner_macro_f1"], height=0.6, color=plotting.BLUE_RAMP[7])
    ax.axvline(floor, color=plotting.INK_2, lw=1, ls="--")
    ax.text(floor + 0.006, -0.75, "majority-class floor", fontsize=8, color=plotting.INK_2, va="center")
    for i, (a, r) in enumerate(sk.iterrows()):
        ax.text(r["inner_macro_f1"] + 0.01, i, f"weight {r['weight']:.2f}   (n={int(r['n_available'])})",
                va="center", fontsize=8.5, color=plotting.INK_2)
    ax.set_yticks(y, [config.ATTRIBUTE_LABELS[a] for a in sk.index], fontsize=8.5)
    ax.set_ylim(len(sk) - 0.5, -1.1)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Macro F1 of the single-attribute fuzzy model (inner 3-fold CV, training split)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Attribute weights = skill above the majority-class floor, normalised")
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(folds: pd.DataFrame, path: Path) -> None:
    order = [m for m in PRETTY if m in folds["model"].unique()]
    means = folds.groupby("model")["macro_f1"].mean().loc[order]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(order))
    colors = [plotting.CATEGORICAL[1] if m == "hybrid_fuzzy" else plotting.CATEGORICAL[0] for m in order]
    ax.barh(y, means.values, height=0.5, color=colors)
    for i, m in enumerate(order):
        vals = folds.loc[folds["model"] == m, "macro_f1"].values
        ax.scatter(vals, np.full(len(vals), i, dtype=float), s=20, color=plotting.INK,
                   edgecolor=plotting.SURFACE, linewidth=1, zorder=3)
        ax.text(vals.max() + 0.02, i, f"{means[m]:.2f}", va="center", fontsize=9, color=plotting.INK_2)
    ax.set_yticks(y, [PRETTY[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 (bar = mean of 5 folds, dots = folds)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Hybrid fuzzy model vs. baselines, 5-fold stratified CV on the training split")
    fig.savefig(path)
    plt.close(fig)


def plot_risk_distribution(risk: np.ndarray, y: np.ndarray, cut: tuple[float, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    bins = np.linspace(0, 1, 41)
    for c in CLASSES:
        ax.hist(risk[y == c], bins=bins, histtype="step", lw=1.8, color=plotting.RISK_COLORS[c], label=f"true {c}")
    top = ax.get_ylim()[1]
    for t, lab, ha, dx in ((cut[0], "Low | Moderate", "right", -0.006), (cut[1], "Moderate | High", "left", 0.006)):
        ax.axvline(t, color=plotting.INK_2, lw=1, ls="--")
        ax.text(t + dx, top * 0.97, f"{lab}\n{t:.2f}", fontsize=8, color=plotting.INK_2, va="top", ha=ha)
    ax.set_xlabel("Defuzzified risk score (out-of-fold, training split)")
    ax.set_ylabel("Patients")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Where the three true classes land on the risk axis, and the learned cut-points")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.8))
    fig.savefig(path)
    plt.close(fig)


def plot_explanation(ex: dict, row: pd.Series, meta: dict, path: Path) -> None:
    rows = sorted(ex["attributes"], key=lambda r: -r["weight"])
    n = len(rows)
    fig = plt.figure(figsize=(12.4, 0.42 * n + 3.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[n, 1.6], wspace=0.08, hspace=0.35)
    axl = fig.add_subplot(gs[0, 0])
    axc = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[:, 1])

    # ---- left, top: per-attribute contributions -------------------------- #
    xmax = max([sum(r["contribution"].values()) for r in rows if not r["missing"]] + [0.1])
    for i, r in enumerate(rows):
        yv = n - i
        if r["missing"]:
            axl.text(0.003, yv, "not measured, skipped", va="center", fontsize=8.5, color=plotting.MUTED, style="italic")
            continue
        left = 0.0
        for c in CLASSES:
            w = r["contribution"][c]
            axl.barh(yv, w, left=left, height=0.62, color=plotting.RISK_COLORS[c],
                     edgecolor=plotting.SURFACE, linewidth=1.2)
            left += w
        fired = "  ·  ".join(f"{t} {m:.2f}" for t, m in sorted(r["fired_terms"].items(), key=lambda kv: -kv[1]))
        axl.text(left + 0.004, yv, f"{fired}   (w {r['effective_weight']:.2f})",
                 va="center", fontsize=8, color=plotting.INK_2)
    labels = [f"{r['attribute']} = {fmt_value(r['attribute'], r['value'])}" for r in rows]
    axl.set_yticks([n - i for i in range(n)], labels, fontsize=8.5)
    axl.set_ylim(0.3, n + 0.7)
    axl.set_xlim(0, xmax * 2.6)
    axl.set_xlabel("Contribution to the class scores (effective weight × rule output)", fontsize=9)
    axl.grid(axis="y", visible=False)
    axl.tick_params(length=0)
    axl.set_title("1 · Which terms fired and what each attribute contributed", fontsize=10)

    # ---- left, bottom: the combined scores -------------------------------- #
    left = 0.0
    for c in CLASSES:
        w = ex["combined_scores"][c]
        axc.barh(0, w, left=left, height=0.6, color=plotting.RISK_COLORS[c], edgecolor=plotting.SURFACE, linewidth=1.2)
        axc.text(left + w / 2, 0, f"{c} {w:.2f}", ha="center", va="center", fontsize=8.5,
                 color=plotting.SURFACE if c != "Low" else plotting.INK)
        left += w
    axc.set_yticks([0], ["combined\n(weighted average)"], fontsize=8.5)
    axc.set_xlim(0, 1)
    axc.set_ylim(-0.6, 0.6)
    axc.set_xlabel("Class scores (sum to 1); these clip the output sets on the right", fontsize=9)
    axc.grid(False)
    axc.tick_params(length=0)

    # ---- right: output sets, aggregation, centroid, cut-points ----------- #
    u = ex["output_axis"]
    t1, t2 = ex["cutpoints"]
    for x0, x1, c in ((0, t1, "Low"), (t1, t2, "Moderate"), (t2, 1, "High")):
        axr.text((x0 + x1) / 2, 1.27, c, ha="center", va="center", fontsize=9,
                 color=plotting.INK if c == ex["predicted"] else plotting.MUTED,
                 fontweight="bold" if c == ex["predicted"] else "normal")
    for k, c in enumerate(CLASSES):
        axr.plot(u, ex["output_sets"][k], color=plotting.RISK_COLORS[c], lw=1.2, alpha=0.6)
        axr.fill_between(u, 0, ex["clipped_sets"][k], color=plotting.RISK_COLORS[c], alpha=0.25, lw=0)
    axr.plot(u, ex["aggregated"], color=plotting.INK, lw=1.6)
    for t in (t1, t2):
        axr.vlines(t, 0, 1.08, color=plotting.INK_2, lw=0.9, ls="--")
        axr.text(t, 1.14, f"{t:.2f}", ha="center", va="center", fontsize=7.5, color=plotting.INK_2)
    axr.vlines(ex["risk_score"], 0, 1.08, color=plotting.CATEGORICAL[1], lw=2)
    axr.text(ex["risk_score"] + 0.012, 0.93, f"centroid\nrisk = {ex['risk_score']:.2f}", fontsize=9,
             color=plotting.CATEGORICAL[1], va="top", fontweight="semibold")
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1.36)
    axr.set_yticks([0, 0.5, 1])
    axr.set_xlabel("Risk axis (output sets clipped by the combined scores; dashed = cut-points)", fontsize=9)
    axr.set_ylabel("membership")
    axr.grid(axis="x", visible=False)
    axr.tick_params(length=0)
    axr.set_title("2 · Output sets, aggregated shape, centroid, decision", fontsize=10)

    fig.suptitle(f"Patient {meta['id']} ({meta['site']}): predicted {ex['predicted']}, true {meta['true']}   ·   "
                 f"{ex['n_attributes_used']} of {n} attributes measured",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    plotting.use_style()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)
    t_all = time.time()

    df = data.load_dataset()
    train, test = data.make_split(df)
    X_train, y_train = data.split_xy(train)
    X_test, y_test = data.split_xy(test)
    attrs = selected_attributes()

    # ---- CV with the phase 2 protocol; argmax variant as a diagnostic ---- #
    cv_results, fold_frames, test_rows, sites = [], [], [], []
    for name, decision in (("hybrid_fuzzy", "centroid"), ("hybrid_fuzzy_argmax", "argmax")):
        est = HybridFuzzyClassifier(attributes=attrs, decision=decision)
        r = evaluation.cross_validate(name, est, X_train, y_train)
        cv_results.append(r)
        fold_frames.append(r.fold_metrics.assign(model=name))
        sites.append(evaluation.per_site(y_train, r.oof_pred, train[config.SITE_COL]).assign(model=name))
        m = HybridFuzzyClassifier(attributes=attrs, decision=decision).fit(X_train, y_train)
        p = m.predict(X_test)
        test_rows.append({"model": name, **evaluation.score(y_test, p),
                          **{f"f1_{c}": v for c, v in evaluation.per_class_f1(y_test, p).items()}})
        if decision == "centroid":
            res, model, pred, test_row, site = r, m, p, test_rows[-1], sites[-1].drop(columns="model")
    folds = fold_frames[0]
    pd.concat(fold_frames).to_csv(config.RESULTS_DIR / "05_fuzzy_folds.csv", index=False)
    evaluation.summarize(cv_results).to_csv(config.RESULTS_DIR / "05_fuzzy_cv.csv")
    pd.concat(sites).reset_index().to_csv(config.RESULTS_DIR / "05_per_site.csv", index=False)
    pd.DataFrame(test_rows).set_index("model").to_csv(config.RESULTS_DIR / "05_fuzzy_test.csv")
    cm = evaluation.confusion_frame(y_test, pred)

    rules = model.rule_table()
    rules.to_csv(config.RESULTS_DIR / "05_rules.csv", index=False)
    model.attribute_skill_.to_csv(config.RESULTS_DIR / "05_attribute_weights.csv")
    (config.RESULTS_DIR / "05_cutpoints.json").write_text(json.dumps(
        {"low_moderate": model.cutpoints_[0], "moderate_high": model.cutpoints_[1], "f1_floor": model.f1_floor_}, indent=2))

    # ---- explanations ----------------------------------------------------- #
    measured = X_test[attrs].notna().sum(axis=1)
    def pick(mask):
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else 0
    i_complete = pick((measured == len(attrs)).to_numpy() & (test[config.SITE_COL] == "Cleveland").to_numpy()
                      & (pred == y_test.to_numpy()) & (y_test.to_numpy() == "High"))
    i_missing = pick((measured <= len(attrs) - 2).to_numpy() & (test[config.SITE_COL] == "VA Long Beach").to_numpy()
                     & (pred == y_test.to_numpy()))
    explanations = {}
    for tag, i in (("complete", i_complete), ("missing", i_missing)):
        row = X_test.iloc[[i]]
        ex = model.explain(row)
        meta = {"id": int(test["id"].iloc[i]), "site": test[config.SITE_COL].iloc[i], "true": y_test.iloc[i],
                "test_index": i}
        plot_explanation(ex, row.iloc[0], meta, config.FIGURES_DIR / f"05_explain_patient_{tag}.png")
        explanations[tag] = {"meta": meta, "predicted": ex["predicted"], "risk_score": ex["risk_score"],
                             "combined_scores": ex["combined_scores"], "n_attributes_used": ex["n_attributes_used"],
                             "attributes": [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                                             for k, v in r.items()} for r in ex["attributes"]]}
    (config.RESULTS_DIR / "05_explanations.json").write_text(json.dumps(explanations, indent=2, default=str))

    # ---- figures ---------------------------------------------------------- #
    plot_rules(rules, config.FIGURES_DIR / "05_rules.png")
    plot_weights(model.attribute_skill_, model.f1_floor_, config.FIGURES_DIR / "05_attribute_weights.png")
    base_folds = pd.read_csv(config.RESULTS_DIR / "02_baselines_folds.csv")
    plot_comparison(pd.concat([base_folds, folds]), config.FIGURES_DIR / "05_fuzzy_vs_baselines.png")
    plot_risk_distribution(model.oof_risk_, y_train.to_numpy(), model.cutpoints_,
                           config.FIGURES_DIR / "05_risk_score_distribution.png")

    # ---- report ----------------------------------------------------------- #
    print(f"attributes: {attrs}")
    print("\n5-fold CV (mean ± std):")
    print(evaluation.summarize(cv_results).round(3).to_string())
    print("\nheld-out test:")
    print(pd.DataFrame(test_rows).set_index("model").round(3).to_string())
    print(f"cut-points: {model.cutpoints_}   floor {model.f1_floor_:.3f}")
    print("\nconfusion (test):"); print(cm)
    print("\nattribute skill and weights:"); print(model.attribute_skill_.round(3).to_string())
    print("\nper hospital (OOF):"); print(site.round(3).to_string())
    print("\nrules:"); print(rules.drop(columns="weight").round(2).to_string(index=False))
    for tag, e in explanations.items():
        print(f"\nexplanation [{tag}] patient {e['meta']['id']} {e['meta']['site']}: true {e['meta']['true']} "
              f"pred {e['predicted']} risk {e['risk_score']:.3f} used {e['n_attributes_used']}")
    print(f"\ntotal runtime {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
