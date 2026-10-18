"""Phase 4: hybrid membership functions for the selected attributes.

Fits trapezoid + Gaussian hybrid terms (Low / Normal / High) for each
continuous attribute in the BBSO subset and crisp terms for the categorical
ones, on the training split only and from observed values only.

Writes:
    results/04_membership_params.csv   a, b, c, d, mu, sigma per attribute and term
    results/04_shape_check.csv         peaks per hybrid term; flags any non-single-peaked term
    results/04_crisp_terms.csv         categories per crisp attribute, with the training mode
    results/04_term_coverage.csv       mean membership per term on the training split
    figures/04_membership_overview.png one panel per continuous attribute
    figures/04_mf_<attribute>.png      one detailed figure per continuous attribute

Run from the repository root:  python scripts/phase4_membership.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfm import config, data, plotting  # noqa: E402
from hfm.membership import Fuzzifier, shape_check  # noqa: E402

TERM_COLORS = {"Low": plotting.BLUE_RAMP[3], "Normal": plotting.BLUE_RAMP[7], "High": plotting.BLUE_RAMP[11]}
HIST_COLOR = "#e1e0d9"


def selected_attributes() -> list[str]:
    path = config.SELECTED_ATTRIBUTES_JSON
    if path.exists():
        return json.loads(path.read_text())["chosen_subset"]
    return list(config.FEATURE_COLS)


def _axis_label(attr: str) -> str:
    """Label with unit, without repeating a unit the label already carries."""
    label, unit = config.ATTRIBUTE_LABELS[attr], config.ATTRIBUTE_UNITS[attr]
    return label if f"({unit})" in label else f"{label} ({unit})"


def _draw_panel(ax, fam, values: pd.Series, detailed: bool) -> None:
    x = fam.grid()
    obs = values.dropna().to_numpy()
    # Training distribution, faint, scaled to the membership axis.
    counts, edges = np.histogram(obs, bins=30)
    ax.bar((edges[:-1] + edges[1:]) / 2, counts / counts.max() * 0.85, width=np.diff(edges),
           color=HIST_COLOR, edgecolor=plotting.SURFACE, linewidth=0.5, zorder=1,
           label=f"training data (n={len(obs)})" if detailed else None)
    for name, term in fam.terms.items():
        col = TERM_COLORS[name]
        if detailed:
            ax.plot(x, term.trapezoid(x), color=col, lw=1.1, ls=(0, (1, 2)), alpha=0.9, zorder=2)
            ax.plot(x, term.gaussian(x), color=col, lw=1.1, ls="--", alpha=0.9, zorder=2)
        ax.plot(x, term(x), color=col, lw=2.2, zorder=3, label=name)
        # Direct label at the plateau centre.
        b = fam.lo if np.isinf(term.b) else term.b
        c = fam.hi if np.isinf(term.c) else term.c
        ax.text((b + c) / 2, 1.04, name, ha="center", va="bottom", fontsize=8.5, color=col, fontweight="semibold")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(_axis_label(fam.attribute))
    ax.set_ylabel("membership")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def plot_overview(fz: Fuzzifier, train: pd.DataFrame, path: Path) -> None:
    attrs = fz.continuous
    n = len(attrs)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.0 * nrow))
    for ax, a in zip(np.ravel(axes), attrs):
        _draw_panel(ax, fz.families[a], train[a], detailed=False)
        ax.set_title(config.ATTRIBUTE_LABELS[a], fontsize=10)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Hybrid membership functions (max of trapezoid and Gaussian), fitted on training data",
                 x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_detail(fam, values: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    _draw_panel(ax, fam, values, detailed=True)
    ax.set_title(f"{config.ATTRIBUTE_LABELS[fam.attribute]}: hybrid terms and their components")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=plotting.INK_2, lw=2.2, label="hybrid = max(trapezoid, Gaussian)"),
        Line2D([], [], color=plotting.INK_2, lw=1.1, ls=(0, (1, 2)), label="trapezoid (training quantiles)"),
        Line2D([], [], color=plotting.INK_2, lw=1.1, ls="--", label="Gaussian (mean, sd of the term's core)"),
        plt.Rectangle((0, 0), 1, 1, color=HIST_COLOR, label=f"training data (n={values.notna().sum()})"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28), fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plotting.use_style()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(exist_ok=True)

    df = data.load_dataset()
    train, test = data.make_split(df)
    attrs = selected_attributes()
    fz = Fuzzifier(attrs).fit(train)
    M_train = fz.transform(train)
    M_test = fz.transform(test)

    params = fz.params()
    params.to_csv(config.RESULTS_DIR / "04_membership_params.csv", index=False)
    checks = pd.concat([shape_check(fz.families[a]) for a in fz.continuous], ignore_index=True)
    checks.to_csv(config.RESULTS_DIR / "04_shape_check.csv", index=False)
    crisp = pd.DataFrame([{"attribute": a, "categories": " | ".join(fz.families[a].categories),
                           "n_terms": len(fz.families[a].categories), "training_mode": fz.families[a].train_mode,
                           "missing_train": int(train[a].isna().sum()),
                           "missing_train_pct": round(100 * train[a].isna().mean(), 1)}
                          for a in fz.crisp])
    crisp.to_csv(config.RESULTS_DIR / "04_crisp_terms.csv", index=False)
    coverage = M_train.mean().rename("mean_membership_train").to_frame()
    coverage["mean_membership_test"] = M_test.mean()
    coverage.to_csv(config.RESULTS_DIR / "04_term_coverage.csv")

    plot_overview(fz, train, config.FIGURES_DIR / "04_membership_overview.png")
    for a in fz.continuous:
        plot_detail(fz.families[a], train[a], config.FIGURES_DIR / f"04_mf_{a}.png")

    print(f"attributes: {attrs}")
    print(f"continuous (hybrid terms): {fz.continuous}")
    print(f"crisp: {fz.crisp}")
    print(f"\nmembership columns: {M_train.shape[1]}  (train rows {len(M_train)}, test rows {len(M_test)})")
    print("\nhybrid term parameters (clinical units):")
    print(params.round(2).to_string(index=False))
    print("\nshape check:")
    print(checks.round(3).to_string(index=False))
    flagged = checks[~checks["quasi_concave"]]
    print(f"\nnon-single-peaked terms: {len(flagged)}" + ("" if flagged.empty else f"\n{flagged}"))
    print("\ncrisp terms:")
    print(crisp.to_string(index=False))


if __name__ == "__main__":
    main()
