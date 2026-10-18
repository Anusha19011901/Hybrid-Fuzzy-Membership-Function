"""Phase 1: load, label, split, preprocess, and describe the dataset.

Writes:
    data/processed/{train,test}_raw.csv        cleaned rows + risk label, pre-imputation
    data/processed/{train,test}_processed.csv  imputed, clipped, scaled, one-hot
    figures/01_target_distribution.png
    figures/01_missingness_by_site.png
    figures/01_risk_by_site.png

Run from the repository root:  python scripts/phase1_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hfm import config, data, plotting  # noqa: E402
from hfm.preprocess import feature_groups, fit_transform_split, fitted_numeric_stats  # noqa: E402


def plot_target_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df[config.RAW_TARGET].value_counts().sort_index()
    mapping = config.RISK_MAPPINGS[config.DEFAULT_RISK_MAPPING]
    colors = [plotting.RISK_COLORS[mapping[k]] for k in counts.index]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.7)
    plotting.bar_labels(ax, bars)
    plotting.tidy(ax)
    ax.set_xlabel("Original target `num`  (0 = no significant narrowing, 1-4 = increasing severity)", fontsize=9)
    ax.set_ylabel("Patients")
    ax.set_title("Target distribution and the Low / Moderate / High mapping")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plotting.RISK_COLORS[r]) for r in config.RISK_LEVELS]
    ax.legend(handles, config.RISK_LEVELS, title="Risk label", loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, path: Path) -> None:
    pct = (
        df.groupby(config.SITE_COL)[config.FEATURE_COLS]
        .apply(lambda g: g.isna().mean() * 100)
        .loc[["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]]
    )
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    im = ax.imshow(pct.values, cmap=plotting.BLUE_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pct.columns)), pct.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pct.index)), pct.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color=plotting.SURFACE if v > 55 else plotting.INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% missing", color=plotting.INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Missing values per attribute and source hospital (after zero -> NaN)")
    fig.savefig(path)
    plt.close(fig)


def plot_risk_by_site(df: pd.DataFrame, path: Path) -> None:
    tab = pd.crosstab(df[config.SITE_COL], df[config.TARGET], normalize="index") * 100
    tab = tab.loc[["Cleveland", "Hungary", "Switzerland", "VA Long Beach"], config.RISK_LEVELS]
    n = df[config.SITE_COL].value_counts()

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    left = np.zeros(len(tab))
    for risk in config.RISK_LEVELS:
        vals = tab[risk].values
        ax.barh(tab.index, vals, left=left, color=plotting.RISK_COLORS[risk],
                label=risk, height=0.62, edgecolor=plotting.SURFACE, linewidth=2)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color=plotting.SURFACE if risk != "Low" else plotting.INK)
        left += vals
    ax.set_yticks(range(len(tab)), [f"{s}  (n={n[s]})" for s in tab.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of patients (%)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_title("Risk label by source hospital")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28), title=None)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    plotting.use_style()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = data.load_dataset()
    train, test = data.make_split(df)
    X_train, y_train = data.split_xy(train)
    X_test, y_test = data.split_xy(test)

    Xtr, Xte, pre = fit_transform_split(X_train, X_test, scaler="standard")

    # ---- persist ---------------------------------------------------------- #
    train.to_csv(config.PROCESSED_DIR / "train_raw.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test_raw.csv", index=False)
    Xtr.assign(**{config.TARGET: y_train.values}).to_csv(
        config.PROCESSED_DIR / "train_processed.csv", index=False)
    Xte.assign(**{config.TARGET: y_test.values}).to_csv(
        config.PROCESSED_DIR / "test_processed.csv", index=False)

    # ---- figures ---------------------------------------------------------- #
    plot_target_distribution(df, config.FIGURES_DIR / "01_target_distribution.png")
    plot_missingness(df, config.FIGURES_DIR / "01_missingness_by_site.png")
    plot_risk_by_site(df, config.FIGURES_DIR / "01_risk_by_site.png")

    # ---- report ----------------------------------------------------------- #
    print(f"rows: {len(df)}  train: {len(train)}  test: {len(test)}")
    print("\nrisk label counts (all / train / test):")
    print(pd.concat([df[config.TARGET].value_counts(), train[config.TARGET].value_counts(),
                     test[config.TARGET].value_counts()], axis=1, keys=["all", "train", "test"]))
    print("\nmissing values per attribute (after zero -> NaN):")
    print(df[config.FEATURE_COLS].isna().sum().to_string())
    print("\ntrain-fitted imputation means and IQR clip fences (observed values only):")
    print(fitted_numeric_stats(pre).round(2).to_string())
    print(f"\nprocessed feature matrix: {Xtr.shape[1]} columns")
    for src, cols in feature_groups(pre).items():
        print(f"  {src:9s} -> {cols}")


if __name__ == "__main__":
    main()
