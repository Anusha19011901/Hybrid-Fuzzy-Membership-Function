# Hybrid Fuzzy Membership Model for Cardiac State Prediction

A clean, open reimplementation of the method described in **Indian Patent No. 601763**,
*System and Method for Predicting a State of a Heart Using a Hybrid Fuzzy Membership Model*
(Application 202441004228, filed 19 January 2024, granted 8 September 2026; applicant Vellore
Institute of Technology; inventors Siva Shanmugam G, Anbarasi M, Anusha Seshadri, Iyashi Pal).

The original research code was not preserved, so this repository rebuilds the pipeline from the
granted specification, on the public UCI Heart Disease dataset, with every design decision written
down. It is a faithful reimplementation of the patented *method*, not a copy of the original
experiments, and the numbers it produces are its own.

**Tags:** Healthcare AI · Fuzzy Logic · Explainable AI · Feature Selection · Metaheuristics · Python

## The method in one paragraph

Patient records are cleaned (mean imputation, scaling, outlier handling), a subset of attributes is
chosen with **Binary Brain Storm Optimization** (a population metaheuristic that maximises
cross-validated F1), each selected attribute is fuzzified with a **hybrid membership function**
that takes the element-wise maximum of a trapezoid and a Gaussian, a rule base maps the fuzzified
inputs to a risk output through weighted-average aggregation, and **centroid defuzzification** turns
the fuzzy output into one crisp, explainable risk value. The design goal is a predictor that
handles the imprecision of clinical data while remaining readable by a clinician.

## Build status

| Phase | Content | Status |
|---|---|---|
| 1 | Data loading, risk labels, leakage-safe preprocessing | done |
| 2 | Baselines: logistic regression, random forest, small MLP, majority-class floor | done |
| 3 | Binary Brain Storm Optimization for feature selection | done |
| 4 | Hybrid trapezoidal + Gaussian membership functions, data-derived | planned |
| 5 | Fuzzy inference: Wang–Mendel rules, weighted average, centroid | planned |
| 6 | Stratified CV evaluation vs. baselines, explainability plots | planned |

## Repository layout

```
data/                heart_disease_uci.csv (raw); processed/ is regenerated and git-ignored
figures/             all figures, numbered by phase
notebooks/           one walkthrough notebook per phase
results/             metric tables (CSV), numbered by phase
scripts/             one runnable script per phase
src/hfm/             the library: config, data, preprocess, models, evaluation, bbso, plotting
tests/               pytest suite
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # makes `import hfm` work everywhere
pytest                                # 28 tests
python scripts/phase1_preprocess.py   # writes data/processed/ and figures/01_*.png
python scripts/phase2_baselines.py    # writes results/02_*.csv and figures/02_*.png
python scripts/phase3_bbso.py         # ~90 s; writes results/03_* and figures/03_*.png
```

## Data

The patent does not name its dataset, but its sample counts (919 rows for the fuzzy model and a
184-row test set for the deep-learning baseline) match the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Janosi, Steinbrunn, Pfisterer, Detrano): 920 patients from Cleveland, Hungary, Switzerland and
VA Long Beach, 13 clinical attributes, and a target `num` from 0 (no significant narrowing) to 4.
The CSV in `data/` is the Kaggle mirror of that dataset.

Two facts about this data shape everything downstream:

- **Zeros that mean missing.** 172 rows have a serum cholesterol of 0 and one has a resting blood
  pressure of 0. These are impossible measurements and are treated as missing before imputation.
- **Missingness follows the hospital.** Switzerland recorded no cholesterol; Hungary and VA
  largely skipped the fluoroscopy (`ca`, 66% missing overall) and thalassemia (`thal`, 53%) tests.

![missingness](figures/01_missingness_by_site.png)

### Risk labels

The patent reports three classes (Low / Moderate / High) but does not say how they were derived.
This project maps the 0–4 target as follows and keeps the alternative available in
`hfm.config.RISK_MAPPINGS`:

| mapping | Low | Moderate | High | sizes |
|---|---|---|---|---|
| `default` (used) | `num = 0` | `num = 1` | `num = 2, 3, 4` | 411 / 265 / 244 |
| `severe_only` | `num = 0` | `num = 1, 2` | `num = 3, 4` | 411 / 374 / 135 |

`default` separates no significant disease, single-vessel disease and multi-vessel disease, and
keeps enough High-risk patients for stratified cross-validation. Neither mapping reproduces the
class supports printed in the patent (450 / 355 / 114), which cannot come from any monotone
grouping of `num`; that discrepancy is documented rather than reverse-engineered.

![target](figures/01_target_distribution.png)

The source hospital is a strong proxy for the label (Hungary has no High-risk patients at all) and
is deliberately **not** used as a feature.

### Preprocessing

Implemented as a scikit-learn `ColumnTransformer` in `src/hfm/preprocess.py`. The stratified
80/20 split (seed 42) is drawn first, and every statistic is fitted on the training rows only.

| branch | attributes | steps |
|---|---|---|
| continuous | age, trestbps, chol, thalch, oldpeak | IQR clip → mean impute → scale |
| discrete | ca | mean impute → scale |
| categorical | sex, cp, fbs, restecg, exang, slope, thal | mode impute → one-hot |

Where this departs from the patent's wording, and why:

1. Outliers are **clipped to Tukey fences rather than removed**, so that every held-out patient
   stays in the evaluation.
2. Fences are computed on **observed values before imputation**. Clipping after imputation on a
   66%-missing column collapses it onto its mean and erases the feature (this was caught by a
   test). The discrete vessel count is never clipped.
3. Categorical attributes are imputed with the **mode**, since a category has no mean.
4. One scaler at a time (`standard`, `minmax` or `none`); the fuzzy phases will use unscaled
   values so membership functions stay in clinical units.

## Baselines

Every model is evaluated as a `Pipeline(preprocessor, model)` so that imputation means, clip
fences and scaling are refitted inside each fold of a stratified 5-fold CV on the 736 training
rows, and once more on all of them before scoring the 184-row test split. Hyper-parameters are
fixed and untuned; these models are context for the fuzzy model, not contenders.

| model | CV macro F1 (mean ± sd) | CV accuracy | test macro F1 | test accuracy |
|---|---|---|---|---|
| majority class (always *Low*) | 0.21 ± 0.00 | 0.45 | 0.21 | 0.45 |
| logistic regression | 0.62 ± 0.05 | 0.65 | 0.61 | 0.65 |
| random forest (500 trees) | 0.63 ± 0.05 | 0.66 | 0.62 | 0.65 |
| MLP (32, 16), early stopping | 0.61 ± 0.04 | 0.64 | 0.56 | 0.61 |

Full tables, per-class F1 and confusion matrices are in `results/02_*.csv` and
`figures/02_*.png`. Two observations carry forward:

- *Low* is the easy class and *Moderate* the hard one: every model confuses single-vessel
  disease with both of its neighbours.
- The majority-class row is the floor for *this* label setup (accuracy 0.45). The patent's
  reported metrics (accuracy 0.43, weighted F1 0.40) were computed on class supports of
  450 / 355 / 114 that do not match any grouping of `num`, so they come from a different label
  setup and are not directly comparable to the numbers here.

**On the neural baseline.** The patent's deep-learning comparison used a Keras Sequential
network. This project uses scikit-learn's `MLPClassifier` with an equivalent architecture (two
hidden layers of 32 and 16 units, ReLU, Adam, early stopping). A Keras wrapper with the same
interface is kept in `src/hfm/keras_optional.py` but is not part of the default pipeline or the
tests, because TensorFlow and scikit-learn each load their own OpenMP runtime on macOS and the
first Keras training step deadlocks after scikit-learn has trained a model in the same process.

**Per-hospital diagnostic.** The source hospital is never a feature, but
`hfm.evaluation.per_site` reports metrics per site for every model. Switzerland is the hard site
for all baselines (no cholesterol values, almost no fluoroscopy, 93% of patients *Moderate* or
*High*); Hungary is the easy one (no *High* patients at all). Later phases are checked against
this breakdown so that gains on easy sites do not hide losses on the hard one.

![per-site](figures/02_per_site.png)

## Feature selection with Binary Brain Storm Optimization

`src/hfm/bbso.py` implements the binary variant of Brain Storm Optimization: a population of
candidate masks is clustered with k-means, new candidates are generated by perturbing a cluster
centre or a random member, or by merging two clusters, a sigmoid transfer function turns the
perturbed positions into bits, and a candidate replaces its parent when it scores better. The
search stops after 12 iterations without improvement (50 at most).

- The search runs over the **13 clinical attributes**, not the 22 encoded columns; a one-hot
  attribute is kept or dropped as a whole.
- **Fitness** is the macro F1 of logistic regression under an inner 3-fold stratified CV on the
  training split only. Preprocessing is fitted per inner fold and reused for every mask, and
  fitness is cached per mask (shared across seeds).
- Ten seeds, then per-attribute selection frequency and the most frequent best mask.

**A deliberate modification of standard BBSO.** In the textbook binary variant each idea is a
real-valued position and every bit is re-sampled from `sigmoid(position)` whenever the idea is
perturbed. On this problem that re-sampling destroyed good masks faster than the search could
find them: on a planted 13-bit toy objective it reached the optimum in 0 of 10 seeds. This
implementation therefore *anchors* a position to the mask it was scored with (sign of each
coordinate set to its bit, magnitude floored at 1.5, so a child keeps each parent bit with
probability about 0.82 before step noise is added). With anchoring the toy objective is solved
in 10 of 10 seeds. The clustering, generation rules, sigmoid transfer, greedy replacement and
early stopping are otherwise as in the standard algorithm. See `BBSO._evaluate` in `src/hfm/bbso.py`.

Ten seeds take about 80 s on a laptop (5,460 fitness calls, 2,720 unique masks scored); the
subset-versus-all evaluation adds about 8 s. Seven of ten seeds agree on the same ten-attribute
mask; `fbs` is never kept and `sex` and `chol` are kept by two seeds each.

![selection frequency](figures/03_selection_frequency.png)

Two findings, stated plainly:

- **The inner-CV gain does not transfer.** Fitness rose from 0.614 (all 13) to 0.640 (chosen
  subset), but under the outer 5-fold CV and on the test split the subset and the full set are
  within one fold's standard deviation for all three baselines. The gain is mostly selection
  optimism. The value of the subset is a smaller, cleaner input for the fuzzy model, not accuracy.
- **`ca`, `thal` and `slope` are kept by every seed.** These are the attributes that are mostly
  missing outside Cleveland and were filled with the training mode or mean, so after imputation
  they partly encode which hospital a patient came from. No missingness indicators were added;
  the per-hospital diagnostic is the check on whether that signal generalises.

| model | all 13, CV macro F1 | BBSO subset, CV macro F1 | all 13, test | subset, test |
|---|---|---|---|---|
| logistic regression | 0.62 ± 0.04 | 0.62 ± 0.05 | 0.61 | 0.60 |
| random forest | 0.63 ± 0.05 | 0.64 ± 0.05 | 0.62 | 0.59 |
| MLP | 0.61 ± 0.04 | 0.58 ± 0.04 | 0.56 | 0.57 |

## Results

The fuzzy model is not built yet. When it is, its results will be reported with the same
stratified cross-validation, held-out split and per-hospital breakdown as the baselines above.
Because the patent's label setup differs from this one (see the note on class supports), the
patent's reported figures are quoted for reference only and are not used as a benchmark.

## Backlog

Ideas worth running once the full pipeline exists, not yet done:

- **Hospital-linked imputation ablation.** `ca`, `thal` and `slope` are mostly missing outside
  Cleveland and are filled with the training mode or mean, so after imputation they partly encode
  the source hospital. Re-run feature selection, the fuzzy model and the baselines with those
  three attributes dropped and compare the per-hospital breakdown, to measure how much of their
  selection and predictive value rests on that imputation.

## Patent

- Indian Patent No. 601763, publication IN601763 B1, granted 8 September 2026
- IPC: G06N20/10, G06F18/2337, G06F18/26, G16H10/60, G16H50/30, A61B5/02, G06N5/048
- Claims 1–8 (claim 1: method; claim 6: system)

## License

Code in this repository is released under the MIT License. The patent itself is owned by Vellore
Institute of Technology; this reimplementation is published by one of the inventors for research
and educational use.
