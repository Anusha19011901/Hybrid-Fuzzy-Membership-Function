# Portfolio figures

Web-friendly copies (max width 1600 px) of the key figures. Regenerate with `scripts/phase4_membership.py`, `phase5_fuzzy_inference.py` and `phase6_final.py`, then `python scripts/make_portfolio_figures.py`.

- `01_membership_functions.png` (1600×919, 287 KB): Hybrid membership functions (max of trapezoid and Gaussian) for the four continuous attributes, fitted on training data in clinical units.
- `02_rule_base.png` (1600×1628, 466 KB): The learned rule base: 31 single-antecedent rules with their class consequents and support.
- `03_patient_explanation.png` (1600×1027, 328 KB): One patient explained end to end: fired terms, per-attribute contributions, combined scores, output sets, centroid and decision.
- `04_test_ci_comparison.png` (1600×515, 125 KB): Held-out test macro F1 and accuracy with bootstrap 95% CIs for every model; diamonds are 5-fold CV means.
- `05_per_hospital.png` (1343×890, 93 KB): Per-hospital diagnostic (out-of-fold macro F1) for every model; the hospital is never a feature.
