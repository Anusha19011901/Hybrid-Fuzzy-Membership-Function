"""Copy the key figures into figures/portfolio/ at web-friendly sizes (max width 1600 px)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "figures", ROOT / "figures" / "portfolio"
PICKS = {
    "04_membership_overview.png": "01_membership_functions.png",
    "05_rules.png": "02_rule_base.png",
    "05_explain_patient_complete.png": "03_patient_explanation.png",
    "06_test_ci.png": "04_test_ci_comparison.png",
    "06_per_site_all.png": "05_per_hospital.png",
}


def main(max_width: int = 1600) -> None:
    DST.mkdir(exist_ok=True)
    for s, d in PICKS.items():
        im = Image.open(SRC / s).convert("RGB")
        w, h = im.size
        if w > max_width:
            im = im.resize((max_width, round(h * max_width / w)), Image.LANCZOS)
        im.save(DST / d, optimize=True)
        print(f"{d:32s} {im.size[0]}x{im.size[1]}  {(DST / d).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1600)
