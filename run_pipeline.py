"""Run the full Phase 1 pipeline: Notebooks 01–04 (executed in place) and the PDF report.

    python run_pipeline.py            # everything (needs MIND small in data/raw/)
    python run_pipeline.py --from 4   # only final charts + PDF (needs outputs/results/)

Executed notebooks keep their outputs so reviewers can see every printed number.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ["01_big_picture", "02_narrowing", "03_relevance_diversity", "04_final_charts"]


def run(cmd: list[str], cwd: Path) -> None:
    t0 = time.perf_counter()
    print("→", " ".join(cmd[-1:]), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)
    print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 1
    for i, name in enumerate(NOTEBOOKS, start=1):
        if i < start:
            continue
        run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=7200", f"{name}.ipynb"], ROOT / "notebooks")
    run([sys.executable, "report/phase1/build_report.py"], ROOT)
    print("\nPipeline complete. PDF: report/phase1/behind-the-feed-phase1.pdf")


if __name__ == "__main__":
    main()
