"""Tiny helper for the notebook builder scripts (build_*.py).

Notebooks are generated from these scripts so their logic is diff-able in review.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

SETUP = r'''
import sys, json, warnings
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import data as D, metrics as M, diversity as V, style as S

SEED = 42
OUT = D.output_dir()
RESULTS, FIGURES = OUT / "results", OUT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
S.apply()
pd.set_option("display.width", 170, "display.max_columns", 40, "display.float_format", "{:,.4f}".format)
print("data dir:", D.data_dir()); print("output dir:", OUT)
'''


class Builder:
    def __init__(self) -> None:
        self.cells: list = []

    def md(self, text: str) -> None:
        self.cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(self, text: str) -> None:
        self.cells.append(nbf.v4.new_code_cell(text.strip()))

    def save(self, path: Path) -> None:
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                          "language_info": {"name": "python"}}
        nbf.write(nb, path)
        print("wrote", path)
