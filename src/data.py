"""Loading, parsing and validation of the MIND (Microsoft News Dataset) files.

File formats follow the official MIND documentation (https://msnews.github.io/):

``news.tsv`` (tab-separated, no header)
    news_id, category, subcategory, title, abstract, url,
    title_entities, abstract_entities

``behaviors.tsv`` (tab-separated, no header)
    impression_id, user_id, time, history, impressions

    * ``history``     space-separated news IDs the user clicked *before* this impression
    * ``impressions`` space-separated ``<news_id>-<label>`` tokens, label 1 = clicked

Every function here is deterministic and side-effect free except the parquet
caching helpers, which only write to ``data/interim/`` (git-ignored).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

NEWS_COLUMNS = [
    "news_id", "category", "subcategory", "title", "abstract", "url",
    "title_entities", "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"  # e.g. "11/11/2019 9:05:58 AM"

SPLIT_DIRS = {"train": "MINDsmall_train", "dev": "MINDsmall_dev"}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def data_dir() -> Path:
    """Directory holding the unzipped MIND folders.

    Defaults to ``<repo>/data/raw``; override with the ``MIND_DATA_DIR`` env var.
    """
    return Path(os.environ.get("MIND_DATA_DIR", REPO_ROOT / "data" / "raw"))


def output_dir() -> Path:
    """Root for saved results/figures (``<repo>/outputs``; env ``MIND_OUTPUT_DIR``)."""
    return Path(os.environ.get("MIND_OUTPUT_DIR", REPO_ROOT / "outputs"))


def split_path(split: str, filename: str) -> Path:
    if split not in SPLIT_DIRS:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_DIRS)}")
    return data_dir() / SPLIT_DIRS[split] / filename


def check_files(splits: Iterable[str] = ("train", "dev")) -> pd.DataFrame:
    """Report which required files exist, with sizes. Raises if any are missing."""
    rows = []
    for split in splits:
        for fname in ("news.tsv", "behaviors.tsv"):
            p = split_path(split, fname)
            rows.append({
                "split": split, "file": fname, "path": str(p), "exists": p.exists(),
                "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None,
            })
    report = pd.DataFrame(rows)
    missing = report.loc[~report["exists"], "path"].tolist()
    if missing:
        raise FileNotFoundError(
            "Missing MIND files (see data/README.md for download instructions):\n  "
            + "\n  ".join(missing)
        )
    return report


# --------------------------------------------------------------------------- #
# Raw loaders
# --------------------------------------------------------------------------- #
def _read_tsv(path: Path, columns: list[str]) -> pd.DataFrame:
    # QUOTE_NONE: titles/abstracts contain unbalanced quote characters.
    return pd.read_csv(
        path, sep="\t", header=None, names=columns, quoting=csv.QUOTE_NONE,
        dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8",
    )


def load_news(split: str) -> pd.DataFrame:
    """Load ``news.tsv`` for one split, with a ``split`` column added."""
    df = _read_tsv(split_path(split, "news.tsv"), NEWS_COLUMNS)
    df["split"] = split
    return df


def load_behaviors(split: str, parse_time: bool = True) -> pd.DataFrame:
    """Load ``behaviors.tsv`` for one split.

    Impression IDs restart in every split, so ``imp_key`` = ``"<split>-<impression_id>"``
    is added as a globally unique impression identifier.
    """
    df = _read_tsv(split_path(split, "behaviors.tsv"), BEHAVIOR_COLUMNS)
    df["impression_id"] = df["impression_id"].astype("int64")
    if parse_time:
        df["time"] = pd.to_datetime(df["time"], format=TIME_FORMAT)
    df["split"] = split
    df["imp_key"] = split + "-" + df["impression_id"].astype(str)
    return df


def load_catalogue(splits: Iterable[str] = ("train", "dev")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union of ``news.tsv`` across splits, one row per ``news_id``.

    Returns
    -------
    catalogue : DataFrame
        One row per unique news_id (first occurrence kept) plus ``in_splits``
        listing the splits whose news.tsv contains the article.
    conflicts : DataFrame
        news_ids whose category/subcategory/title differ between splits
        (expected to be empty; returned so the caller can document it).
    """
    news = pd.concat([load_news(s) for s in splits], ignore_index=True)
    in_splits = news.groupby("news_id")["split"].agg(lambda s: ",".join(sorted(set(s))))
    meta_cols = ["category", "subcategory", "title"]
    n_variants = news.drop_duplicates(["news_id"] + meta_cols).groupby("news_id").size()
    conflict_ids = n_variants[n_variants > 1].index
    conflicts = news[news["news_id"].isin(conflict_ids)].sort_values("news_id")
    catalogue = news.drop_duplicates("news_id", keep="first").drop(columns="split")
    catalogue = catalogue.merge(in_splits.rename("in_splits"), on="news_id", how="left")
    return catalogue.reset_index(drop=True), conflicts


# --------------------------------------------------------------------------- #
# Parsing impression and history lists into long format
# --------------------------------------------------------------------------- #
def explode_impressions(behaviors: pd.DataFrame) -> pd.DataFrame:
    """One row per (impression, shown article) — the *exposure records*.

    Columns: imp_key, split, user_id, time, news_id, clicked (int8), position.

    ``position`` is the article's index within the logged impression list. MIND
    does not document this order as the on-screen rank, so it must not be
    interpreted as display position.
    """
    base = behaviors[["imp_key", "split", "user_id", "time", "impressions"]].copy()
    base["token"] = base["impressions"].fillna("").str.split(" ")
    base = base.drop(columns="impressions").explode("token", ignore_index=True)
    base = base[base["token"].notna() & (base["token"] != "")]
    base["position"] = base.groupby("imp_key", sort=False).cumcount().astype("int32")
    parts = base["token"].str.rsplit("-", n=1, expand=True)
    base["news_id"] = parts[0]
    base["clicked"] = pd.to_numeric(parts[1], errors="coerce").astype("Int8")
    return base.drop(columns="token").reset_index(drop=True)


def explode_history(behaviors: pd.DataFrame, dedupe_per_user: bool = True) -> pd.DataFrame:
    """One row per (user, historically clicked article), with ``hist_pos`` order.

    If ``dedupe_per_user`` the history string is taken once per user (first
    impression); validation in Notebook 01 checks whether histories actually
    vary across a user's impressions.
    """
    cols = ["user_id", "split", "history"]
    base = behaviors[cols]
    if dedupe_per_user:
        base = base.drop_duplicates("user_id", keep="first")
    base = base.assign(news_id=base["history"].fillna("").str.split(" ")).drop(columns="history")
    base = base.explode("news_id", ignore_index=True)
    base = base[base["news_id"].notna() & (base["news_id"] != "")]
    base["hist_pos"] = base.groupby("user_id", sort=False).cumcount().astype("int32")
    return base.reset_index(drop=True)


def load_exposures(splits: Iterable[str] = ("train", "dev"), use_cache: bool = True) -> pd.DataFrame:
    """Exposure records for the given splits, cached as parquet in ``data/interim``."""
    splits = list(splits)
    cache = REPO_ROOT / "data" / "interim" / f"exposures_{'_'.join(splits)}.parquet"
    if use_cache and cache.exists() and "MIND_DATA_DIR" not in os.environ:
        return pd.read_parquet(cache)
    exp = pd.concat([explode_impressions(load_behaviors(s)) for s in splits], ignore_index=True)
    if use_cache and "MIND_DATA_DIR" not in os.environ:
        cache.parent.mkdir(parents=True, exist_ok=True)
        exp.to_parquet(cache, index=False)
    return exp


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_news(news: pd.DataFrame) -> dict:
    """Schema / quality checks for one split's news table."""
    return {
        "rows": len(news),
        "columns_ok": list(news.columns[: len(NEWS_COLUMNS)]) == NEWS_COLUMNS,
        "unique_news_ids": int(news["news_id"].nunique()),
        "duplicate_news_ids": int(news["news_id"].duplicated().sum()),
        "missing_by_column": news[NEWS_COLUMNS].isna().sum().to_dict(),
        "n_categories": int(news["category"].nunique()),
        "n_subcategories": int(news["subcategory"].nunique()),
        "bad_id_format": int((~news["news_id"].str.fullmatch(r"N\d+")).sum()),
    }


def validate_behaviors(behaviors: pd.DataFrame, exposures: pd.DataFrame) -> dict:
    """Schema / quality checks for one split's behaviour log and its exposure records."""
    imp_sizes = exposures.groupby("imp_key").size()
    clicks_per_imp = exposures.groupby("imp_key")["clicked"].sum()
    return {
        "rows": len(behaviors),
        "columns_ok": list(behaviors.columns[: len(BEHAVIOR_COLUMNS)]) == BEHAVIOR_COLUMNS,
        "duplicate_impression_ids": int(behaviors["impression_id"].duplicated().sum()),
        "unique_users": int(behaviors["user_id"].nunique()),
        "missing_by_column": behaviors[BEHAVIOR_COLUMNS].isna().sum().to_dict(),
        "time_min": str(behaviors["time"].min()),
        "time_max": str(behaviors["time"].max()),
        "unparseable_labels": int(exposures["clicked"].isna().sum()),
        "labels_not_0_or_1": int((~exposures["clicked"].isin([0, 1])).sum()),
        "exposure_records": len(exposures),
        "impressions_with_no_articles": int(len(behaviors) - imp_sizes.size),
        "articles_per_impression": imp_sizes.describe().round(2).to_dict(),
        "impressions_with_zero_clicks": int((clicks_per_imp == 0).sum()),
        "impressions_with_2plus_clicks": int((clicks_per_imp >= 2).sum()),
        "duplicate_article_within_impression": int(
            exposures.duplicated(["imp_key", "news_id"]).sum()
        ),
    }


def join_coverage(ids: pd.Series, catalogue_ids: Iterable[str]) -> dict:
    """How many of ``ids`` (with repeats) and unique ids are found in the catalogue."""
    cat = pd.Index(pd.unique(pd.Series(list(catalogue_ids))))
    found = ids.isin(cat)
    uniq = pd.Series(ids.unique())
    return {
        "records": int(len(ids)),
        "records_unmatched": int((~found).sum()),
        "unique_ids": int(len(uniq)),
        "unique_ids_unmatched": int((~uniq.isin(cat)).sum()),
    }


def history_stability(behaviors: pd.DataFrame) -> dict:
    """Does a user's ``history`` string change across their impressions?

    This determines whether a within-dataset temporal design (earlier vs later
    history) is possible for Notebook 02.
    """
    per_user = behaviors.groupby("user_id")["history"].nunique(dropna=False)
    multi = behaviors.groupby("user_id").size()
    multi_users = multi[multi > 1].index
    return {
        "users": int(per_user.size),
        "users_with_2plus_impressions": int(len(multi_users)),
        "users_with_varying_history": int((per_user.loc[multi_users] > 1).sum()),
    }
