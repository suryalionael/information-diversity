# Data

**The MIND dataset is not committed to Git** (size and Microsoft Research License terms).
Everything under `data/raw/` and `data/interim/` is git-ignored.

## Source

Official page: https://msnews.github.io/ — download only from there (accept the Microsoft Research
License Terms on the page). Do not use unofficial mirrors.

## Required files (MIND small)

| Download | Unzip into |
|---|---|
| `MINDsmall_train.zip` | `data/raw/MINDsmall_train/` |
| `MINDsmall_dev.zip`   | `data/raw/MINDsmall_dev/`   |

Each zip contains `news.tsv`, `behaviors.tsv`, `entity_embedding.vec`, `relation_embedding.vec`.
Only `news.tsv` and `behaviors.tsv` are required.

Expected layout:

```text
data/raw/
├── MINDsmall_train/
│   ├── behaviors.tsv
│   └── news.tsv
└── MINDsmall_dev/
    ├── behaviors.tsv
    └── news.tsv
```

From the repository root:

```bash
mkdir -p data/raw
unzip MINDsmall_train.zip -d data/raw/MINDsmall_train
unzip MINDsmall_dev.zip   -d data/raw/MINDsmall_dev
```

To keep the data elsewhere, set `MIND_DATA_DIR` to the folder that contains `MINDsmall_train/` and
`MINDsmall_dev/`.

## Derived data

`data/interim/` holds parquet caches of parsed impression logs created by `src/data.py`. Safe to delete;
it is regenerated on the next run.
