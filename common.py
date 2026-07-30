"""
Shared utilities for all experiments.

Coverage Asymmetry, Not Sample Size — reproducibility package.
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

EPS = 1e-9
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (DATA, RESULTS, FIGURES):
    _p.mkdir(exist_ok=True)

# Groups in the DoH dataset
TOOLS = ["dns2tcp", "dnscat2", "iodin"]
BROWSERS = ["chrome", "firefox"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_doh(path: str | Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load the de-duplicated DoH exfiltration dataset."""
    path = Path(path) if path else DATA / "doh_exfil_dedup.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run  python src/data/build_doh.py  first."
        )
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c not in ("y", "grp")]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    return df, feats


def load_dapt_recon(path: str | Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load the DAPT 2020 reconnaissance subset."""
    path = Path(path) if path else DATA / "dapt_recon.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run  python src/data/build_dapt_recon.py  first."
        )
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c not in ("y", "day")]
    return df, feats


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_models(seed: int) -> dict:
    """The six classifier families used in Section 10."""
    models = {
        "HistGB": HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.15, max_bins=32,
            random_state=seed, early_stopping=False, class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=80, min_samples_leaf=3, n_jobs=1,
            random_state=seed, class_weight="balanced_subsample",
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=80, min_samples_leaf=3, n_jobs=1,
            random_state=seed, class_weight="balanced_subsample",
        ),
        "LogReg": make_pipeline(
            QuantileTransformer(output_distribution="normal", n_quantiles=300,
                                random_state=seed),
            LogisticRegression(max_iter=300, class_weight="balanced"),
        ),
        "MLP": make_pipeline(
            QuantileTransformer(output_distribution="normal", n_quantiles=300,
                                random_state=seed),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, random_state=seed),
        ),
    }
    try:                                        # optional dependency
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=120, max_depth=6, learning_rate=0.2, tree_method="hist",
            random_state=seed, n_jobs=1, eval_metric="logloss",
        )
    except ImportError:
        pass
    return models


def default_model(seed: int = 0):
    return make_models(seed)["HistGB"]


# ---------------------------------------------------------------------------
# Fitting and scoring
# ---------------------------------------------------------------------------
def fit_score(df, feats, tr, te, model=None, seed: int = 0) -> dict:
    """
    Fit on `tr`, score on `te`.

    Imputation medians are estimated on the TRAINING indices only — this is the
    single most common source of leakage in the literature and is enforced here.
    """
    y = df["y"].to_numpy()
    med = df.iloc[tr][feats].median()
    X = df[feats].fillna(med).to_numpy(np.float32)
    m = model if model is not None else default_model(seed)
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    yp = (p >= 0.5).astype(int)
    return dict(
        roc=roc_auc_score(y[te], p),
        pr=average_precision_score(y[te], p),
        f1=f1_score(y[te], yp),
        precision=precision_score(y[te], yp, zero_division=0),
        recall=recall_score(y[te], yp),
    )


# ---------------------------------------------------------------------------
# Coverage (Equation 9)
# ---------------------------------------------------------------------------
def top_dispersed(a: pd.DataFrame, b: pd.DataFrame, cols, k: int = 6) -> list[str]:
    """The k features whose 5–95 percentile spread differs most between a and b."""
    ratios = {}
    for c in cols:
        sa = a[c].quantile(0.95) - a[c].quantile(0.05)
        sb = b[c].quantile(0.95) - b[c].quantile(0.05)
        if sa > 0 and sb > 0:
            ratios[c] = max(sa / sb, sb / sa)
    return [c for c, _ in sorted(ratios.items(), key=lambda x: -x[1])[:k]]


def containment(x: pd.DataFrame, y: pd.DataFrame, cols,
                lo: float = 0.01, hi: float = 0.99) -> float:
    """
    C(X -> Y): fraction of X's rows inside Y's central operating region.

    Implements Equation 9 of the paper.
    """
    l = y[cols].quantile(lo)
    h = y[cols].quantile(hi)
    return float(((x[cols] >= l) & (x[cols] <= h)).all(axis=1).mean())


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def p3_splits(df):
    """
    Protocol P3: leave-one-tool-and-one-client-out.

    Yields (tool, browser, train_idx, test_idx) for all 3 x 2 combinations.
    """
    for tool in TOOLS:
        for br in BROWSERS:
            other = [b for b in BROWSERS if b != br][0]
            tr = np.flatnonzero(
                df.grp.isin([t for t in TOOLS if t != tool]) | (df.grp == other)
            )
            te = np.flatnonzero((df.grp == tool) | (df.grp == br))
            yield tool, br, tr, te


def subsample(idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic subsample, used to keep runtimes tractable."""
    if n >= len(idx):
        return idx
    return np.random.RandomState(seed).choice(idx, n, replace=False)


def summarise(rows, by, metric="roc") -> pd.DataFrame:
    """mean ± std of `metric` grouped by `by`."""
    r = pd.DataFrame(rows)
    g = r.groupby(by)[metric]
    return pd.DataFrame({"mean": g.mean().round(4), "std": g.std().round(4),
                         "n": g.size()})
