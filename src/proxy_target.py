# src/proxy_target.py
"""Task 4 – Proxy Target Variable Engineering

Creates a binary column ``is_high_risk`` that acts as a proxy for credit‑default
when no true default label exists.

Workflow
--------
1. **Compute RFM** per ``CustomerId``.
   * *Recency*  = days since the customer’s last transaction relative to a
     snapshot date.
   * *Frequency* = count of transactions.
   * *Monetary*  = total absolute amount spent.
2. **Scale RFM** with ``StandardScaler`` so that each dimension contributes
equally to K‑Means distance computations.
3. **Cluster** customers into *three* segments using ``KMeans`` with
   ``random_state=42`` for reproducibility.
4. **Label high‑risk cluster**  = the cluster with the *highest* mean Recency
   **and** the *lowest* mean Frequency × Monetary product (least engaged).
5. **Merge** the resulting ``is_high_risk`` flag back into the original
   transaction‑level DataFrame so it is ready for modelling.

The main entry point is :pyfunc:`add_is_high_risk`, which wraps all steps.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# ----------------------------------------------------------------------
# 1. RFM computation
# ----------------------------------------------------------------------

def compute_rfm(df: pd.DataFrame, *, snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Return an RFM table indexed by *CustomerId*.

    Parameters
    ----------
    df : DataFrame
        Raw transaction data containing at least ``CustomerId``,
        ``TransactionStartTime`` and ``Amount``.
    snapshot_date : pandas.Timestamp, optional
        Cut‑off date for Recency calculation.  Defaults to *one day after* the
        latest ``TransactionStartTime`` in *df*, ensuring Recency ≥ 1.
    """
    tx_time_col = "TransactionStartTime"

    df = df.copy()
    df[tx_time_col] = pd.to_datetime(df[tx_time_col])

    if snapshot_date is None:
        snapshot_date = df[tx_time_col].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    rfm = (
        df.assign(Value=df["Amount"].abs())
        .groupby("CustomerId")
        .agg(
            recency=(tx_time_col, lambda x: (snapshot_date - x.max()).days),
            frequency=("TransactionId", "count"),
            monetary=("Value", "sum"),
        )
        .reset_index()
    )
    return rfm

# ----------------------------------------------------------------------
# 2.–4. Scaling, clustering, labelling
# ----------------------------------------------------------------------

def _cluster_rfm(rfm: pd.DataFrame, *, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Cluster RFM features and return *rfm* with ``cluster`` column added."""
    features = rfm[["recency", "frequency", "monetary"]].values
    scaled = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = km.fit_predict(scaled)
    rfm = rfm.copy()
    rfm["cluster"] = labels
    return rfm

def _identify_high_risk_cluster(rfm: pd.DataFrame) -> int:
    """Determine which cluster is the least engaged / highest risk.

    Criterion
    ---------
    * Highest mean Recency (longest time since last activity)
    * Lowest engagement score where engagement = Frequency × Monetary
    """
    prof = (
        rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .assign(engagement=lambda d: d["frequency"] * d["monetary"])
    )

    # Rank by Recency descending & engagement ascending
    prof["risk_rank"] = prof["recency"].rank(ascending=False, method="min") + prof[
        "engagement"
    ].rank(ascending=True, method="min")

    high_risk_cluster = prof["risk_rank"].idxmax()
    return int(high_risk_cluster)

# ----------------------------------------------------------------------
# 5. Public helper
# ----------------------------------------------------------------------

def add_is_high_risk(
    df: pd.DataFrame,
    *,
    snapshot_date: Optional[str | datetime] = None,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """Return *df* with an additional binary column ``is_high_risk``.

    This function is idempotent – if ``is_high_risk`` already exists it will be
    overwritten.
    """
    rfm = compute_rfm(df, snapshot_date=snapshot_date)
    rfm = _cluster_rfm(rfm, n_clusters=n_clusters)

    high_risk_cluster = _identify_high_risk_cluster(rfm)
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    # Merge back to transactional level
    result = df.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")
    return result

# ----------------------------------------------------------------------
# CLI demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Create proxy high‑risk label.")
    p.add_argument("--input", default="data/raw/data.csv", help="Path to raw CSV")
    p.add_argument("--output", default="data/processed/data_with_target.csv", help="Where to save")
    p.add_argument("--snapshot-date", default=None, help="YYYY‑MM‑DD for Recency calculation")
    args = p.parse_args()

    df_raw = pd.read_csv(args.input)
    df_labeled = add_is_high_risk(df_raw, snapshot_date=args.snapshot_date)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(args.output, index=False)
    print(f"Wrote labeled data → {args.output}")
