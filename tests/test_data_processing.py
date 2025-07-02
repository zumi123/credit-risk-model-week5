# tests/test_data_processing.py
"""Unit tests for helper functions used in the preprocessing pipeline.

To run:
    pytest tests/test_data_processing.py
"""

import pandas as pd

from src.data_processing import DateTimeFeatures
from src.proxy_target import compute_rfm


def test_datetime_feature_extraction():
    """DateTimeFeatures should extract hour, day, month, year correctly."""
    df = pd.DataFrame({
        "TransactionStartTime": ["2025-01-02T12:34:00"]
    })
    tf = DateTimeFeatures()
    out = tf.fit_transform(df)

    # Required columns exist
    expected_cols = {"tx_hour", "tx_day", "tx_month", "tx_year"}
    assert expected_cols.issubset(out.columns)

    # Values are correct
    assert out.loc[0, "tx_hour"] == 12
    assert out.loc[0, "tx_day"] == 2
    assert out.loc[0, "tx_month"] == 1
    assert out.loc[0, "tx_year"] == 2025


def test_compute_rfm_basic():
    """compute_rfm should return correct R, F, M values for simple input."""
    data = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionStartTime": ["2025-01-01", "2025-01-05", "2025-01-03"],
        "TransactionId": [101, 102, 103],
        "Amount": [100, 200, 50],
    })

    snapshot_date = pd.to_datetime("2025-01-06")
    rfm = compute_rfm(data, snapshot_date=snapshot_date)

    # Customer 1: two transactions, total 300, last on 2025‑01‑05 → recency 1 day.
    cust1 = rfm.loc[rfm.CustomerId == 1].iloc[0]
    assert cust1["frequency"] == 2
    assert cust1["monetary"] == 300
    assert cust1["recency"] == 1

    # Customer 2: one transaction, total 50, last on 2025‑01‑03 → recency 3 days.
    cust2 = rfm.loc[rfm.CustomerId == 2].iloc[0]
    assert cust2["frequency"] == 1
    assert cust2["monetary"] == 50
    assert cust2["recency"] == 3
