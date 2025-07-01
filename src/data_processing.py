"""Credit‑Risk Model — Task 3: Feature Engineering

This module builds a scikit‑learn preprocessing pipeline that converts raw
transaction‑level records into model‑ready numerical arrays.

Main steps
----------
1. Date / time features — extracts hour, day, month, year from
   `TransactionStartTime`.
2. Aggregate RFM metrics — optional transformer that groups by
   `CustomerId` and outputs Recency, Frequency, Monetary features.
3. Categorical encoding — one‑hot encodes high‑cardinality but
   informative categories and drops low‑variance columns.
4. Numerical scaling — applies StandardScaler to prevent features with
   large magnitudes dominating the model.

The resulting pipeline is fully reproducible and can be fit / transformed as
part of a larger modelling workflow (see ``src/train.py``).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

# ---------------------------------------------------------------------------
# 1. Custom transformers
# ---------------------------------------------------------------------------

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Extracts time‑based features from a timestamp column.

    Parameters
    ----------
    column : str
        Name of the datetime column.
    drop : bool, default ``True``
        Whether to drop the original column after feature extraction.
    """

    def __init__(self, column: str = "TransactionStartTime", drop: bool = True):
        self.column = column
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N803
        if self.column not in X.columns:
            raise KeyError(f"Column '{self.column}' not found in input data.")
        return self

    def transform(self, X: pd.DataFrame):  # noqa: N803
        X_ = X.copy()
        ts = pd.to_datetime(X_[self.column], errors="coerce")
        X_["tx_hour"] = ts.dt.hour.astype("Int64")
        X_["tx_day"] = ts.dt.day.astype("Int64")
        X_["tx_month"] = ts.dt.month.astype("Int64")
        X_["tx_year"] = ts.dt.year.astype("Int64")
        if self.drop:
            X_ = X_.drop(columns=[self.column])
        return X_

    
    def set_output(self, *, transform=None):
        return self


class RFMFeatures(BaseEstimator, TransformerMixin):
    """Computes Recency, Frequency, Monetary value per CustomerId.

    Recency is computed in days relative to ``snapshot_date``.
    Monetary is the total absolute spend per customer.
    """

    def __init__(self, snapshot_date: pd.Timestamp | None = None):
        self.snapshot_date = snapshot_date or pd.Timestamp.utcnow().normalize()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # noqa: N803
        if "CustomerId" not in X.columns:
            raise KeyError("Input must contain 'CustomerId' column.")
        if "TransactionStartTime" not in X.columns:
            raise KeyError("Input must contain 'TransactionStartTime' column.")
        return self

    def transform(self, X: pd.DataFrame):  # noqa: N803
        X_ = X.copy()
        X_["TransactionStartTime"] = pd.to_datetime(
            X_["TransactionStartTime"], errors="coerce"
        )
        snapshot = self.snapshot_date
        recency = (
            snapshot - X_.groupby("CustomerId")["TransactionStartTime"].max()
        ).dt.days.rename("Recency")
        frequency = (
            X_.groupby("CustomerId")
            .size()
            .astype(int)
            .rename("Frequency")
        )
        monetary = (
            X_.groupby("CustomerId")["Value"].sum().rename("Monetary")
        )

        rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
        return rfm


# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline builder
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(return_dataframe: bool = False) -> Pipeline:
    """Return a fitted preprocessing pipeline.

    Parameters
    ----------
    return_dataframe : bool, default ``False``
        Whether the final output should be a pandas DataFrame (with column
        names) or a NumPy array. ``True`` is useful for debugging / feature
        selection; ``False`` integrates better with scikit‑learn estimators.
    """

    # Columns that are known from the dataset description
    date_cols = ["TransactionStartTime"]
    numeric_cols = ["Amount", "Value"]

    # Drop low‑variance categorical variables (CurrencyCode & CountryCode) later
    categorical_cols = [
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
        "FraudResult",
    ]

    # 1. Datetime extraction pipeline
    datetime_pipe = Pipeline(
        steps=[
            ("extract", DateTimeFeatures(column="TransactionStartTime", drop=True)),
            (
                "impute",
                SimpleImputer(strategy="most_frequent"),
            ),
        ]
    )

    # 2. Numerical pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # 3. Categorical pipeline
    category_pipe = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),

            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("datetime", datetime_pipe, date_cols),
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", category_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Wrap in a top‑level Pipeline so we can later append a classifier/regressor
    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
        ]
    )

    # Optionally force output to DataFrame
    if return_dataframe:
        pipeline.set_output(transform="pandas")
    return pipeline


# ---------------------------------------------------------------------------
# 3. Convenience function for stand‑alone execution
# ---------------------------------------------------------------------------

def run_demo(path: str | Path = "data/raw/data.csv", n_rows: int | None = 10_000):
    """Run a small demo on the raw CSV to verify pipeline output."""
    import sys
    from pathlib import Path as _P

    path = _P(path)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = pd.read_csv(path, nrows=n_rows)
    pipe = build_preprocessing_pipeline(return_dataframe=True)
    X_transformed = pipe.fit_transform(df)
    print("\nTransformed feature matrix shape:", X_transformed.shape)
    print("Sample columns:", X_transformed.columns[:20].tolist())


if __name__ == "__main__":
    run_demo()
