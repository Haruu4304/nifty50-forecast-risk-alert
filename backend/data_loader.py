"""
data_loader.py  (BACKEND)
-------------------------
Responsible for getting NIFTY 50 data and cleaning it.

Data structure used: pandas DataFrame with a DatetimeIndex and columns
Open, High, Low, Close, Volume (+ Return, Outlier after cleaning).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICKER = "^NSEI"  # NIFTY 50 on Yahoo Finance


def fetch_nifty(start, end, ticker: str = TICKER) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance using yfinance."""
    import yfinance as yf

    end_inclusive = pd.to_datetime(end) + pd.Timedelta(days=1)  # yfinance 'end' is exclusive
    df = yf.download(
        ticker,
        start=pd.to_datetime(start),
        end=end_inclusive,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(
            "No data returned from Yahoo Finance. Check your internet connection "
            "or the selected date range."
        )
    # Newer yfinance versions return MultiIndex columns -> flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def load_csv(file) -> pd.DataFrame:
    """Load a user-uploaded CSV. Must contain at least 'Date' and 'Close' columns."""
    raw = pd.read_csv(file)
    lookup = {c.lower().strip(): c for c in raw.columns}
    if "date" not in lookup or "close" not in lookup:
        raise ValueError("CSV must contain 'Date' and 'Close' columns.")

    rename = {lookup["date"]: "Date", lookup["close"]: "Close"}
    for name in ("open", "high", "low", "volume"):
        if name in lookup:
            rename[lookup[name]] = name.capitalize()
    df = raw.rename(columns=rename)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(df: pd.DataFrame, outlier_z: float = 3.0):
    """
    Clean the raw data and add log returns.

    Steps: remove duplicate dates -> sort -> drop missing/invalid Close ->
    forward-fill other gaps -> compute log returns -> FLAG (not remove) outliers.
    Outliers are kept because market crashes are genuine events.

    Returns (clean_df, report_dict)
    """
    df = df.copy()
    report = {"Raw rows": len(df)}

    report["Duplicate dates removed"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")].sort_index()

    report["Missing Close values dropped"] = int(df["Close"].isna().sum())
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    others = df.drop(columns=["Close"])
    report["Other missing values forward-filled"] = int(others.isna().sum().sum())
    df = df.ffill()

    df["Return"] = np.log(df["Close"]).diff()
    df = df.dropna(subset=["Return"])

    z = (df["Return"] - df["Return"].mean()) / df["Return"].std()
    df["Outlier"] = z.abs() > outlier_z
    report[f"Outlier days flagged (|z| > {outlier_z:g})"] = int(df["Outlier"].sum())
    report["Clean rows"] = len(df)
    return df, report
