"""
eda.py  (BACKEND)
-----------------
Exploratory analysis helpers: decomposition, stationarity tests, ACF/PACF plots.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


def decompose(close: pd.Series, period: int = 252):
    """Multiplicative decomposition into trend, seasonal and residual parts.
    252 ~ number of trading days in a year. Falls back to a shorter period for short series."""
    if len(close) < 2 * period:
        period = max(5, len(close) // 3)
    return seasonal_decompose(close, model="multiplicative", period=period), period


def stationarity_report(series: pd.Series, name: str) -> dict:
    """ADF (H0: non-stationary) and KPSS (H0: stationary) tests with a verdict."""
    s = series.dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_p = kpss(s, regression="c", nlags="auto")[1]

    adf_ok = adf_p < 0.05      # reject H0 -> stationary
    kpss_ok = kpss_p > 0.05    # fail to reject H0 -> stationary
    if adf_ok and kpss_ok:
        verdict = "Stationary"
    elif (not adf_ok) and (not kpss_ok):
        verdict = "Non-stationary"
    else:
        verdict = "Inconclusive (tests disagree)"
    return {
        "Series": name,
        "ADF p-value": round(float(adf_p), 4),
        "KPSS p-value": round(float(kpss_p), 4),
        "Verdict": verdict,
    }


def acf_pacf_figure(series: pd.Series, title: str, lags: int = 40, colors: dict | None = None):
    """Matplotlib ACF + PACF side by side, styled for the dark theme."""
    c = colors or {}
    bg, fg, line = c.get("card", "#131C2E"), c.get("text", "#E5E7EB"), c.get("primary", "#14B8A6")
    s = series.dropna()
    lags = min(lags, len(s) // 2 - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), facecolor=bg)
    plot_acf(s, lags=lags, ax=axes[0], color=line, vlines_kwargs={"colors": line})
    plot_pacf(s, lags=lags, ax=axes[1], color=line, vlines_kwargs={"colors": line}, method="ywm")
    axes[0].set_title(f"ACF: {title}")
    axes[1].set_title(f"PACF: {title}")
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.title.set_color(fg)
        for sp in ax.spines.values():
            sp.set_color("#2A3654")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig
