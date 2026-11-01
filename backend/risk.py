"""
risk.py  (BACKEND)
------------------
Volatility regime, risk level, anomaly detection and recommendation text.
Uses a GARCH(1,1) model with Student-t errors fitted on the full history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch_full(returns: pd.Series):
    """Fit GARCH on log returns (in %). Returns (fit, annualised cond. vol %, std residuals)."""
    fit = arch_model(returns * 100, vol="GARCH", p=1, q=1, dist="t", mean="Constant").fit(disp="off")
    cond_vol = fit.conditional_volatility * np.sqrt(252)   # annualised %
    return fit, cond_vol, fit.std_resid


def risk_level(cond_vol: pd.Series) -> dict:
    """LOW / MEDIUM / HIGH from the 33rd and 66th percentiles of historical volatility."""
    low_thr, high_thr = np.percentile(cond_vol.dropna(), [33, 66])
    current = float(cond_vol.iloc[-1])
    if current < low_thr:
        level = "LOW"
    elif current < high_thr:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return {"level": level, "current": current, "low_thr": float(low_thr), "high_thr": float(high_thr)}


def detect_anomalies(std_resid: pd.Series, returns: pd.Series, threshold: float) -> pd.DataFrame:
    """Days where the standardised GARCH residual exceeds `threshold` sigma."""
    mask = std_resid.abs() > threshold
    out = pd.DataFrame({
        "Return (%)": (returns[mask] * 100).round(2),
        "Std. residual (sigma)": std_resid[mask].round(2),
    })
    out["Direction"] = np.where(out["Return (%)"] < 0, "Sharp fall", "Sharp rise")
    out.index.name = "Date"
    return out.sort_index(ascending=False)


def recommendation(level: str, current_vol: float, recent_anomalies: int, expected_change_pct=None) -> str:
    """Human-readable decision-support message."""
    base = {
        "LOW": "Market is calm. Normal position sizing is reasonable; swings are likely to be modest.",
        "MEDIUM": "Volatility is moderate. Monitor positions closely and avoid over-leveraging.",
        "HIGH": "Volatility is elevated. Expect large daily swings; consider reducing exposure, "
                "using stop-losses and staggering entries instead of a single large trade.",
    }[level]
    msg = f"{base} Current annualised volatility is {current_vol:.1f}%."
    if recent_anomalies:
        msg += f" {recent_anomalies} anomalous move(s) occurred in the last 30 trading days."
    if expected_change_pct is not None:
        direction = "rise" if expected_change_pct >= 0 else "fall"
        msg += (f" The best model projects a {abs(expected_change_pct):.2f}% {direction} by the end "
                "of the forecast horizon (treat this as indicative, not certain).")
    msg += " This tool is for education and decision support, not financial advice."
    return msg
