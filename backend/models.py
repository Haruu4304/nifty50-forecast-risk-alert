"""
models.py  (BACKEND)
--------------------
Forecasting models + evaluation.

Models compared:
  1. Naive            - baseline: tomorrow = today
  2. Holt-Winters     - damped-trend exponential smoothing
  3. ARIMA            - order (p,1,q) chosen by minimum AIC
  4. ARIMA+GARCH      - HYBRID: ARIMA gives the mean forecast, GARCH(1,1) gives the
                        time-varying uncertainty -> 95% confidence band

Data structures: pandas Series (prices), numpy arrays (forecasts),
dict {model_name: forecast_array}, pandas DataFrame (metrics table).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

NAIVE = "Naive"
HW = "Holt-Winters"
ARIMA = "ARIMA"
HYBRID = "ARIMA+GARCH"
ALL_MODELS = [NAIVE, HW, ARIMA, HYBRID]


# ----------------------------------------------------------------- evaluation
def evaluate(actual, pred, name: str) -> dict:
    """RMSE, MAE and MAPE for one model."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return {
        "Model": name,
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "MAE": float(mean_absolute_error(actual, pred)),
        "MAPE (%)": float(np.mean(np.abs((actual - pred) / actual)) * 100),
    }


# ---------------------------------------------------------------------- ARIMA
def select_arima_order(values: np.ndarray, max_p: int = 3, max_q: int = 3, d: int = 1):
    """Grid search p,q in 0..3 with d=1; pick the order with the lowest AIC."""
    best_aic, best_order = np.inf, (1, d, 0)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fit = SM_ARIMA(values, order=(p, d, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order = fit.aic, (p, d, q)
            except Exception:
                continue
    return best_order


# ---------------------------------------------------------------------- GARCH
def garch_band(train: pd.Series, mean_pred: np.ndarray, h: int, z: float = 1.96):
    """
    Turn a GARCH(1,1)-t volatility forecast into a price confidence band.
    cumulative std after k days = sqrt(sum of daily variances); band = price * cum_std.
    """
    ret_pct = np.log(train).diff().dropna() * 100
    fit = arch_model(ret_pct, vol="GARCH", p=1, q=1, dist="t", mean="Constant").fit(disp="off")
    var_path = fit.forecast(horizon=h, reindex=False).variance.values[-1]
    cum_std = np.sqrt(np.cumsum(var_path))
    band = float(train.iloc[-1]) * cum_std / 100
    return mean_pred - z * band, mean_pred + z * band


# ------------------------------------------------------------- fit + forecast
def fit_and_forecast(train: pd.Series, h: int, selected, arima_order=None) -> dict:
    """Fit every selected model on `train` and forecast `h` steps ahead."""
    values = np.asarray(train.values, float)
    out = {"forecasts": {}, "bands": {}, "arima_order": arima_order, "notes": []}

    if NAIVE in selected:
        out["forecasts"][NAIVE] = np.repeat(values[-1], h)

    if HW in selected:
        try:
            hw = ExponentialSmoothing(values, trend="add", damped_trend=True).fit()
            out["forecasts"][HW] = np.asarray(hw.forecast(h))
        except Exception as e:  # keep the app alive if one model fails
            out["notes"].append(f"Holt-Winters failed: {e}")

    if ARIMA in selected or HYBRID in selected:
        try:
            if arima_order is None:
                arima_order = select_arima_order(values)
            out["arima_order"] = arima_order
            pred = np.asarray(SM_ARIMA(values, order=arima_order).fit().forecast(h))
            if ARIMA in selected:
                out["forecasts"][ARIMA] = pred
            if HYBRID in selected:
                try:
                    lo, hi = garch_band(train, pred, h)
                    out["forecasts"][HYBRID] = pred
                    out["bands"][HYBRID] = (lo, hi)
                except Exception as e:
                    out["notes"].append(f"GARCH band failed: {e}")
        except Exception as e:
            out["notes"].append(f"ARIMA failed: {e}")
    return out


# ------------------------------------------------------------------- backtest
def run_backtest(close: pd.Series, h: int, selected, arima_order=None) -> dict:
    """Time-based split: last `h` trading days = test set. Never a random split."""
    selected = list(selected)
    train, test = close.iloc[:-h], close.iloc[-h:]
    out = fit_and_forecast(train, h, selected, arima_order)

    rows, coverage = [], None
    for name, pred in out["forecasts"].items():
        rows.append(evaluate(test.values, pred, name))
    if HYBRID in out["bands"]:
        lo, hi = out["bands"][HYBRID]
        coverage = float(np.mean((test.values >= lo) & (test.values <= hi)) * 100)

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values("RMSE").reset_index(drop=True)
    return {
        "train": train, "test": test,
        "forecasts": out["forecasts"], "bands": out["bands"],
        "results": results, "coverage": coverage,
        "arima_order": out["arima_order"], "notes": out["notes"],
    }


def forecast_future(close: pd.Series, h: int, selected, arima_order=None):
    """Refit on ALL data and forecast the next `h` business days."""
    out = fit_and_forecast(close, h, list(selected), arima_order)
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=h)
    return dates, out


# -------------------------------------------------------------- interpretation
def interpret_results(results: pd.DataFrame, coverage, horizon: int) -> str:
    """Plain-English explanation of the metrics table (marks are given for interpretation)."""
    if results.empty:
        return "No model produced a forecast."
    best = results.iloc[0]
    text = (f"**{best['Model']}** achieved the lowest RMSE ({best['RMSE']:.1f} index points, "
            f"MAPE {best['MAPE (%)']:.2f}%) over the {horizon}-day test window. ")
    naive = results[results["Model"] == NAIVE]
    if not naive.empty and best["Model"] != NAIVE:
        gain = (naive.iloc[0]["RMSE"] - best["RMSE"]) / naive.iloc[0]["RMSE"] * 100
        text += (f"It improves on the naive baseline by {gain:.1f}%. ")
        if gain < 10:
            text += ("The gain is small: index prices behave close to a random walk, so "
                     "short-term point forecasts are hard to beat the last observed value by much. ")
    elif best["Model"] == NAIVE:
        text += ("The naive baseline won, which is typical for efficient markets: prices are close "
                 "to a random walk, so complex models add little for short-horizon point forecasts. ")
    if coverage is not None:
        text += (f"The ARIMA+GARCH 95% band contained {coverage:.0f}% of the actual test values; "
                 "its main value is quantifying uncertainty that widens when volatility is high.")
    return text
