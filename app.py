"""
app.py  (FRONTEND - Streamlit)
------------------------------
NIFTY 50 Forecasting and Volatility Risk Alert System
Run with:  streamlit run app.py
All heavy logic lives in backend/ ; this file only handles the UI.
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backend import data_loader, eda, models, risk

# ------------------------------------------------------------------ theme
C = {
    "bg": "#0B1220", "card": "#131C2E", "border": "#1F2A44",
    "primary": "#14B8A6", "accent": "#F59E0B", "blue": "#3B82F6",
    "green": "#22C55E", "red": "#EF4444", "text": "#E5E7EB", "muted": "#9CA3AF",
}
MODEL_COLORS = {models.NAIVE: C["muted"], models.HW: C["blue"],
                models.ARIMA: C["accent"], models.HYBRID: "#A855F7"}
RISK_COLORS = {"LOW": C["green"], "MEDIUM": C["accent"], "HIGH": C["red"]}

st.set_page_config(page_title="NIFTY 50 Forecast & Risk Alert", page_icon="📈", layout="wide")
st.markdown(f"""
<style>
div[data-testid="stMetric"] {{background:{C['card']}; border:1px solid {C['border']};
    border-radius:12px; padding:14px 18px;}}
div[data-testid="stMetricLabel"] p {{color:{C['muted']};}}
.stTabs [data-baseweb="tab-list"] {{gap:6px;}}
.stTabs [data-baseweb="tab"] {{background:{C['card']}; border-radius:8px 8px 0 0; padding:8px 16px;}}
.stTabs [aria-selected="true"] {{border-bottom:3px solid {C['primary']};}}
.card {{background:{C['card']}; border:1px solid {C['border']}; border-radius:12px; padding:16px 20px;}}
.badge {{display:inline-block; padding:10px 28px; border-radius:30px; font-size:1.6rem;
    font-weight:700; color:#0B1220;}}
h1, h2, h3 {{color:{C['text']};}}
</style>
""", unsafe_allow_html=True)


def style(fig, title="", height=420):
    """Apply the dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark", title=title, height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor="#1F2A44")
    fig.update_yaxes(gridcolor="#1F2A44")
    return fig


# ---------------------------------------------------------------- caching
@st.cache_data(show_spinner=False, ttl=3600)
def load_live(start, end):
    return data_loader.fetch_nifty(start, end)


@st.cache_data(show_spinner=False)
def cached_backtest(close, h, selected):
    return models.run_backtest(close, h, selected)


@st.cache_data(show_spinner=False)
def cached_future(close, h, selected, order):
    return models.forecast_future(close, h, selected, order)


@st.cache_data(show_spinner=False)
def cached_garch(returns):
    fit, cond_vol, std_resid = risk.fit_garch_full(returns)
    return str(fit.summary()), cond_vol, std_resid


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 📈 NIFTY 50 Analyzer")
    source = st.radio("Data source", ["Live (Yahoo Finance)", "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV with Date and Close columns", type=["csv"])
    start = st.date_input("Start date", dt.date(2015, 1, 1), min_value=dt.date(2007, 9, 17))
    end = st.date_input("End date", dt.date.today())
    horizon = st.slider("Forecast horizon (trading days)", 7, 30, 14)
    selected = st.multiselect("Models to compare", models.ALL_MODELS, default=models.ALL_MODELS)
    threshold = st.slider("Anomaly threshold (sigma)", 2.0, 4.0, 3.0, 0.1)
    run = st.button("▶ Run Analysis", type="primary", width="stretch")
    st.caption("Educational project. Not financial advice.")

# ------------------------------------------------------------- run pipeline
if run:
    try:
        if not selected:
            raise ValueError("Select at least one model.")
        if start >= end:
            raise ValueError("Start date must be before end date.")

        with st.spinner("Loading and cleaning data..."):
            if source == "Upload CSV":
                if uploaded is None:
                    raise ValueError("Please upload a CSV file first.")
                raw = data_loader.load_csv(uploaded).loc[str(start):str(end)]
            else:
                raw = load_live(start, end)
            df, report = data_loader.clean_data(raw)
            if len(df) < max(300, horizon * 5):
                raise ValueError(f"Only {len(df)} usable rows. Choose a longer date range (300+ trading days).")
            close = df["Close"]

        with st.spinner("Fitting models (ARIMA order search may take ~20s the first time)..."):
            bt = cached_backtest(close, horizon, tuple(selected))
            dates, fut = cached_future(close, horizon, tuple(selected), bt["arima_order"])

        with st.spinner("Fitting GARCH risk model..."):
            summary, cond_vol, std_resid = cached_garch(df["Return"])

        st.session_state["res"] = dict(
            df=df, report=report, close=close, bt=bt, fut_dates=dates, fut=fut,
            garch_summary=summary, cond_vol=cond_vol, std_resid=std_resid,
            horizon=horizon, threshold=threshold, source=source,
        )
    except Exception as e:
        st.session_state.pop("res", None)
        st.error(f"Could not complete the analysis: {e}")

res = st.session_state.get("res")

# --------------------------------------------------------------------- tabs
st.title("NIFTY 50 Forecasting & Volatility Risk Alert System")
tab_about, tab_dash, tab_eda, tab_fc, tab_risk, tab_lim = st.tabs(
    ["About", "Dashboard", "EDA", "Forecast", "Risk & Alerts", "Limitations"])

# ----------------------------------------------------------------- About
with tab_about:
    st.markdown(f"""
<div class="card">

### Problem
Retail investors and finance students find it hard to judge where the NIFTY 50 may head and how risky the
market currently is. This app turns historical index data into **forecasts, model comparisons, volatility
regimes, anomaly alerts and plain-language recommendations**.

### Intended users
Retail investors, finance students and analysts who want data-driven decision support.

### How to use
1. Choose a data source (live Yahoo Finance or your own CSV), date range, forecast horizon and models in the sidebar.
2. Click **Run Analysis**.
3. Explore the **Dashboard**, **EDA**, **Forecast** and **Risk & Alerts** tabs.

### TSDA techniques used
Moving averages, seasonal decomposition, ADF/KPSS stationarity tests, ACF/PACF, Holt-Winters exponential
smoothing, ARIMA, GARCH(1,1), an ARIMA+GARCH hybrid and anomaly detection.
</div>
""", unsafe_allow_html=True)
    if res is None:
        st.info("👈 Set your inputs in the sidebar and click **Run Analysis** to begin.")

if res is None:
    for t in (tab_dash, tab_eda, tab_fc, tab_risk):
        with t:
            st.info("Run the analysis from the sidebar to see results here.")
else:
    df, close, bt = res["df"], res["close"], res["bt"]
    h = res["horizon"]

    # ------------------------------------------------------------ Dashboard
    with tab_dash:
        last, prev = float(close.iloc[-1]), float(close.iloc[-2])
        vol30 = float(df["Return"].tail(30).std() * np.sqrt(252) * 100)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest close", f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%")
        c2.metric("30-day volatility (ann.)", f"{vol30:.1f}%")
        c3.metric("Data range", f"{df.index.min():%b %Y} to {df.index.max():%b %Y}")
        c4.metric("Trading days", f"{len(df):,}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=close, name="Close", line=dict(color=C["primary"], width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=close.rolling(50).mean(), name="50-day MA",
                                 line=dict(color=C["accent"], width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=close.rolling(200).mean(), name="200-day MA",
                                 line=dict(color=C["red"], width=1.5)))
        st.plotly_chart(style(fig, "NIFTY 50 closing price with moving averages"), width="stretch")

        if "Volume" in df.columns and df["Volume"].fillna(0).sum() > 0:
            vfig = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color=C["blue"]))
            st.plotly_chart(style(vfig, "Trading volume", 260), width="stretch")

        with st.expander("Data quality report"):
            st.dataframe(pd.DataFrame(res["report"].items(), columns=["Check", "Count"]),
                         hide_index=True, width="stretch")

    # ------------------------------------------------------------------ EDA
    with tab_eda:
        st.subheader("Trend, seasonality and residual")
        try:
            dec, period = eda.decompose(close)
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                subplot_titles=("Observed", "Trend", f"Seasonal (period={period})", "Residual"))
            for i, (s, col) in enumerate([(dec.observed, C["primary"]), (dec.trend, C["accent"]),
                                          (dec.seasonal, C["blue"]), (dec.resid, C["muted"])], start=1):
                fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=col), showlegend=False), row=i, col=1)
            st.plotly_chart(style(fig, "", 640), width="stretch")
            st.caption("A clear upward trend dominates; the seasonal component is tiny compared to the trend, "
                       "so seasonality is weak in daily index data.")
        except Exception as e:
            st.warning(f"Decomposition unavailable: {e}")

        st.subheader("Returns and volatility clustering")
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Scatter(x=df.index, y=df["Return"] * 100, line=dict(color="#A855F7", width=1)))
            out = df[df["Outlier"]]
            fig.add_trace(go.Scatter(x=out.index, y=out["Return"] * 100, mode="markers", name="Outliers",
                                     marker=dict(color=C["red"], size=5)))
            st.plotly_chart(style(fig, "Daily log returns (%)", 340), width="stretch")
        with b:
            roll = df["Return"].rolling(30).std() * np.sqrt(252) * 100
            fig = go.Figure(go.Scatter(x=roll.index, y=roll, line=dict(color=C["red"])))
            st.plotly_chart(style(fig, "30-day rolling annualised volatility (%)", 340), width="stretch")

        a, b = st.columns([2, 1])
        with a:
            fig = go.Figure(go.Histogram(x=df["Return"] * 100, nbinsx=100, marker_color=C["primary"]))
            st.plotly_chart(style(fig, "Distribution of daily returns (%)", 320), width="stretch")
        with b:
            st.metric("Skewness", f"{df['Return'].skew():.2f}")
            st.metric("Excess kurtosis", f"{df['Return'].kurt():.2f}")
            st.caption("Kurtosis far above 0 means fat tails: extreme days are more common than a normal "
                       "distribution predicts. Volatility comes in clusters, which motivates GARCH.")

        st.subheader("Stationarity tests")
        rep = pd.DataFrame([
            eda.stationarity_report(close, "Close price"),
            eda.stationarity_report(close.diff(), "First difference of price"),
            eda.stationarity_report(df["Return"], "Log returns"),
        ])
        st.dataframe(rep, hide_index=True, width="stretch")
        st.caption("Prices are normally non-stationary while differenced prices / returns are stationary. "
                   "This justifies d=1 in ARIMA and modelling returns with GARCH.")

        st.subheader("ACF / PACF")
        st.pyplot(eda.acf_pacf_figure(close.diff(), "differenced Close", colors=C))
        st.pyplot(eda.acf_pacf_figure(df["Return"] ** 2, "squared returns", colors=C))
        st.caption("Significant autocorrelation in squared returns confirms volatility clustering (ARCH effects).")

    # ------------------------------------------------------------- Forecast
    with tab_fc:
        for n in bt["notes"] + res["fut"]["notes"]:
            st.warning(n)
        if bt["results"].empty:
            st.error("No model produced a forecast. Try a longer date range.")
        else:
            st.subheader(f"Backtest: last {h} trading days held out as test set")
            fig = go.Figure()
            tail = bt["train"].iloc[-120:]
            fig.add_trace(go.Scatter(x=tail.index, y=tail, name="Train (last 120 days)",
                                     line=dict(color=C["muted"])))
            fig.add_trace(go.Scatter(x=bt["test"].index, y=bt["test"], name="Actual",
                                     line=dict(color=C["primary"], width=4)))
            if models.HYBRID in bt["bands"]:
                lo, hi = bt["bands"][models.HYBRID]
                fig.add_trace(go.Scatter(x=bt["test"].index, y=hi, line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=bt["test"].index, y=lo, fill="tonexty", line=dict(width=0),
                                         fillcolor="rgba(168,85,247,0.2)", name="ARIMA+GARCH 95% band"))
            for name, pred in bt["forecasts"].items():
                fig.add_trace(go.Scatter(x=bt["test"].index, y=pred, name=name,
                                         line=dict(color=MODEL_COLORS[name], dash="dash")))
            st.plotly_chart(style(fig, "Actual vs predicted"), width="stretch")

            st.subheader("Model comparison")
            r = bt["results"]
            st.dataframe(
                r.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "MAPE (%)": "{:.3f}"})
                 .highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color="#0F766E"),
                hide_index=True, width="stretch")
            if bt["arima_order"]:
                st.caption(f"ARIMA order selected by AIC: {bt['arima_order']}")
            st.info(models.interpret_results(r, bt["coverage"], h))

            st.subheader(f"Forecast for the next {h} business days")
            dates, fut = res["fut_dates"], res["fut"]
            fig = go.Figure()
            hist = close.iloc[-120:]
            fig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(color=C["primary"], width=3)))
            if models.HYBRID in fut["bands"]:
                lo, hi = fut["bands"][models.HYBRID]
                fig.add_trace(go.Scatter(x=dates, y=hi, line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=dates, y=lo, fill="tonexty", line=dict(width=0),
                                         fillcolor="rgba(168,85,247,0.2)", name="95% band (GARCH)"))
            for name, pred in fut["forecasts"].items():
                fig.add_trace(go.Scatter(x=dates, y=pred, name=name, line=dict(color=MODEL_COLORS[name], dash="dash")))
            st.plotly_chart(style(fig, "Future forecast"), width="stretch")

            table = pd.DataFrame({"Date": dates})
            for name, pred in fut["forecasts"].items():
                table[name] = np.round(pred, 2)
            if models.HYBRID in fut["bands"]:
                table["Lower 95%"] = np.round(fut["bands"][models.HYBRID][0], 2)
                table["Upper 95%"] = np.round(fut["bands"][models.HYBRID][1], 2)
            st.dataframe(table, hide_index=True, width="stretch")
            st.download_button("⬇ Download forecast CSV", table.to_csv(index=False).encode(),
                               "nifty_forecast.csv", "text/csv")
            st.caption("Future dates skip weekends only; Indian market holidays are not modelled.")

    # --------------------------------------------------------- Risk & Alerts
    with tab_risk:
        cond_vol, std_resid = res["cond_vol"], res["std_resid"]
        info = risk.risk_level(cond_vol)
        anomalies = risk.detect_anomalies(std_resid, df["Return"], res["threshold"])
        recent = int((anomalies.index >= df.index[-30]).sum()) if len(anomalies) else 0

        expected = None
        if not bt["results"].empty:
            best = bt["results"].iloc[0]["Model"]
            if best in res["fut"]["forecasts"]:
                expected = (res["fut"]["forecasts"][best][-1] / float(close.iloc[-1]) - 1) * 100

        a, b = st.columns([1, 2])
        with a:
            col = RISK_COLORS[info["level"]]
            st.markdown(f"<div class='card' style='text-align:center'>"
                        f"<div style='color:{C['muted']}'>Current market risk</div><br>"
                        f"<span class='badge' style='background:{col}'>{info['level']}</span><br><br>"
                        f"<div style='color:{C['muted']}'>Annualised volatility</div>"
                        f"<h2 style='margin:0'>{info['current']:.1f}%</h2></div>", unsafe_allow_html=True)
        with b:
            st.markdown(f"<div class='card'><b>Recommendation</b><br><br>"
                        f"{risk.recommendation(info['level'], info['current'], recent, expected)}</div>",
                        unsafe_allow_html=True)

        fig = go.Figure(go.Scatter(x=cond_vol.index, y=cond_vol, name="GARCH volatility",
                                   line=dict(color=C["primary"])))
        fig.add_hline(y=info["low_thr"], line_dash="dot", line_color=C["green"], annotation_text="LOW / MEDIUM")
        fig.add_hline(y=info["high_thr"], line_dash="dot", line_color=C["red"], annotation_text="MEDIUM / HIGH")
        st.plotly_chart(style(fig, "GARCH(1,1) conditional volatility (annualised %)"), width="stretch")

        st.subheader(f"Anomalous days (> {res['threshold']:.1f} sigma): {len(anomalies)} found")
        if len(anomalies):
            st.dataframe(anomalies, width="stretch")
        else:
            st.success("No anomalies at this threshold.")
        with st.expander("GARCH model summary"):
            st.code(res["garch_summary"])

# ------------------------------------------------------------ Limitations
with tab_lim:
    st.markdown("""
### Assumptions
- Historical patterns (autocorrelation, volatility clustering) persist into the near future.
- Log returns are modelled as stationary; GARCH(1,1) with Student-t errors captures fat tails.

### Strengths and weaknesses
| Model | Strength | Weakness |
|---|---|---|
| Naive | Strong baseline for random-walk-like series | No learning, no uncertainty |
| Holt-Winters (damped) | Simple, smooth trend extrapolation | Ignores volatility, weak at turning points |
| ARIMA | Captures short-term autocorrelation, interpretable | Assumes constant variance, long-horizon forecasts flatten |
| ARIMA+GARCH | Adds time-varying uncertainty bands | Point forecast same as ARIMA; GARCH assumes stable parameters |

### Limitations
- Markets react to news, policy and global events that a univariate model cannot see.
- Forecast accuracy falls quickly with longer horizons.
- Indian market holidays are not modelled in future dates.
- Backtest uses a single hold-out window; rolling-origin validation would be more robust.

### Future enhancements
India VIX, FII/DII flows and news sentiment as exogenous inputs; LSTM / Prophet models; rolling-origin
cross-validation; Bank Nifty and sector indices; email/Telegram alerts when risk turns HIGH.
""")
