# NIFTY 50 Forecasting and Volatility Risk Alert System

TSDA mini project: an interactive Streamlit application that forecasts the NIFTY 50 index, compares
time series models, detects volatility regimes and anomalies, and gives risk recommendations.

## Problem and users
Retail investors and finance students find it hard to judge future index movement and current market
risk. The app turns historical data into forecasts, uncertainty bands, a LOW/MEDIUM/HIGH risk badge,
anomaly alerts and plain-language advice.

## Dataset
- Source: Yahoo Finance, ticker `^NSEI` (NIFTY 50), fetched live through `yfinance`
- Attributes: Open, High, Low, Close, Volume
- Frequency: daily (trading days); default duration 2015 to today
- Alternative: upload your own CSV with `Date` and `Close` columns

## TSDA techniques
Moving averages, multiplicative decomposition, ADF and KPSS tests, ACF/PACF, Holt-Winters (damped),
ARIMA (order by AIC), GARCH(1,1) with Student-t errors, ARIMA+GARCH hybrid, GARCH-residual anomaly detection.
Models are compared with RMSE, MAE and MAPE on a time-based hold-out window against a naive baseline.

## Project structure
```
tsda_nifty_app/
├── app.py                  # Streamlit frontend (UI only)
├── backend/
│   ├── data_loader.py      # yfinance fetch, CSV upload, cleaning, log returns
│   ├── eda.py              # decomposition, stationarity tests, ACF/PACF
│   ├── models.py           # Naive, Holt-Winters, ARIMA, hybrid, evaluation
│   └── risk.py             # GARCH risk level, anomalies, recommendation
├── .streamlit/config.toml  # dark finance theme
├── requirements.txt
├── README.md
└── screenshots/            # add your screenshots here
```

## Setup and run
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501. An internet connection is needed for live data.

## Realistic use case (demo)
1. Sidebar: source = Live, start date = 2 years ago, horizon = 14 days, all four models, threshold = 3.
2. Click **Run Analysis**.
3. Dashboard shows the latest close and volatility; EDA shows non-stationary prices and stationary returns.
4. Forecast tab shows actual vs predicted, the metrics table and the written interpretation.
5. Risk & Alerts tab shows the risk badge (for example HIGH), anomalous days and the recommendation.

## Limitations
Univariate models cannot see news or macro events; accuracy drops over longer horizons; Indian market
holidays are not modelled in future dates; a single hold-out window is used for evaluation.

## Future work
India VIX / FII-DII / sentiment as exogenous inputs, LSTM and Prophet, rolling-origin validation,
alerts by email or Telegram.

## Team
Add group member names and contributions here.
