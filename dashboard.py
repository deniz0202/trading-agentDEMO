# dashboard.py
# Robustes SMA-Crossover-Dashboard (flacht yfinance-Spalten ab)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ---------- Seiteneinstellungen ----------
st.set_page_config(
    page_title="KI-Trading-Agent – Analyse & Backtesting",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 KI-Trading-Agent – Analyse & Backtesting")

# ---------- UI ----------
ticker = st.text_input("Aktien-Ticker eingeben (z. B. NVDA, AAPL, TSLA):", value="NVDA").upper()

period_options = {
    "1m": "1mo",
    "3m": "3mo",
    "6m": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y",
    "10y": "10y",
    "max": "max",
}
period_key = st.selectbox("Zeitraum", list(period_options.keys()), index=2)
period_val = period_options[period_key]

interval_options = {
    "1d (Tagesdaten)": "1d",
    "1h (Stunden)": "1h",
}
interval_key = st.selectbox("Intervall", list(interval_options.keys()), index=0)
interval_val = interval_options[interval_key]

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Einstellungen")
fast = st.sidebar.number_input("SMA schnell", min_value=5, max_value=100, value=20)
slow = st.sidebar.number_input("SMA langsam", min_value=10, max_value=250, value=50)
if fast >= slow:
    st.sidebar.warning("Hinweis: 'SMA schnell' sollte < 'SMA langsam' sein.")
fee_bps = st.sidebar.number_input("Gebühren pro Trade (bps)", min_value=0, max_value=100, value=5)  # 5 bps = 0.05%

# ---------- Helpers ----------
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex-Spalten zu flachen Namen machen."""
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [str(p) for p in tup if p is not None and str(p) != ""]
            name = "_".join(parts).strip("_")
            new_cols.append(name)
        df = df.copy()
        df.columns = new_cols
    return df

def find_close_column(cols) -> str | None:
    """Eine passende Close-Spalte finden (Close, Adj Close, *_Close, Close_* etc.)."""
    candidates = [c for c in cols if c.lower() == "close"]
    if candidates:
        return candidates[0]
    # häufige Varianten aus yfinance-MultiIndex-Flattening
    for key in ("adj close", "close", ".close", "_close"):
        hits = [c for c in cols if key in c.lower()]
        if hits:
            return hits[0]
    # fallback: irgendwas mit "close"
    any_hits = [c for c in cols if "close" in c.lower()]
    return any_hits[0] if any_hits else None

# ---------- Daten laden (mit Caching) ----------
@st.cache_data(ttl=3600)
def load_prices(tick: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tick,
        period=period,
        interval=interval,
        auto_adjust=True,   # bereits adjustierte Close
        progress=False,
    )
    df = flatten_columns(df)

    # Close-Spalte robust bestimmen
    close_col = find_close_column(df.columns)
    if close_col is None:
        return pd.DataFrame()  # leer -> später handled

    # Einheitliche 'Close'-Spalte anlegen, damit Rest des Codes simpel bleibt
    out = df.copy()
    out["Close"] = out[close_col].astype(float)
    return out

df = load_prices(ticker, period_val, interval_val)

if df is None or df.empty or "Close" not in df.columns:
    st.warning("Keine Kursdaten gefunden. Prüfe Ticker, Zeitraum oder Intervall.")
    st.stop()

# ---------- Indikatoren ----------
df = df.copy()
df["SMA_fast"] = df["Close"].rolling(int(fast)).mean()
df["SMA_slow"] = df["Close"].rolling(int(slow)).mean()

# ---------- Chart ----------
st.subheader("📊 Kursverlauf mit gleitenden Durchschnitten")
plot_cols = df[["Close", "SMA_fast", "SMA_slow"]].dropna().rename(
    columns={"SMA_fast": f"SMA{fast}", "SMA_slow": f"SMA{slow}"}
)
st.line_chart(plot_cols)

# ---------- Signale ----------
position = (df["SMA_fast"] > df["SMA_slow"]).astype(int)
signal = position.diff().fillna(0)       # +1 = Buy, -1 = Sell

buy_idx = signal[signal == 1].index
sell_idx = signal[signal == -1].index

buy_signals = pd.DataFrame({"Date": buy_idx, "Close": df.loc[buy_idx, "Close"].values})
sell_signals = pd.DataFrame({"Date": sell_idx, "Close": df.loc[sell_idx, "Close"].values})

st.subheader("💡 Letzte Handelssignale")
c1, c2 = st.columns(2)
with c1:
    st.write("**Kaufsignale:**")
    st.dataframe(buy_signals.tail(10), use_container_width=True)
with c2:
    st.write("**Verkaufssignale:**")
    st.dataframe(sell_signals.tail(10), use_container_width=True)

# ---------- Backtest (einfach, mit Gebühren) ----------
ret = df["Close"].pct_change().fillna(0)
turnover = position.diff().abs().fillna(0)               # 1 bei Entry/Exit
cost = turnover * (fee_bps / 10000.0)                    # bps => %

strat_ret = ret * position.shift(1).fillna(0) - cost
equity = (1 + strat_ret).cumprod().fillna(1.0)
perf = float(equity.iloc[-1] - 1.0)

# Kennzahlen
def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return float(dd.min())

def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

periods_map = {"1d": 252, "1h": 252 * 6}
ppy = periods_map.get(interval_val, 252)

days = max(1, len(df))
try:
    cagr = equity.iloc[-1] ** (ppy / days) - 1.0
except Exception:
    cagr = 0.0

dd = max_drawdown(equity)
sr = sharpe_ratio(strat_ret.fillna(0), periods_per_year=ppy)

st.subheader("📈 Equity-Kurve (Strategie)")
st.line_chart(equity.rename("Equity"))

st.subheader("📊 Backtesting-Ergebnis")
m1, m2, m3 = st.columns(3)
m1.metric("Gesamt-Performance", f"{perf * 100:.2f}%")
m2.metric("Max Drawdown", f"{dd * 100:.2f}%")
m3.metric("Sharpe", f"{sr:.2f}")

st.caption(
    "Hinweis: Vereinfachte Demo (SMA-Crossover). Gebühren als bps pro Entry/Exit, "
    "keine Slippage/Steuern/Ausführungsrisiken. Keine Finanzberatung."
)
