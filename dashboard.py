import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="KI-Trading-Agent", layout="wide")
st.title("🤖 KI-Trading-Agent – Analyse & Backtesting")

# Eingaben
ticker = st.text_input("Aktien-Ticker eingeben (z. B. NVDA, AAPL, TSLA):", "NVDA").upper().strip()
period = st.selectbox("Zeitraum", ["1mo", "3mo", "6mo", "1y"], index=2)
interval = st.selectbox("Intervall", ["1d", "1h"], index=0)

@st.cache_data(show_spinner=False)
def load_data(tkr: str, per: str, intr: str) -> pd.DataFrame:
    """Robuster Datenabruf + Fallbacks; gibt DataFrame mit Spalte 'Close' zurück."""
    tk = yf.Ticker(tkr)
    df = tk.history(period=per, interval=intr, auto_adjust=False)

    # Fallbacks, falls leer
    if df.empty and per != "3mo":
        df = tk.history(period="3mo", interval=intr, auto_adjust=False)
    if df.empty and per != "1mo":
        df = tk.history(period="1mo", interval=intr, auto_adjust=False)

    # Falls Close fehlt, 'Adj Close' verwenden
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    return df

if ticker:
    data = load_data(ticker, period, interval)

    if data.empty or "Close" not in data.columns:
        st.warning("Keine Kursdaten gefunden. Prüfe Ticker (z. B. NVDA, AAPL, TSLA) oder Internetverbindung.")
    else:
        data = data.dropna(subset=["Close"]).copy()

        # Indikatoren
        data["SMA20"] = data["Close"].rolling(window=20, min_periods=20).mean()
        data["SMA50"] = data["Close"].rolling(window=50, min_periods=50).mean()

        # Handelssignale: SMA-Crossover
        data["Signal"] = np.where(data["SMA20"] > data["SMA50"], 1, 0)
        data["Position"] = data["Signal"].diff()

        st.subheader("📊 Kursverlauf mit gleitenden Durchschnitten")
        st.line_chart(data[["Close", "SMA20", "SMA50"]].dropna(), use_container_width=True)

        buy_signals = data[data["Position"] == 1]
        sell_signals = data[data["Position"] == -1]

        st.subheader("💡 Letzte Handelssignale")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Kaufsignale:**")
            st.dataframe(buy_signals[["Close"]].tail())
        with c2:
            st.write("**Verkaufssignale:**")
            st.dataframe(sell_signals[["Close"]].tail())

        # Backtest (sehr einfach)
        data["Return"] = data["Close"].pct_change()
        data["Strategy"] = data["Signal"].shift(1) * data["Return"]
        perf = (1 + data["Strategy"].dropna()).prod() - 1

        st.subheader("📈 Backtesting-Ergebnis")
        st.metric("Gesamt-Performance", f"{perf*100:.2f}%")
