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

# ========= Risikomanagement & Positionsgröße =========
st.subheader("🛡️ Risikomanagement & Positionsgröße")

# Sidebar-Parameter
st.sidebar.header("Risikoparameter")
equity = st.sidebar.number_input("Kontogröße (€)", min_value=100.0, value=5000.0, step=100.0)
risk_pct = st.sidebar.slider("Risiko pro Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
sl_mode = st.sidebar.selectbox("Stop-Loss-Methode", ["Prozent", "ATR x"])
sl_percent = st.sidebar.number_input("Stop-Loss (% unter Preis)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
atr_mult = st.sidebar.number_input("ATR-Multiplikator", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
atr_window = st.sidebar.number_input("ATR-Fenster", min_value=5, max_value=50, value=14, step=1)

# Aktueller Preis
price = float(data["Close"].iloc[-1])

# ATR (ohne Zusatzbibliotheken)
def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    prev_close = df["Close"].shift(1)
    tr1 = (df["High"] - df["Low"]).abs()
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

atr = compute_atr(data, atr_window)
atr_value = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else None

# Stop-Loss bestimmen
if sl_mode == "Prozent":
    stop_loss = price * (1 - sl_percent / 100.0)
elif atr_value is not None:
    stop_loss = price - atr_mult * atr_value
else:
    # Fallback: 2% wenn ATR nicht verfügbar ist
    stop_loss = price * 0.98

risk_per_share = max(price - stop_loss, 0.0)
max_risk_eur = equity * (risk_pct / 100.0)
position_size = int(max_risk_eur // risk_per_share) if risk_per_share > 0 else 0
notional = position_size * price
potential_loss = position_size * risk_per_share
take_profit = price + 2 * risk_per_share  # simples Chance/Risiko 2:1

c1, c2, c3 = st.columns(3)
c1.metric("Aktueller Preis", f"{price:,.2f} €" if price < 10000 else f"{price:,.2f}")
c2.metric("Vorgesehener Stop-Loss", f"{stop_loss:,.2f}")
c3.metric("Take-Profit (2:1 R:R)", f"{take_profit:,.2f}")

st.write(f"**Risiko pro Trade:** {risk_pct:.2f}% von {equity:,.0f} € → **{max_risk_eur:,.2f} €**")
st.write(f"**Risiko je Stück:** {risk_per_share:,.2f} → **Positionsgröße:** {position_size} Stück")
st.write(f"**Ordervolumen (ca.):** {notional:,.2f} €  |  **max. Verlust bei Stop:** {potential_loss:,.2f} €")

if atr_value is not None:
    st.caption(f"ATR({atr_window}) ≈ {atr_value:.3f} (für ATR-Stop-Loss genutzt, falls ausgewählt)")

st.divider()
