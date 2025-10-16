# app/core/strategy.py (nur Funktion enrich anpassen)
import pandas as pd
from .indicators import sma, rsi, macd, atr, _as_series

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = _as_series(df["Close"])
    df["SMA20"] = sma(close, 20)
    df["SMA50"] = sma(close, 50)
    df["RSI14"] = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)
    df["MACD"], df["MACDsig"], df["MACDhist"] = macd_line, macd_signal, macd_hist
    df["ATR14"] = atr(df)
    return df

def signal_engine(df: pd.DataFrame) -> pd.Series:
    """
    Long-Signal wenn: SMA20 > SMA50 UND RSI > 50 UND MACD > MACD-Signal.
    (Kein Short in v0)
    """
    cond = (df["SMA20"] > df["SMA50"]) & (df["RSI14"] > 50) & (df["MACD"] > df["MACDsig"])
    return pd.Series(0, index=df.index).mask(cond, 1)

def position_size(equity: float, atr_val: float, risk_per_trade=0.01, atr_mult=2.0) -> float:
    """
    Positionsgröße (Dollar): (Equity * Risiko%) / (ATR * ATR-Multiplikator)
    """
    if pd.isna(atr_val) or atr_val <= 0:
        return 0.0
    risk_dollar = equity * risk_per_trade
    dollar_per_unit = atr_val * atr_mult
    return max(risk_dollar / dollar_per_unit, 0.0)

def backtest(df: pd.DataFrame, equity_start=10_000.0, fee_bp=2):
    """
    Einfache Simulation mit ATR-Stop (2*ATR).
    fee_bp = Gebühren in Basispunkten (2 = 0.02%) pro Trade.
    """
    df = enrich(df)
    sig = signal_engine(df)
    fee = fee_bp / 10_000

    pos = 0.0           # gehaltene Einheiten
    entry_price = None
    stop = None
    equity = equity_start
    curve = []

    for ts, row in df.iterrows():
        price = row["Close"]
        if pos == 0 and sig.loc[ts] == 1:
            # BUY
            size_dollar = position_size(equity, row["ATR14"])
            if size_dollar > 0:
                units = size_dollar / price
                entry_price = price
                stop = price - 2 * row["ATR14"]
                pos = units
                equity -= price * units * fee  # entry fee
        elif pos > 0:
            # trailing stop
            new_stop = price - 2 * row["ATR14"]
            if new_stop > stop:
                stop = new_stop
            # EXIT
            if sig.loc[ts] == 0 or price <= stop:
                equity += price * pos * (1 - fee)  # exit & fee
                pos = 0.0
                entry_price = None
                stop = None

        # Mark-to-market (aktueller Kontostand)
        equity_mtm = equity + (price * pos if pos > 0 else 0)
        curve.append((ts, equity_mtm))

    curve_df = pd.DataFrame(curve, columns=["Time", "Equity"]).set_index("Time")
    perf = (curve_df["Equity"].iloc[-1] / equity_start - 1) * 100
    return curve_df, perf
