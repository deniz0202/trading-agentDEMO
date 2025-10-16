# app/core/indicators.py
import numpy as np
import pandas as pd

def _as_series(obj: pd.DataFrame | pd.Series) -> pd.Series:
    """
    Nimmt DataFrame/Series entgegen und gibt garantiert eine 1D-Series zurück.
    Behebt Fälle, in denen yfinance eine (n,1)-Matrix liefert.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        # Notfalls zusammenpressen:
        return obj.squeeze()
    return obj

def sma(series: pd.Series, window: int) -> pd.Series:
    s = _as_series(series).astype(float)
    return s.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = _as_series(series).astype(float)
    delta = s.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=s.index).rolling(period).mean()
    roll_down = pd.Series(down, index=s.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    s = _as_series(series).astype(float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _as_series(df["High"]).astype(float)
    low  = _as_series(df["Low"]).astype(float)
    close = _as_series(df["Close"]).astype(float)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
