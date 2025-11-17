import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time


TICKER = "COP" #stock
PERIOD = "60d" #time period
INTERVAL = "5m" #candle interval


ATR_STOP_MULT = 1.2  #distance below price for trailing stop (in ATR units)

#filters
RVOL_THRESHOLD = 1.1  #at least slightly above average volume
SESSION_START = time(9, 35)
SESSION_END   = time(15, 55)

def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def to_eastern_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is tz-aware in America/New_York."""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    out = df.copy()
    out.index = idx
    return out


def add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-day VWAP."""
    df = df.copy()
    df["Date"] = df.index.date
    df["PV"] = df["Close"] * df["Volume"]
    df["CumVol"] = df.groupby("Date")["Volume"].cumsum()
    df["CumPV"] = df.groupby("Date")["PV"].cumsum()
    df["VWAP"] = df["CumPV"] / df["CumVol"]
    return df


def add_orh_avwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    AVWAP anchored from Opening Range High (first 30m).
    For each day:
      - find 9:30–10:00 high
      - anchor AVWAP there and forward
    """
    df = df.copy()
    df["ORH_AVWAP"] = np.nan
    df["Date"] = df.index.date
    df["Time"] = df.index.time

    for d, day_df in df.groupby("Date"):
        or_mask = (pd.Series(day_df["Time"]) >= time(9, 30)) & (pd.Series(day_df["Time"]) <= time(10, 0))
        or_df = day_df[or_mask]
        if or_df.empty:
            continue

        anchor_idx = or_df["High"].idxmax()
        anchored = day_df.loc[anchor_idx:]

        if anchored.empty:
            continue

        cum_vol = anchored["Volume"].cumsum()
        cum_pv = (anchored["Close"] * anchored["Volume"]).cumsum()
        avwap_vals = cum_pv / cum_vol

        df.loc[anchored.index, "ORH_AVWAP"] = avwap_vals

    df.drop(columns=["Time"], inplace=True)
    return df


print(f"Downloading {TICKER} + SPY {INTERVAL} data for last {PERIOD}...")

sym = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=False,
    progress=False,
)
spy = yf.download(
    "SPY",
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=False,
    progress=False,
)

if sym.empty or spy.empty:
    raise SystemExit("Download failed or returned no data. Check internet or ticker symbol.")

sym = flatten_cols(sym).dropna()
spy = flatten_cols(spy).dropna()

sym = to_eastern_index(sym)
spy = to_eastern_index(spy)

common_idx = sym.index.intersection(spy.index)
sym = sym.loc[common_idx].copy()
spy = spy.loc[common_idx].copy()

print(f"{TICKER} columns:", sym.columns.tolist())
print("First few rows:")
print(sym.head())


df = sym.copy()

# EMAs (fast / mid / slow)
df["EMA9"]  = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

# Session VWAP
df = add_session_vwap(df)

high_low   = df["High"] - df["Low"]
high_close = (df["High"] - df["Close"].shift()).abs()
low_close  = (df["Low"]  - df["Close"].shift()).abs()
tr_df = pd.DataFrame({"HL": high_low, "HC": high_close, "LC": low_close})
true_range = tr_df.max(axis=1)
df["ATR"] = true_range.rolling(14, min_periods=5).mean()
df["ATR_SMA"] = df["ATR"].rolling(14, min_periods=5).mean()

df["TimeIdx"] = df.index.strftime("%H:%M")
df["RVOL"] = df["Volume"] / df.groupby("TimeIdx")["Volume"].transform(
    lambda x: x.rolling(20, min_periods=5).mean()
)

df = add_orh_avwap(df)

#SPY indicators
spy["EMA20"] = spy["Close"].ewm(span=20, adjust=False).mean()

df = df.dropna(subset=["Close", "High", "Low", "Volume", "VWAP", "ATR", "ATR_SMA", "RVOL"])
spy = spy.loc[df.index]  #align again after drop


in_trade = False
trades = []

entry_price = None
entry_time  = None
stop_price  = None

#debug counters
cnt_type_A = 0   #VWAP reclaim + retest
cnt_type_B = 0   #VWAP tap & go
cnt_type_C = 0   #momentum ignition
cnt_signals = 0

for i in range(2, len(df)):
    row   = df.iloc[i]
    prev  = df.iloc[i - 1]
    prev2 = df.iloc[i - 2]
    idx   = df.index[i]
    t     = idx.time()
    spy_row = spy.iloc[i]

    if t < SESSION_START or t > SESSION_END:
        if in_trade and t >= SESSION_END:
            exit_price = row["Close"]
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return %": (exit_price - entry_price) / entry_price * 100.0,
            })
            in_trade = False
        continue

    if in_trade:
        low = row["Low"]
        high = row["High"]

        #trailing ATR stop: move stop up as price rises
        if not np.isnan(row["ATR"]):
            trailing = row["Close"] - ATR_STOP_MULT * row["ATR"]
            if np.isfinite(trailing):
                stop_price = max(stop_price, trailing)

        # 1)stop-loss hit?
        if low <= stop_price:
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx,
                "Entry Price": entry_price,
                "Exit Price": stop_price,
                "Return %": (stop_price - entry_price) / entry_price * 100.0,
            })
            in_trade = False
            continue

        # 2)EMA9 momentum exit (close crosses below EMA9)
        if (prev["Close"] >= prev["EMA9"]) and (row["Close"] < row["EMA9"]):
            exit_price = row["Close"]
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return %": (exit_price - entry_price) / entry_price * 100.0,
            })
            in_trade = False
            continue

        # 3)time-based exit at SESSION_END
        if t >= SESSION_END:
            exit_price = row["Close"]
            trades.append({
                "Entry Time": entry_time,
                "Exit Time": idx,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Return %": (exit_price - entry_price) / entry_price * 100.0,
            })
            in_trade = False
            continue

        #still in trade
        continue


    if any(pd.isna([row["VWAP"], row["ATR"], row["ATR_SMA"], row["RVOL"], prev["VWAP"], prev["ATR"]])):
        continue

    market_ok = spy_row["Close"] > spy_row["EMA20"]


    atr_slope = row["ATR"] - prev["ATR"]
    atr_ok = (row["ATR"] > row["ATR_SMA"]) or (atr_slope > 0)

    rvol_ok = (row["RVOL"] >= RVOL_THRESHOLD) or (row["RVOL"] > prev["RVOL"])

    if not (market_ok and atr_ok and rvol_ok):
        continue

    emas_trend = (row["EMA21"] > row["EMA50"]) and (row["EMA9"] >= row["EMA21"])

    if not emas_trend:
        continue


    orh_ok = True
    if not pd.isna(row["ORH_AVWAP"]):
        orh_ok = row["Close"] >= row["ORH_AVWAP"]
    if not orh_ok:
        continue


    vwap_slope = row["VWAP"] - prev["VWAP"]
    ema9_slope = row["EMA9"] - prev["EMA9"]


    reclaim = (prev2["Close"] < prev2["VWAP"]) and (prev["Close"] > prev["VWAP"])
    retest  = (row["Low"] <= row["VWAP"] * 1.002) and (row["Close"] > row["VWAP"]) and (row["Close"] > row["Open"])
    entry_A = reclaim and retest

    tap = (prev["Close"] > prev["VWAP"]) and (row["Low"] <= row["VWAP"] * 1.002)
    go  = (row["Close"] > prev["Close"]) and (vwap_slope > 0)
    entry_B = tap and go


    break_vwap  = (prev["Close"] <= prev["VWAP"]) and (row["Close"] > row["VWAP"])
    break_ema21 = (prev["Close"] <= prev["EMA21"]) and (row["Close"] > row["EMA21"])
    entry_C = (ema9_slope > 0) and break_vwap and break_ema21

    entry_signal = False
    entry_type = None

    if entry_A:
        cnt_type_A += 1
        entry_signal = True
        entry_type = "A"
    elif entry_B:
        cnt_type_B += 1
        entry_signal = True
        entry_type = "B"
    elif entry_C:
        cnt_type_C += 1
        entry_signal = True
        entry_type = "C"

    if not entry_signal:
        continue

    cnt_signals += 1

    # entry price: mid-candle
    entry_price = (row["Open"] + row["Close"]) / 2.0
    entry_time  = idx

    # initial ATR-based stop
    atr_here = row["ATR"]
    if np.isnan(atr_here) or atr_here <= 0:
        # fallback small fixed stop if ATR is weird
        stop_price = entry_price * (1.0 - 0.012)
    else:
        stop_price = entry_price - ATR_STOP_MULT * atr_here

    in_trade = True



df_trades = pd.DataFrame(trades)

print("\n---- DEBUG COUNTS ----")
print("Entry Type A (VWAP reclaim + retest):", cnt_type_A)
print("Entry Type B (VWAP tap & go):       ", cnt_type_B)
print("Entry Type C (momentum breakout):   ", cnt_type_C)
print("Total entry signals taken:          ", cnt_signals)

print(f"\n---- BACKTEST RESULTS ({TICKER} VWAP v2.0) ----")
if df_trades.empty:
    print("No trades triggered. Try loosening filters or changing parameters.")
else:
    wins   = df_trades[df_trades["Return %"] > 0]
    losses = df_trades[df_trades["Return %"] <= 0]

    total_trades = len(df_trades)
    win_rate  = len(wins) / total_trades * 100.0 if total_trades > 0 else 0.0
    avg_win   = wins["Return %"].mean() if not wins.empty else 0.0
    avg_loss  = losses["Return %"].mean() if not losses.empty else 0.0
    net_ret   = df_trades["Return %"].sum()

    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Avg win: {avg_win:.2f}%")
    print(f"Avg loss: {avg_loss:.2f}%")
    print(f"Net return (sum of %): {net_ret:.2f}%")

    print("\nSample trades:")
    print(df_trades.head())

    out_file = "results.csv"
    df_trades.to_csv(out_file, index=False)
    print(f"\nTrade log saved → {out_file}")
