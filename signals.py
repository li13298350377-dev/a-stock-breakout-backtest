"""股票池过滤与买卖信号。"""
from __future__ import annotations

import pandas as pd

from config import (
    BREAKOUT_LOOKBACK,
    MAX_20D_PCT,
    MAX_DAILY_PCT,
    MAX_NEXT_OPEN_GAP,
    MIN_NEXT_OPEN_GAP,
    MIN_5D_PCT,
    MIN_AVG_AMOUNT_20,
    MIN_DAILY_PCT,
    MIN_LISTING_DAYS,
    VOLUME_LOOKBACK,
    VOLUME_MULTIPLIER,
)


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """补齐指标列，统一字段类型。"""
    if df.empty:
        return df
    data = df.copy().sort_values("日期").reset_index(drop=True)
    num_cols = ["开盘", "收盘", "最高", "最低", "成交额", "涨跌幅"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["ma5"] = data["收盘"].rolling(5).mean()
    data["ma10"] = data["收盘"].rolling(10).mean()
    data["ma20"] = data["收盘"].rolling(20).mean()
    data["avg_amount_20"] = data["成交额"].rolling(VOLUME_LOOKBACK).mean()
    data["high_60_prev"] = data["收盘"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    data["pct_5"] = data["收盘"] / data["收盘"].shift(5) * 100 - 100
    data["pct_20"] = data["收盘"] / data["收盘"].shift(20) * 100 - 100
    data["next_open"] = data["开盘"].shift(-1)
    data["next_date"] = data["日期"].shift(-1)
    data["next_open_gap"] = data["next_open"] / data["收盘"] * 100 - 100
    return data


def passes_history_filters(df: pd.DataFrame) -> bool:
    """过滤上市不足、停牌/缺失、近 20 日平均成交额不足的股票。"""
    if df.empty or len(df) < max(MIN_LISTING_DAYS, BREAKOUT_LOOKBACK + 2):
        return False
    recent = df.tail(20)
    if recent[["开盘", "收盘", "成交额"]].isna().any().any():
        return False
    if (recent["成交额"] <= 0).any():
        return False
    return recent["成交额"].mean() > MIN_AVG_AMOUNT_20


def add_buy_signals(
    df: pd.DataFrame,
    max_next_open_gap: float = MAX_NEXT_OPEN_GAP,
    min_next_open_gap: float | None = MIN_NEXT_OPEN_GAP,
) -> pd.DataFrame:
    """生成突破买入信号；信号日次日开盘买入，可按次日开盘涨跌幅过滤。"""
    data = prepare_history(df)
    if data.empty:
        return data
    signal_rule = (
        (data["收盘"] > data["high_60_prev"])
        & (data["成交额"] > data["avg_amount_20"] * VOLUME_MULTIPLIER)
        & (data["涨跌幅"].between(MIN_DAILY_PCT, MAX_DAILY_PCT))
        & (data["pct_5"] > MIN_5D_PCT)
        & (data["pct_20"] < MAX_20D_PCT)
        & (data["收盘"] > data["ma5"])
        & (data["ma5"] > data["ma10"])
        & (data["ma10"] > data["ma20"])
    )
    data["buy_signal_rule"] = signal_rule
    data["buy_signal_before_gap"] = signal_rule & data["next_open"].notna()
    gap_filter = data["next_open_gap"] <= max_next_open_gap
    if min_next_open_gap is not None:
        gap_filter &= data["next_open_gap"] >= min_next_open_gap
    data["buy_signal"] = data["buy_signal_before_gap"] & gap_filter
    return data
