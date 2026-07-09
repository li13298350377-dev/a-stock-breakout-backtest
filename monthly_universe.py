"""历史月度动态股票池研究原型。

第一阶段只生成 2023 年 1 月股票池，不运行交易策略、不做 A1 回测。
所有筛选与评分仅使用筛选截止日及以前可获得的历史数据。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

from config import (
    BASE_DIR,
    MAIN_BOARD_PREFIXES,
    MAX_MARKET_CAP,
    MAX_PRICE,
    MIN_AVG_AMOUNT_20,
    MIN_LISTING_DAYS,
    MIN_MARKET_CAP,
    MIN_PRICE,
)
from data_loader import load_stock_history, normalize_code, retry_call

DATA_START_DATE = "20220701"
RESEARCH_START_DATE = "20230101"
TARGET_MONTH = "2023-01"
RESULT_DIR = BASE_DIR / "monthly_universe_results" / "2023_01"


@dataclass
class Diagnostics:
    initial_candidates: int = 0
    history_success: int = 0
    share_success: int = 0
    market_cap_missing: int = 0
    base_passed: int = 0
    final_pool: int = 0


def _yyyymmdd(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def get_trade_dates() -> pd.DatetimeIndex:
    """读取交易日历，优先使用 AKShare 新浪交易日历。"""
    dates = retry_call(ak.tool_trade_date_hist_sina)
    if dates is None or dates.empty:
        raise RuntimeError("无法获取交易日历，不能确定月度首个交易日和筛选截止日。")

    date_col = "trade_date" if "trade_date" in dates.columns else dates.columns[0]
    values = pd.to_datetime(dates[date_col], errors="coerce").dropna().sort_values().unique()
    return pd.DatetimeIndex(values)


def resolve_month_dates(target_month: str = TARGET_MONTH) -> tuple[pd.Timestamp, pd.Timestamp]:
    """返回目标月份第一个实际交易日及其前一个实际交易日。"""
    trade_dates = get_trade_dates()
    month_start = pd.Timestamp(f"{target_month}-01")
    next_month = month_start + pd.offsets.MonthBegin(1)
    month_dates = trade_dates[(trade_dates >= month_start) & (trade_dates < next_month)]
    if month_dates.empty:
        raise RuntimeError(f"{target_month} 没有可识别交易日。")

    first_trade_date = month_dates[0]
    previous_dates = trade_dates[trade_dates < first_trade_date]
    if previous_dates.empty:
        raise RuntimeError(f"无法找到 {first_trade_date.date()} 之前的筛选截止交易日。")
    return first_trade_date, previous_dates[-1]


def load_historical_candidates() -> pd.DataFrame:
    """构建历史候选集合：只使用证券代码清单，不使用实时行情或 FALLBACK_SYMBOLS。"""
    codes = retry_call(ak.stock_info_a_code_name)
    if codes is None or codes.empty:
        raise RuntimeError("无法获取 A 股代码名称清单，不能构建历史候选股集合。")

    code_col = "code" if "code" in codes.columns else "代码"
    name_col = "name" if "name" in codes.columns else "名称"
    df = codes[[code_col, name_col]].rename(columns={code_col: "code", name_col: "name"}).copy()
    df["code"] = df["code"].map(normalize_code)
    df = df[df["code"].str.startswith(MAIN_BOARD_PREFIXES)]
    return df.drop_duplicates("code").sort_values("code").reset_index(drop=True)


def _find_column(columns: list[str], keywords: tuple[str, ...]) -> Optional[str]:
    for col in columns:
        text = str(col).strip().lower()
        if all(keyword.lower() in text for keyword in keywords):
            return col
    return None


def load_historical_total_shares(code: str, cutoff_date: pd.Timestamp) -> tuple[Optional[float], str]:
    """读取截至筛选日已生效的最近一期总股本，不用当前总股本兜底。"""
    df = retry_call(ak.stock_zh_a_gbjg_em, symbol=normalize_code(code))
    if df is None or df.empty:
        return None, "股本结构接口无数据或请求失败"

    columns = list(df.columns)
    date_col = _find_column(columns, ("变更", "日期")) or _find_column(columns, ("公告", "日期"))
    shares_col = _find_column(columns, ("总股本",))
    if date_col is None or shares_col is None:
        return None, f"股本结构字段缺失: {list(df.columns)}"

    data = df[[date_col, shares_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[shares_col] = pd.to_numeric(data[shares_col], errors="coerce")
    data = data.dropna(subset=[date_col, shares_col])
    data = data[data[date_col] <= cutoff_date].sort_values(date_col)
    if data.empty:
        return None, "筛选截止日前无已生效总股本记录"

    shares = float(data.iloc[-1][shares_col])
    if "万" in str(shares_col):
        shares *= 10_000
    return shares, ""


def _history_until_cutoff(code: str, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    history = load_stock_history(code, DATA_START_DATE, _yyyymmdd(cutoff_date), adjust="")
    if history.empty:
        return history
    history = history.copy()
    history["日期"] = pd.to_datetime(history["日期"], errors="coerce")
    return history[history["日期"] <= cutoff_date].sort_values("日期").reset_index(drop=True)


def active_days_20(amount: pd.Series) -> int:
    flags = []
    for idx in range(len(amount) - 20, len(amount)):
        previous_60 = amount.iloc[max(0, idx - 60):idx].dropna()
        if len(previous_60) < 20:
            flags.append(False)
        else:
            flags.append(amount.iloc[idx] > previous_60.median())
    return int(sum(flags))


def abnormal_attention_count_20(history: pd.DataFrame) -> int:
    count = 0
    tail_start = len(history) - 20
    for idx in range(tail_start, len(history)):
        previous_20 = history["成交额"].iloc[max(0, idx - 20):idx].dropna()
        if len(previous_20) < 20:
            continue
        pct = history["涨跌幅"].iloc[idx]
        amount = history["成交额"].iloc[idx]
        if pd.notna(pct) and pd.notna(amount) and pct >= 5 and amount >= previous_20.mean() * 1.5:
            count += 1
    return count


def build_base_universe(candidates: pd.DataFrame, screen_date: pd.Timestamp, cutoff_date: pd.Timestamp) -> tuple[pd.DataFrame, Diagnostics]:
    diagnostics = Diagnostics(initial_candidates=len(candidates))
    rows: list[dict] = []

    for item in candidates.itertuples(index=False):
        code = item.code
        name = item.name
        row = {
            "code": code,
            "name": name,
            "screen_date": _yyyymmdd(screen_date),
            "data_cutoff_date": _yyyymmdd(cutoff_date),
            "listing_days": 0,
            "close": pd.NA,
            "historical_total_shares": pd.NA,
            "historical_market_cap": pd.NA,
            "avg_amount_20": pd.NA,
            "historical_st_status": "UNKNOWN",
            "st_status_source": "UNKNOWN",
            "passed": False,
            "exclude_reason": "",
            "data_status": "OK",
        }

        history = _history_until_cutoff(code, cutoff_date)
        if history.empty:
            row.update(data_status="DATA_MISSING", exclude_reason="历史行情缺失")
            rows.append(row)
            continue
        diagnostics.history_success += 1

        row["listing_days"] = int(len(history))
        latest = history.iloc[-1]
        close = float(latest["收盘"])
        row["close"] = close
        if len(history) >= 20:
            row["avg_amount_20"] = float(history["成交额"].tail(20).mean())

        shares, share_reason = load_historical_total_shares(code, cutoff_date)
        if shares is None:
            diagnostics.market_cap_missing += 1
            row.update(data_status="DATA_MISSING", exclude_reason=share_reason)
            rows.append(row)
            continue
        diagnostics.share_success += 1
        row["historical_total_shares"] = shares
        row["historical_market_cap"] = close * shares

        reasons = []
        if row["listing_days"] < MIN_LISTING_DAYS:
            reasons.append("上市交易日不足120")
        if not (MIN_PRICE <= close <= MAX_PRICE):
            reasons.append("收盘价不在5-40元")
        if pd.isna(row["avg_amount_20"]) or row["avg_amount_20"] <= MIN_AVG_AMOUNT_20:
            reasons.append("20日平均成交额不足5000万")
        if not (MIN_MARKET_CAP <= row["historical_market_cap"] <= MAX_MARKET_CAP):
            reasons.append("历史总市值不在20-100亿元")

        row["passed"] = not reasons
        row["exclude_reason"] = ";".join(reasons)
        rows.append(row)

    base = pd.DataFrame(rows)
    diagnostics.base_passed = int(base["passed"].sum()) if not base.empty else 0
    return base, diagnostics


def build_popularity_ranking(base: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for item in base[base["passed"]].itertuples(index=False):
        history = _history_until_cutoff(item.code, cutoff_date)
        if len(history) < 80:
            continue
        amount = history["成交额"]
        rows.append({
            "code": item.code,
            "name": item.name,
            "avg_amount_20": float(amount.tail(20).mean()),
            "avg_amount_5": float(amount.tail(5).mean()),
            "avg_amount_60": float(amount.tail(60).mean()),
            "heat_ratio": float(amount.tail(5).mean() / amount.tail(60).mean()) if amount.tail(60).mean() else 0,
            "active_days_20": active_days_20(amount),
            "abnormal_attention_count_20": abnormal_attention_count_20(history),
        })

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    ranking["amount_score"] = ranking["avg_amount_20"].rank(pct=True) * 30
    ranking["heat_score"] = ranking["heat_ratio"].rank(pct=True) * 30
    ranking["active_score"] = ranking["active_days_20"].rank(pct=True) * 20
    ranking["attention_score"] = ranking["abnormal_attention_count_20"].rank(pct=True) * 20
    ranking["popularity_score"] = ranking[["amount_score", "heat_score", "active_score", "attention_score"]].sum(axis=1)
    ranking = ranking.sort_values(["popularity_score", "avg_amount_20"], ascending=[False, False]).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking


def main() -> None:
    screen_date, cutoff_date = resolve_month_dates()
    candidates = load_historical_candidates()
    base, diagnostics = build_base_universe(candidates, screen_date, cutoff_date)
    ranking = build_popularity_ranking(base, cutoff_date)
    pool = ranking.head(50).copy()
    diagnostics.final_pool = len(pool)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base.to_csv(RESULT_DIR / "base_universe.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(RESULT_DIR / "popularity_ranking.csv", index=False, encoding="utf-8-sig")
    pool.to_csv(RESULT_DIR / "monthly_pool.csv", index=False, encoding="utf-8-sig")

    print(f"目标月份: {TARGET_MONTH}")
    print(f"月度第一个交易日: {_yyyymmdd(screen_date)}")
    print(f"数据截止日: {_yyyymmdd(cutoff_date)}")
    print(f"初始候选股票数量: {diagnostics.initial_candidates}")
    print(f"成功获得历史行情数量: {diagnostics.history_success}")
    print(f"成功获得历史股本数量: {diagnostics.share_success}")
    print(f"历史市值缺失数量: {diagnostics.market_cap_missing}")
    print(f"基础资格通过数量: {diagnostics.base_passed}")
    print(f"最终人气池数量: {diagnostics.final_pool}")
    print("Top 10 股票及其 popularity_score:")
    if pool.empty:
        print("无")
    else:
        for item in pool.head(10).itertuples(index=False):
            print(f"{item.rank}. {item.code} {item.name} {item.popularity_score:.2f}")


if __name__ == "__main__":
    main()
