"""2023-01 historical monthly universe built from batch daily snapshots only."""
from __future__ import annotations

import argparse
import akshare as ak
import pandas as pd

from config import BASE_DIR, MAIN_BOARD_PREFIXES, MAX_MARKET_CAP, MAX_PRICE, MIN_AVG_AMOUNT_20, MIN_LISTING_DAYS, MIN_MARKET_CAP, MIN_PRICE
from data_loader import retry_call
from market_snapshot_provider import CacheDiagnostics, TushareBatchProvider, load_cached_or_fetch_market_daily, load_cached_or_fetch_market_snapshot, load_many_market_daily

TARGET_MONTH = "2023-01"
RESULT_DIR = BASE_DIR / "monthly_universe_results" / "2023_01"
MARKET_DAILY_CACHE_DIR = BASE_DIR / "data_cache" / "market_daily"
MARKET_SNAPSHOT_CACHE_DIR = BASE_DIR / "data_cache" / "market_snapshot"
LOOKBACK_TRADING_DAYS = 140


def yyyymmdd(ts: pd.Timestamp | str) -> str:
    return pd.to_datetime(ts).strftime("%Y%m%d")


def get_trade_dates() -> pd.DatetimeIndex:
    dates = retry_call(ak.tool_trade_date_hist_sina)
    if dates is None or dates.empty:
        raise RuntimeError("无法获取交易日历，不能确定 screen_date/effective_date。")
    date_col = "trade_date" if "trade_date" in dates.columns else dates.columns[0]
    values = pd.to_datetime(dates[date_col], errors="coerce").dropna().sort_values().unique()
    return pd.DatetimeIndex(values)


def resolve_month_dates_from_calendar(trade_dates: pd.DatetimeIndex, target_month: str = TARGET_MONTH) -> tuple[pd.Timestamp, pd.Timestamp]:
    month_start = pd.Timestamp(f"{target_month}-01")
    next_month = month_start + pd.offsets.MonthBegin(1)
    month_dates = trade_dates[(trade_dates >= month_start) & (trade_dates < next_month)]
    if month_dates.empty:
        raise RuntimeError(f"{target_month} 没有可识别交易日。")
    screen_date = pd.Timestamp(month_dates[0])
    later = trade_dates[trade_dates > screen_date]
    if later.empty:
        raise RuntimeError(f"无法找到 {screen_date.date()} 后的下一个实际交易日。")
    return screen_date, pd.Timestamp(later[0])


def resolve_month_dates(target_month: str = TARGET_MONTH) -> tuple[pd.Timestamp, pd.Timestamp]:
    return resolve_month_dates_from_calendar(get_trade_dates(), target_month)


def required_history_dates(trade_dates: pd.DatetimeIndex, screen_date: pd.Timestamp, lookback: int = LOOKBACK_TRADING_DAYS) -> list[str]:
    upto = trade_dates[trade_dates <= screen_date]
    if len(upto) < lookback + 1:
        raise RuntimeError(f"截至 {yyyymmdd(screen_date)} 的交易日不足 {lookback + 1} 个。")
    return [yyyymmdd(d) for d in upto[-(lookback + 1):]]


def is_main_board(code: str) -> bool:
    return str(code).zfill(6).startswith(MAIN_BOARD_PREFIXES)


def add_history_metrics(long_df: pd.DataFrame, screen_date: pd.Timestamp) -> pd.DataFrame:
    data = long_df.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], format="%Y%m%d")
    data = data[data["trade_date"] <= screen_date].sort_values(["code", "trade_date"])

    rows: list[dict] = []
    for code, history in data.groupby("code", sort=False):
        history = history.sort_values("trade_date").reset_index(drop=True)
        if history.empty or history.iloc[-1]["trade_date"] != screen_date:
            continue
        amount = pd.to_numeric(history["amount"], errors="coerce")
        pct = pd.to_numeric(history["pct_chg"], errors="coerce")
        active = 0
        abnormal = 0
        for idx in range(max(0, len(history) - 20), len(history)):
            prev60 = amount.iloc[max(0, idx - 60):idx].dropna()
            if len(prev60) >= 20 and amount.iloc[idx] > prev60.median():
                active += 1
            prev20 = amount.iloc[max(0, idx - 20):idx].dropna()
            if len(prev20) >= 20 and pd.notna(pct.iloc[idx]) and pct.iloc[idx] >= 5 and amount.iloc[idx] >= prev20.mean() * 1.5:
                abnormal += 1
        avg60 = amount.tail(60).mean()
        rows.append({
            "code": str(code).zfill(6),
            "listing_days": int(len(history)),
            "avg_amount_5": float(amount.tail(5).mean()),
            "avg_amount_20": float(amount.tail(20).mean()),
            "avg_amount_60": float(avg60),
            "heat_ratio": float(amount.tail(5).mean() / avg60) if pd.notna(avg60) and avg60 else 0.0,
            "active_days_20": int(active),
            "abnormal_attention_count_20": int(abnormal),
        })
    return pd.DataFrame(rows)


def build_base_universe(snapshot: pd.DataFrame, metrics: pd.DataFrame, screen_date: pd.Timestamp, effective_date: pd.Timestamp) -> pd.DataFrame:
    screen = snapshot.copy()
    screen["code"] = screen["code"].astype(str).str.zfill(6)
    screen = screen[screen["code"].map(is_main_board)].merge(metrics, on="code", how="left")
    if "historical_st_status" not in screen.columns:
        screen["historical_st_status"] = "UNKNOWN"
        screen["st_status_source"] = "UNKNOWN"
    if "st_status_source" not in screen.columns:
        screen["st_status_source"] = "UNKNOWN"

    reasons = []
    for row in screen.itertuples(index=False):
        item_reasons = []
        if not (MIN_PRICE <= row.close <= MAX_PRICE):
            item_reasons.append("收盘价不在5-40元")
        if not (MIN_MARKET_CAP <= row.total_market_cap <= MAX_MARKET_CAP):
            item_reasons.append("历史总市值不在20-100亿元")
        if pd.isna(row.listing_days) or row.listing_days < MIN_LISTING_DAYS:
            item_reasons.append("截至screen_date历史不足120个交易日")
        if pd.isna(row.avg_amount_20) or row.avg_amount_20 <= MIN_AVG_AMOUNT_20:
            item_reasons.append("20日平均成交额不足5000万")
        if getattr(row, "historical_st_status", "UNKNOWN") == "ST":
            item_reasons.append("screen_date历史ST状态")
        reasons.append(";".join(item_reasons))

    screen["screen_date"] = yyyymmdd(screen_date)
    screen["effective_date"] = yyyymmdd(effective_date)
    screen["historical_market_cap"] = screen["total_market_cap"]
    screen["exclude_reason"] = reasons
    screen["passed"] = screen["exclude_reason"] == ""
    cols = ["screen_date", "effective_date", "code", "name", "close", "historical_market_cap", "listing_days", "avg_amount_20", "historical_st_status", "st_status_source", "passed", "exclude_reason"]
    return screen[cols].sort_values(["passed", "code"], ascending=[False, True]).reset_index(drop=True)


def build_popularity_ranking(base: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    ranking = base[base["passed"]][["code", "name", "historical_market_cap"]].merge(metrics, on="code", how="inner")
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


def diagnostics_frame(diagnostics: CacheDiagnostics, **extra: object) -> pd.DataFrame:
    row = {
        "data_source": diagnostics.data_source,
        "requested_trade_dates": ",".join(diagnostics.requested_trade_dates),
        "successful_trade_dates": ",".join(diagnostics.successful_trade_dates),
        "failed_trade_dates": ",".join(diagnostics.failed_trade_dates),
        "cache_hit_dates": ",".join(diagnostics.cache_hit_dates),
        "downloaded_dates": ",".join(diagnostics.downloaded_dates),
        "messages": " | ".join(diagnostics.messages),
    }
    row.update(extra)
    return pd.DataFrame([row])


def save_diagnostics(diagnostics: CacheDiagnostics, **extra: object) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics_frame(diagnostics, **extra).to_csv(RESULT_DIR / "data_diagnostics.csv", index=False, encoding="utf-8-sig")


def _print_probe_frame(provider_name: str, trade_date: str, data_type: str, df: pd.DataFrame) -> None:
    print(f"数据源: {provider_name}")
    print(f"日期: {trade_date}")
    print(f"数据类型: {data_type}")
    print(f"行数: {len(df)}")
    print(f"code去重数: {df['code'].nunique()}")
    print(f"字段列表: {list(df.columns)}")
    print(df.head(5).to_string(index=False))
    if data_type == "snapshot":
        counts = df.get("historical_st_status", pd.Series(dtype=object)).value_counts(dropna=False)
        print(f"历史ST状态 ST数量: {int(counts.get('ST', 0))}")
        print(f"历史ST状态 NON_ST数量: {int(counts.get('NON_ST', 0))}")
        print(f"历史ST状态 UNKNOWN数量: {int(counts.get('UNKNOWN', 0))}")


def print_probe(provider: TushareBatchProvider) -> None:
    trade_dates = get_trade_dates()
    screen_date, effective_date = resolve_month_dates_from_calendar(trade_dates)
    history_dates = required_history_dates(trade_dates, screen_date, 2)[:-1]
    diagnostics = CacheDiagnostics(provider.name)
    print(f"screen_date: {yyyymmdd(screen_date)}")
    print(f"effective_date: {yyyymmdd(effective_date)}")
    for date in history_dates:
        df = load_cached_or_fetch_market_daily(provider, date, MARKET_DAILY_CACHE_DIR, diagnostics)
        _print_probe_frame(provider.name, date, "daily", df)
    snapshot = load_cached_or_fetch_market_snapshot(provider, yyyymmdd(screen_date), MARKET_SNAPSHOT_CACHE_DIR, diagnostics)
    _print_probe_frame(provider.name, yyyymmdd(screen_date), "snapshot", snapshot)


def run_full(provider: TushareBatchProvider) -> None:
    trade_dates = get_trade_dates()
    screen_date, effective_date = resolve_month_dates_from_calendar(trade_dates)
    needed_dates = required_history_dates(trade_dates, screen_date)
    diagnostics = CacheDiagnostics(provider.name)
    base = ranking = pool = pd.DataFrame()
    snapshot = pd.DataFrame()
    try:
        long_df = load_many_market_daily(provider, needed_dates, MARKET_DAILY_CACHE_DIR, diagnostics)
        snapshot = load_cached_or_fetch_market_snapshot(provider, yyyymmdd(screen_date), MARKET_SNAPSHOT_CACHE_DIR, diagnostics)
        metrics = add_history_metrics(long_df, screen_date)
        base = build_base_universe(snapshot, metrics, screen_date, effective_date)
        ranking = build_popularity_ranking(base, metrics)
        pool = ranking.head(50).copy()

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        base.to_csv(RESULT_DIR / "base_universe.csv", index=False, encoding="utf-8-sig")
        ranking.to_csv(RESULT_DIR / "popularity_ranking.csv", index=False, encoding="utf-8-sig")
        pool.to_csv(RESULT_DIR / "monthly_pool.csv", index=False, encoding="utf-8-sig")
        save_diagnostics(
            diagnostics,
            screen_date=yyyymmdd(screen_date),
            effective_date=yyyymmdd(effective_date),
            snapshot_rows=len(snapshot),
            snapshot_unique_codes=snapshot["code"].nunique() if "code" in snapshot else 0,
            base_passed=int(base["passed"].sum()) if not base.empty else 0,
            final_pool_count=len(pool),
        )
    except Exception:
        save_diagnostics(
            diagnostics,
            screen_date=yyyymmdd(screen_date),
            effective_date=yyyymmdd(effective_date),
            snapshot_rows=len(snapshot),
            snapshot_unique_codes=snapshot["code"].nunique() if "code" in snapshot else 0,
            base_passed=int(base["passed"].sum()) if not base.empty and "passed" in base else 0,
            final_pool_count=len(pool),
        )
        raise

    print(f"目标月份: {TARGET_MONTH}")
    print(f"screen_date: {yyyymmdd(screen_date)}")
    print(f"effective_date: {yyyymmdd(effective_date)}")
    print(f"数据源: {provider.name}")
    print(f"需要的交易日数量: {len(needed_dates)}")
    print(f"成功日期数量: {len(set(diagnostics.successful_trade_dates))}")
    print(f"失败日期数量: {len(set(diagnostics.failed_trade_dates))}")
    print(f"缓存命中数量: {len(set(diagnostics.cache_hit_dates))}")
    print(f"新下载数量: {len(set(diagnostics.downloaded_dates))}")
    print(f"screen_date 全市场股票数量: {snapshot['code'].nunique()}")
    print(f"基础池通过数量: {int(base['passed'].sum())}")
    print(f"人气排名股票数量: {len(ranking)}")
    print(f"最终 Top 50 数量: {len(pool)}")
    print("Top 10:")
    cols = ["rank", "code", "name", "historical_market_cap", "avg_amount_20", "heat_ratio", "active_days_20", "abnormal_attention_count_20", "popularity_score"]
    print("无" if pool.empty else pool[cols].head(10).to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="只快速探测批量数据源与日期缓存，不构建股票池。")
    args = parser.parse_args()
    provider = TushareBatchProvider()
    if args.probe:
        print_probe(provider)
    else:
        run_full(provider)


if __name__ == "__main__":
    main()
