"""Batch, trade-date based market data provider for historical monthly universes.

This module intentionally exposes only all-market-by-date interfaces. It must
not fall back to per-symbol historical requests because that architecture is not
viable for full-market monthly screening.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Callable, Iterable

import pandas as pd


REQUIRED_DAILY_FIELDS = ["trade_date", "code", "close", "pct_chg", "amount"]
REQUIRED_SNAPSHOT_FIELDS = [
    "trade_date",
    "code",
    "name",
    "close",
    "amount",
    "total_market_cap",
]
MIN_REASONABLE_UNIQUE_CODES = 1000


@dataclass
class CacheDiagnostics:
    data_source: str
    requested_trade_dates: list[str] = field(default_factory=list)
    successful_trade_dates: list[str] = field(default_factory=list)
    failed_trade_dates: list[str] = field(default_factory=list)
    cache_hit_dates: list[str] = field(default_factory=list)
    downloaded_dates: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class MarketSnapshotProvider:
    """Abstract provider for historical full-market snapshots by trade date."""

    name = "abstract"

    def fetch_market_daily(self, trade_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_market_snapshot(self, trade_date: str) -> pd.DataFrame:
        raise NotImplementedError

    def load_market_daily(self, trade_date: str) -> pd.DataFrame:
        return self.fetch_market_daily(trade_date)

    def load_market_snapshot(self, trade_date: str) -> pd.DataFrame:
        return self.fetch_market_snapshot(trade_date)


class TushareBatchProvider(MarketSnapshotProvider):
    """Tushare Pro implementation using daily for history and daily_basic only for screen snapshots."""

    name = "tushare_pro_batch"

    def __init__(self, token: str | None = None):
        token = token or os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError(
                "缺少 TUSHARE_TOKEN。请在环境变量中配置 Tushare Pro token；"
                "本模块不会回退到逐股票历史行情请求。"
            )
        import tushare as ts

        ts.set_token(token)
        self.pro = ts.pro_api()
        self._name_cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def _code_from_ts_code(ts_code: str) -> str:
        return str(ts_code).split(".", 1)[0].zfill(6)

    def fetch_market_daily(self, trade_date: str) -> pd.DataFrame:
        """Fetch ordinary all-market daily bars by trade date.

        This intentionally calls only pro.daily. It must not request daily_basic
        or namechange for historical lookback dates.
        """
        date = normalize_trade_date(trade_date)
        daily = self.pro.daily(trade_date=date)
        if daily is None or daily.empty:
            raise RuntimeError(f"Tushare daily 在 {date} 未返回全市场数据。")
        return self._normalize_daily(daily, date)

    def fetch_market_snapshot(self, trade_date: str) -> pd.DataFrame:
        """Fetch screen-date all-market snapshot with market cap and historical ST status."""
        date = normalize_trade_date(trade_date)
        daily = self.pro.daily(trade_date=date)
        basic = self.pro.daily_basic(
            trade_date=date,
            fields="ts_code,trade_date,total_mv,circ_mv,turnover_rate,volume_ratio",
        )
        if daily is None or daily.empty:
            raise RuntimeError(f"Tushare daily 在 {date} 未返回全市场数据。")
        if basic is None or basic.empty:
            raise RuntimeError(f"Tushare daily_basic 在 {date} 未返回全市场市值数据。")

        df = self._normalize_daily(daily, date)
        df = df.merge(basic[["ts_code", "total_mv"]], on="ts_code", how="left")
        df["total_market_cap"] = pd.to_numeric(df["total_mv"], errors="coerce") * 10000
        names = self._names_as_of_screen_date(date)
        df = df.merge(names, on="ts_code", how="left")
        df["historical_st_status"] = df["name"].map(classify_historical_st_status)
        df["name"] = df["name"].fillna("")
        df["st_status_source"] = df["historical_st_status"].map(
            lambda status: "UNKNOWN" if status == "UNKNOWN" else "TUSHARE_NAMECHANGE_AS_OF_SCREEN_DATE"
        )
        return df[[
            "trade_date", "code", "name", "close", "pct_chg", "amount",
            "total_market_cap", "historical_st_status", "st_status_source",
        ]]

    def _normalize_daily(self, daily: pd.DataFrame, date: str) -> pd.DataFrame:
        df = daily.copy()
        df["code"] = df["ts_code"].map(self._code_from_ts_code)
        df["trade_date"] = date
        # Tushare daily amount is 千元; normalize to 元.
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
        return df[["ts_code", "trade_date", "code", "close", "pct_chg", "amount"]]

    def _names_as_of_screen_date(self, date: str) -> pd.DataFrame:
        if date not in self._name_cache:
            names = self.pro.namechange(
                start_date="19900101",
                end_date=date,
                fields="ts_code,name,start_date,end_date,change_reason",
            )
            self._name_cache[date] = self._names_as_of(names, date)
        return self._name_cache[date].copy()

    @staticmethod
    def _names_as_of(names: pd.DataFrame, date: str) -> pd.DataFrame:
        if names is None or names.empty:
            return pd.DataFrame(columns=["ts_code", "name"])
        data = names.copy()
        data["start_date"] = pd.to_datetime(data["start_date"], errors="coerce")
        data["end_date"] = pd.to_datetime(data["end_date"], errors="coerce")
        cutoff = pd.to_datetime(date)
        active = data[(data["start_date"].isna() | (data["start_date"] <= cutoff)) & (data["end_date"].isna() | (data["end_date"] >= cutoff))]
        active = active.sort_values(["ts_code", "start_date"]).drop_duplicates("ts_code", keep="last")
        return active[["ts_code", "name"]]


def classify_historical_st_status(name: object) -> str:
    if pd.isna(name) or str(name).strip() == "":
        return "UNKNOWN"
    return "ST" if "ST" in str(name).upper() else "NON_ST"


def normalize_trade_date(trade_date: str | pd.Timestamp) -> str:
    return pd.to_datetime(trade_date).strftime("%Y%m%d")


def cache_path(cache_dir: Path, trade_date: str) -> Path:
    return cache_dir / f"{normalize_trade_date(trade_date)}.csv"


def validate_market_daily(df: pd.DataFrame, trade_date: str, *, require_snapshot: bool = False, min_unique_codes: int = MIN_REASONABLE_UNIQUE_CODES) -> None:
    date = normalize_trade_date(trade_date)
    required = REQUIRED_SNAPSHOT_FIELDS if require_snapshot else REQUIRED_DAILY_FIELDS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{date} 缺少必需字段: {missing}")
    if df.empty:
        raise ValueError(f"{date} 全市场数据为空。")
    actual_dates = set(pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d").dropna())
    if actual_dates != {date}:
        raise ValueError(f"{date} 缓存 trade_date 不一致: {sorted(actual_dates)}")
    if df["code"].isna().any() or (df["code"].astype(str).str.strip() == "").any():
        raise ValueError(f"{date} 存在空 code。")
    unique_codes = df["code"].nunique()
    if unique_codes < min_unique_codes:
        raise ValueError(f"{date} code 去重数量异常: {unique_codes}")
    for col in ["close", "amount"] + (["total_market_cap"] if require_snapshot else []):
        if pd.to_numeric(df[col], errors="coerce").notna().sum() == 0:
            raise ValueError(f"{date} 字段 {col} 无有效数值。")


def _load_cached_or_fetch(
    fetcher: Callable[[str], pd.DataFrame],
    trade_date: str,
    cache_dir: Path,
    diagnostics: CacheDiagnostics,
    *,
    require_snapshot: bool,
    min_unique_codes: int,
) -> pd.DataFrame:
    date = normalize_trade_date(trade_date)
    diagnostics.requested_trade_dates.append(date)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, date)
    try:
        if path.exists():
            cached = pd.read_csv(path, dtype={"code": str, "trade_date": str})
            validate_market_daily(cached, date, require_snapshot=require_snapshot, min_unique_codes=min_unique_codes)
            diagnostics.cache_hit_dates.append(date)
            diagnostics.successful_trade_dates.append(date)
            return cached

        df = fetcher(date)
        validate_market_daily(df, date, require_snapshot=require_snapshot, min_unique_codes=min_unique_codes)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        diagnostics.downloaded_dates.append(date)
        diagnostics.successful_trade_dates.append(date)
        return pd.read_csv(path, dtype={"code": str, "trade_date": str})
    except Exception as exc:
        diagnostics.failed_trade_dates.append(date)
        diagnostics.messages.append(f"{date}: {exc}")
        raise


def load_cached_or_fetch_market_daily(provider: MarketSnapshotProvider, trade_date: str, cache_dir: Path, diagnostics: CacheDiagnostics, *, min_unique_codes: int = MIN_REASONABLE_UNIQUE_CODES) -> pd.DataFrame:
    return _load_cached_or_fetch(
        provider.load_market_daily,
        trade_date,
        cache_dir,
        diagnostics,
        require_snapshot=False,
        min_unique_codes=min_unique_codes,
    )


def load_cached_or_fetch_market_snapshot(provider: MarketSnapshotProvider, trade_date: str, cache_dir: Path, diagnostics: CacheDiagnostics, *, min_unique_codes: int = MIN_REASONABLE_UNIQUE_CODES) -> pd.DataFrame:
    return _load_cached_or_fetch(
        provider.load_market_snapshot,
        trade_date,
        cache_dir,
        diagnostics,
        require_snapshot=True,
        min_unique_codes=min_unique_codes,
    )


def load_many_market_daily(provider: MarketSnapshotProvider, trade_dates: Iterable[str], cache_dir: Path, diagnostics: CacheDiagnostics, *, min_unique_codes: int = MIN_REASONABLE_UNIQUE_CODES) -> pd.DataFrame:
    frames = [load_cached_or_fetch_market_daily(provider, d, cache_dir, diagnostics, min_unique_codes=min_unique_codes) for d in trade_dates]
    if not frames:
        return pd.DataFrame(columns=REQUIRED_DAILY_FIELDS)
    return pd.concat(frames, ignore_index=True)
