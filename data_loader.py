"""数据拉取与 CSV 缓存。只做历史回测，不连接券商。"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

from config import DATA_CACHE_DIR, HISTORY_DIR, MAIN_BOARD_PREFIXES


def ensure_dirs() -> None:
    """创建缓存目录。"""
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def retry_call(func, retries: int = 3, delay: float = 1.5, **kwargs):
    """失败自动重试；最终失败返回 None，避免程序崩溃。"""
    for attempt in range(1, retries + 1):
        try:
            return func(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 数据源异常类型不稳定，统一兜底
            print(f"[WARN] 第 {attempt}/{retries} 次请求失败：{exc}")
            if attempt < retries:
                time.sleep(delay * attempt)
    return None


def normalize_code(code: str) -> str:
    """将股票代码统一为 6 位字符串。"""
    return str(code).zfill(6)


def load_realtime_quotes(force_refresh: bool = False) -> pd.DataFrame:
    """拉取 A 股实时行情，并缓存到 data_cache/realtime_quotes.csv。"""
    ensure_dirs()
    cache_file = DATA_CACHE_DIR / "realtime_quotes.csv"
    if cache_file.exists() and not force_refresh:
        return pd.read_csv(cache_file, dtype={"代码": str})

    df = retry_call(ak.stock_zh_a_spot_em)
    if df is None or df.empty:
        print("[WARN] 实时行情拉取失败，尝试读取旧缓存。")
        if cache_file.exists():
            return pd.read_csv(cache_file, dtype={"代码": str})
        return pd.DataFrame()

    df["代码"] = df["代码"].map(normalize_code)
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df


def get_history_path(code: str) -> Path:
    """返回单只股票历史 CSV 路径。"""
    return HISTORY_DIR / f"{normalize_code(code)}.csv"


def load_stock_history(code: str, start_date: str, end_date: Optional[str] = None, adjust: str = "qfq") -> pd.DataFrame:
    """读取或拉取单只股票历史日线，每只股票一个 CSV。"""
    ensure_dirs()
    code = normalize_code(code)
    cache_file = get_history_path(code)
    if cache_file.exists():
        df = pd.read_csv(cache_file, dtype={"股票代码": str}, parse_dates=["日期"])
    else:
        end = end_date or datetime.now().strftime("%Y%m%d")
        df = retry_call(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end,
            adjust=adjust,
        )
        if df is None or df.empty:
            print(f"[WARN] {code} 历史数据拉取失败，跳过。")
            return pd.DataFrame()
        df["股票代码"] = code
        df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        df = pd.read_csv(cache_file, dtype={"股票代码": str}, parse_dates=["日期"])

    df = df.sort_values("日期").reset_index(drop=True)
    return df


def build_candidate_universe(quotes: pd.DataFrame, top_n: Optional[int]) -> pd.DataFrame:
    """按主板代码、ST/退市、价格、市值和成交额排序筛选初始股票池。"""
    if quotes.empty:
        return quotes

    df = quotes.copy()
    df["代码"] = df["代码"].map(normalize_code)
    df = df[df["代码"].str.startswith(MAIN_BOARD_PREFIXES)]
    df = df[~df["名称"].astype(str).str.contains("ST|退", regex=True, na=False)]

    # AKShare 东方财富实时行情常见字段：最新价、成交额、总市值；转数值后过滤。
    for col in ["最新价", "成交额", "总市值"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["最新价", "成交额", "总市值"])

    from config import MAX_MARKET_CAP, MAX_PRICE, MIN_MARKET_CAP, MIN_PRICE

    df = df[
        (df["总市值"].between(MIN_MARKET_CAP, MAX_MARKET_CAP))
        & (df["最新价"].between(MIN_PRICE, MAX_PRICE))
        & (df["成交额"] > 0)
    ]
    df = df.sort_values("成交额", ascending=False)
    if top_n:
        df = df.head(top_n)
    return df.reset_index(drop=True)
