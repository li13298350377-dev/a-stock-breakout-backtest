"""BaoStock enrichment for screen-date ST status and published totalShare."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Iterable

import pandas as pd

from baostock_raw_cache import (
    BaoStockRawCache,
    has_profit_quarter,
    has_st_year,
    load_profit_quarter,
    lookup_st_raw,
    store_profit_quarter,
    store_st_year,
)

SHARE_SOURCE = "BAOSTOCK_QUERY_PROFIT_DATA"
SHARE_UNKNOWN = "TOTAL_SHARE_UNKNOWN"
ST_SOURCE = "BAOSTOCK_ISST_SCREEN_DATE"
MARKET_CAP_METHOD = "CLOSE_X_LATEST_PUBLISHED_QUARTER_TOTAL_SHARE"
FETCH_SUCCESS = "SUCCESS"
FETCH_DATA_UNKNOWN = "DATA_UNKNOWN"
FETCH_REQUEST_FAILED_RETRYABLE = "REQUEST_FAILED_RETRYABLE"


@dataclass
class BaoStockDiagnostics:
    requested_count: int = 0
    cache_hit_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    name_map_rows: int = 0
    name_map_unique_codes: int = 0
    name_map_non_empty_names: int = 0
    name_map_failed: bool = False
    messages: list[str] = field(default_factory=list)


def to_baostock_code(code: str) -> str:
    c = str(code).zfill(6)
    return ("sh." if c.startswith(("600", "601", "603", "605")) else "sz.") + c


def normalize_st_status(value: object) -> str:
    if str(value).strip() == "1":
        return "ST"
    if str(value).strip() == "0":
        return "NON_ST"
    return "UNKNOWN"


def quarter_cursor(screen_date: str | pd.Timestamp) -> list[tuple[int, int]]:
    dt = pd.to_datetime(screen_date)
    q = ((dt.month - 1) // 3) + 1
    year = int(dt.year)
    out = []
    for _ in range(8):
        out.append((year, q))
        q -= 1
        if q == 0:
            q = 4
            year -= 1
    return out


def select_latest_published_total_share(records: pd.DataFrame, screen_date: str | pd.Timestamp) -> dict:
    if records is None or records.empty:
        return {"share_pub_date": "", "share_stat_date": "", "total_share": float("nan"), "share_source": SHARE_UNKNOWN}
    df = records.copy()
    df["pubDate_dt"] = pd.to_datetime(df.get("pubDate"), errors="coerce")
    df["statDate_dt"] = pd.to_datetime(df.get("statDate"), errors="coerce")
    cutoff = pd.to_datetime(screen_date)
    df["total_share_num"] = pd.to_numeric(df.get("totalShare"), errors="coerce")

    valid = df[
        (df["pubDate_dt"].notna())
        & (df["pubDate_dt"] <= cutoff)
        & (df["total_share_num"] > 0)
    ].copy()

    if valid.empty:
        return {
            "share_pub_date": "",
            "share_stat_date": "",
            "total_share": float("nan"),
            "share_source": SHARE_UNKNOWN,
        }

    # Point-in-time rule:
    # 1. pubDate <= screen_date controls data availability;
    # 2. among available records, choose the latest reporting period statDate;
    # 3. pubDate is only the tie-breaker for duplicate statDate records.
    with_stat_date = valid[valid["statDate_dt"].notna()]

    if not with_stat_date.empty:
        row = with_stat_date.sort_values(
            ["statDate_dt", "pubDate_dt"]
        ).iloc[-1]
    else:
        row = valid.sort_values("pubDate_dt").iloc[-1]
    return {
        "share_pub_date": str(row.get("pubDate", "")),
        "share_stat_date": str(row.get("statDate", "")),
        "total_share": float(row["total_share_num"]),
        "share_source": SHARE_SOURCE,
    }


class BaoStockScreenProvider:
    name = "baostock_screen_enrichment"

    def __init__(
        self,
        retry: int = 2,
        request_interval: float = 0.05,
        raw_cache: BaoStockRawCache | None = None,
    ):
        self.retry = retry
        self.request_interval = request_interval
        self.raw_cache = raw_cache
        self.bs = None

    def __enter__(self):
        import baostock as bs
        self.bs = bs
        rs = bs.login()
        if getattr(rs, "error_code", "0") != "0":
            raise RuntimeError(f"BaoStock login failed: {getattr(rs, 'error_msg', '')}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.bs is not None:
            self.bs.logout()

    @staticmethod
    def _rs_to_df(rs) -> pd.DataFrame:
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        return pd.DataFrame(rows, columns=rs.fields) if rows else pd.DataFrame(columns=getattr(rs, "fields", []))

    def _call(self, func, *args, **kwargs) -> pd.DataFrame:
        last = None
        for _ in range(self.retry + 1):
            try:
                rs = func(*args, **kwargs)
                if getattr(rs, "error_code", "0") == "0":
                    result = self._rs_to_df(rs)
                    time.sleep(self.request_interval)
                    return result
                last = RuntimeError(getattr(rs, "error_msg", "BaoStock error"))
            except Exception as exc:
                last = exc
            time.sleep(self.request_interval)
        raise last or RuntimeError("BaoStock call failed")

    def fetch_name_map(self, screen_date: str) -> pd.DataFrame:
        if not hasattr(self.bs, "query_all_stock"):
            return pd.DataFrame(columns=["code", "name"])
        df = self._call(self.bs.query_all_stock, day=pd.to_datetime(screen_date).strftime("%Y-%m-%d"))
        if df.empty or "code" not in df:
            return pd.DataFrame(columns=["code", "name"])
        name_col = "code_name" if "code_name" in df.columns else ("name" if "name" in df.columns else None)
        if name_col is None:
            return pd.DataFrame(columns=["code", "name"])
        out = df[["code", name_col]].rename(columns={name_col: "name"})
        out["code"] = out["code"].astype(str).str[-6:].str.zfill(6)
        return out.drop_duplicates("code")

    def _fetch_one_without_raw_cache(
        self,
        code: str,
        screen_date: str,
        name_map: dict[str, str] | None = None,
    ) -> dict:
        bs_code = to_baostock_code(code)
        date_dash = pd.to_datetime(screen_date).strftime("%Y-%m-%d")

        st_df = self._call(
            self.bs.query_history_k_data_plus,
            bs_code,
            "date,code,close,isST",
            start_date=date_dash,
            end_date=date_dash,
            frequency="d",
            adjustflag="3",
        )

        st = normalize_st_status(
            st_df.iloc[0].get("isST")
            if not st_df.empty
            else None
        )

        share = select_latest_published_total_share(
            pd.DataFrame(),
            screen_date,
        )

        for y, q in quarter_cursor(screen_date):
            quarter_df = self._call(
                self.bs.query_profit_data,
                code=bs_code,
                year=y,
                quarter=q,
            )

            candidate_share = (
                select_latest_published_total_share(
                    quarter_df,
                    screen_date,
                )
            )

            if pd.notna(
                candidate_share.get("total_share")
            ):
                share = candidate_share
                break

        name = (
            name_map or {}
        ).get(
            str(code).zfill(6),
            "",
        )

        fetch_status = (
            FETCH_SUCCESS
            if pd.notna(share.get("total_share"))
            else FETCH_DATA_UNKNOWN
        )

        return {
            "code": str(code).zfill(6),
            "name": (
                name
                or str(code).zfill(6)
            ),
            "name_source": (
                "BAOSTOCK_QUERY_ALL_STOCK"
                if name
                else "CODE_FALLBACK"
            ),
            "historical_st_status": st,
            "st_status_source": (
                ST_SOURCE
                if st != "UNKNOWN"
                else "UNKNOWN"
            ),
            **share,
            "fetch_status": fetch_status,
        }


    def _fetch_one_with_raw_cache(
        self,
        code: str,
        screen_date: str,
        name_map: dict[str, str] | None = None,
    ) -> dict:
        if self.raw_cache is None:
            raise RuntimeError(
                "raw cache is not configured"
            )

        normalized_code = str(code).zfill(6)
        bs_code = to_baostock_code(normalized_code)

        screen_dt = pd.to_datetime(screen_date)
        screen_year = int(screen_dt.year)

        bundle = self.raw_cache.load(
            normalized_code
        )

        dirty = False

        try:
            # ==========================================
            # ST raw cache: one successful range query
            # per stock per calendar year.
            # ==========================================

            if not has_st_year(
                bundle,
                screen_year,
            ):
                st_df = self._call(
                    self.bs.query_history_k_data_plus,
                    bs_code,
                    "date,code,isST",
                    start_date=f"{screen_year}-01-01",
                    end_date=f"{screen_year}-12-31",
                    frequency="d",
                    adjustflag="3",
                )

                store_st_year(
                    bundle,
                    screen_year,
                    st_df,
                )

                dirty = True

            st_raw = lookup_st_raw(
                bundle,
                screen_date,
            )

            st = normalize_st_status(
                st_raw
            )


            # ==========================================
            # totalShare raw cache:
            # reuse successful quarter queries across
            # all monthly screen dates.
            # ==========================================

            share = select_latest_published_total_share(
                pd.DataFrame(),
                screen_date,
            )

            for year, quarter in quarter_cursor(
                screen_date
            ):
                if has_profit_quarter(
                    bundle,
                    year,
                    quarter,
                ):
                    quarter_df = load_profit_quarter(
                        bundle,
                        year,
                        quarter,
                    )

                else:
                    quarter_df = self._call(
                        self.bs.query_profit_data,
                        code=bs_code,
                        year=year,
                        quarter=quarter,
                    )

                    store_profit_quarter(
                        bundle,
                        year,
                        quarter,
                        quarter_df,
                    )

                    dirty = True

                candidate_share = (
                    select_latest_published_total_share(
                        quarter_df,
                        screen_date,
                    )
                )

                if pd.notna(
                    candidate_share.get(
                        "total_share"
                    )
                ):
                    share = candidate_share
                    break


            name = (
                name_map or {}
            ).get(
                normalized_code,
                "",
            )

            fetch_status = (
                FETCH_SUCCESS
                if pd.notna(
                    share.get("total_share")
                )
                else FETCH_DATA_UNKNOWN
            )

            return {
                "code": normalized_code,
                "name": (
                    name
                    or normalized_code
                ),
                "name_source": (
                    "BAOSTOCK_QUERY_ALL_STOCK"
                    if name
                    else "CODE_FALLBACK"
                ),
                "historical_st_status": st,
                "st_status_source": (
                    ST_SOURCE
                    if st != "UNKNOWN"
                    else "UNKNOWN"
                ),
                **share,
                "fetch_status": fetch_status,
            }

        finally:
            # Preserve partial successful network work
            # even if a later request for this stock fails.
            if dirty:
                self.raw_cache.save(
                    normalized_code,
                    bundle,
                )


    def fetch_one(
        self,
        code: str,
        screen_date: str,
        name_map: dict[str, str] | None = None,
    ) -> dict:
        if self.raw_cache is None:
            return self._fetch_one_without_raw_cache(
                code,
                screen_date,
                name_map,
            )

        return self._fetch_one_with_raw_cache(
            code,
            screen_date,
            name_map,
        )


def load_or_fetch_enrichment(candidates: Iterable[str], screen_date: str, cache_dir: Path, provider: BaoStockScreenProvider | None = None, diagnostics: BaoStockDiagnostics | None = None) -> pd.DataFrame:
    diagnostics = diagnostics or BaoStockDiagnostics()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pd.to_datetime(screen_date).strftime('%Y%m%d')}.csv"
    cols = ["code", "name", "name_source", "historical_st_status", "st_status_source", "share_pub_date", "share_stat_date", "total_share", "share_source", "fetch_status"]
    existing = pd.read_csv(path, dtype={"code": str}) if path.exists() else pd.DataFrame(columns=cols)
    existing["code"] = existing.get("code", pd.Series(dtype=str)).astype(str).str.zfill(6)
    if "fetch_status" not in existing.columns:
        completed = (existing.get("share_source", pd.Series(dtype=object)) == SHARE_SOURCE) | (existing.get("st_status_source", pd.Series(dtype=object)) == ST_SOURCE)
        existing["fetch_status"] = completed.map(lambda ok: FETCH_SUCCESS if ok else FETCH_REQUEST_FAILED_RETRYABLE)
    for col in cols:
        if col not in existing.columns:
            existing[col] = "" if col != "total_share" else float("nan")
    wanted = [str(c).zfill(6) for c in candidates]
    completed_statuses = {FETCH_SUCCESS, FETCH_DATA_UNKNOWN}
    done_rows = existing[existing["fetch_status"].isin(completed_statuses)]
    done = set(done_rows["code"])
    diagnostics.cache_hit_count += sum(1 for c in wanted if c in done)
    todo = [c for c in wanted if c not in done]
    diagnostics.requested_count += len(todo)
    rows = []
    if not todo:
        return existing[existing["code"].isin(wanted)][cols]
    own = provider is None
    provider = provider or BaoStockScreenProvider(
        raw_cache=BaoStockRawCache(
            cache_dir.parent / "baostock_raw_cache"
        )
    )
    ctx = provider if not own else provider.__enter__()
    try:
        try:
            name_map_df = ctx.fetch_name_map(screen_date)
            diagnostics.name_map_rows = len(name_map_df)
            diagnostics.name_map_unique_codes = int(name_map_df["code"].nunique()) if "code" in name_map_df else 0
            diagnostics.name_map_non_empty_names = int(name_map_df.get("name", pd.Series(dtype=object)).astype(str).str.strip().ne("").sum()) if "name" in name_map_df else 0
            name_map = dict(zip(name_map_df.get("code", []), name_map_df.get("name", [])))
        except Exception as exc:
            diagnostics.name_map_failed = True
            diagnostics.messages.append(f"name_map: {exc}")
            name_map = {}
        for code in todo:
            try:
                row = ctx.fetch_one(code, screen_date, name_map)
                row.setdefault("fetch_status", FETCH_SUCCESS if pd.notna(row.get("total_share")) else FETCH_DATA_UNKNOWN)
                rows.append(row)
                diagnostics.success_count += 1
            except Exception as exc:
                diagnostics.failed_count += 1
                diagnostics.messages.append(f"{code}: {exc}")
                rows.append({"code": code, "name": code, "name_source": "CODE_FALLBACK", "historical_st_status": "UNKNOWN", "st_status_source": "UNKNOWN", "share_pub_date": "", "share_stat_date": "", "total_share": float("nan"), "share_source": SHARE_UNKNOWN, "fetch_status": FETCH_REQUEST_FAILED_RETRYABLE})
            pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("code", keep="last")[cols].to_csv(path, index=False, encoding="utf-8-sig")
    finally:
        if own:
            provider.__exit__(None, None, None)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("code", keep="last")
    return combined[combined["code"].isin(wanted)][cols]
