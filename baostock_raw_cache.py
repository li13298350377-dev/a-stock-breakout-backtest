
"""Persistent cross-month raw cache for BaoStock data."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = 1
QUERY_SUCCESS = "SUCCESS"


def normalize_code(code: str) -> str:
    return str(code).strip().split(".")[-1].zfill(6)


def quarter_key(year: int, quarter: int) -> str:
    year = int(year)
    quarter = int(quarter)

    if quarter not in {1, 2, 3, 4}:
        raise ValueError(
            f"invalid quarter: {quarter}"
        )

    return f"{year}Q{quarter}"


def empty_bundle(code: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "code": normalize_code(code),
        "st_years": {},
        "profit_quarters": {},
    }


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if pd.isna(value):
        return None

    return value


def dataframe_records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []

    records = []

    for row in df.to_dict(orient="records"):
        records.append({
            str(key): _json_scalar(value)
            for key, value in row.items()
        })

    return records


class BaoStockRawCache:
    """
    One persistent JSON bundle per stock.

    Bundle contains:
    - yearly ST rows;
    - raw query_profit_data results by quarter;
    - successful empty queries, so they are not repeated.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def path_for(self, code: str) -> Path:
        return (
            self.root
            / f"{normalize_code(code)}.json"
        )

    def load(self, code: str) -> dict:
        code = normalize_code(code)
        path = self.path_for(code)

        if not path.exists():
            return empty_bundle(code)

        try:
            bundle = json.loads(
                path.read_text(
                    encoding="utf-8",
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"raw cache corrupted: {path}"
            ) from exc

        if int(
            bundle.get(
                "schema_version",
                -1,
            )
        ) != SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported raw cache schema: {path}"
            )

        if normalize_code(
            bundle.get("code", "")
        ) != code:
            raise RuntimeError(
                f"raw cache code mismatch: {path}"
            )

        bundle.setdefault(
            "st_years",
            {},
        )

        bundle.setdefault(
            "profit_quarters",
            {},
        )

        return bundle

    def save(
        self,
        code: str,
        bundle: dict,
    ) -> Path:
        code = normalize_code(code)

        payload = dict(bundle)

        payload["schema_version"] = (
            SCHEMA_VERSION
        )

        payload["code"] = code

        payload.setdefault(
            "st_years",
            {},
        )

        payload.setdefault(
            "profit_quarters",
            {},
        )

        path = self.path_for(code)

        tmp_path = path.with_suffix(
            ".json.tmp"
        )

        tmp_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        os.replace(
            tmp_path,
            path,
        )

        return path


def has_st_year(
    bundle: dict,
    year: int,
) -> bool:
    entry = (
        bundle
        .get("st_years", {})
        .get(str(int(year)))
    )

    return (
        isinstance(entry, dict)
        and entry.get("status")
        == QUERY_SUCCESS
    )


def store_st_year(
    bundle: dict,
    year: int,
    df: pd.DataFrame,
) -> None:
    year_key = str(int(year))

    rows = dataframe_records(df)

    # Keep only deterministic date rows.
    if rows:
        normalized = pd.DataFrame(rows)

        if "date" in normalized.columns:
            normalized["date"] = (
                pd.to_datetime(
                    normalized["date"],
                    errors="coerce",
                )
                .dt.strftime("%Y-%m-%d")
            )

            normalized = (
                normalized
                .dropna(subset=["date"])
                .drop_duplicates(
                    "date",
                    keep="last",
                )
                .sort_values("date")
            )

            rows = dataframe_records(
                normalized
            )

    bundle.setdefault(
        "st_years",
        {},
    )[year_key] = {
        "status": QUERY_SUCCESS,
        "rows": rows,
    }


def lookup_st_raw(
    bundle: dict,
    screen_date: str | pd.Timestamp,
) -> str | None:
    dt = pd.to_datetime(screen_date)

    year_key = str(int(dt.year))

    entry = (
        bundle
        .get("st_years", {})
        .get(year_key)
    )

    if not isinstance(entry, dict):
        return None

    if entry.get("status") != QUERY_SUCCESS:
        return None

    wanted = dt.strftime("%Y-%m-%d")

    for row in entry.get("rows", []):
        row_date = pd.to_datetime(
            row.get("date"),
            errors="coerce",
        )

        if pd.isna(row_date):
            continue

        if (
            row_date.strftime("%Y-%m-%d")
            == wanted
        ):
            value = row.get("isST")

            if value is None:
                return None

            return str(value).strip()

    # Exact-date lookup only.
    # Never carry future or previous ST state.
    return None


def has_profit_quarter(
    bundle: dict,
    year: int,
    quarter: int,
) -> bool:
    key = quarter_key(
        year,
        quarter,
    )

    entry = (
        bundle
        .get("profit_quarters", {})
        .get(key)
    )

    return (
        isinstance(entry, dict)
        and entry.get("status")
        == QUERY_SUCCESS
    )


def store_profit_quarter(
    bundle: dict,
    year: int,
    quarter: int,
    df: pd.DataFrame,
) -> None:
    key = quarter_key(
        year,
        quarter,
    )

    bundle.setdefault(
        "profit_quarters",
        {},
    )[key] = {
        "status": QUERY_SUCCESS,
        "rows": dataframe_records(df),
    }


def load_profit_quarter(
    bundle: dict,
    year: int,
    quarter: int,
) -> pd.DataFrame:
    key = quarter_key(
        year,
        quarter,
    )

    entry = (
        bundle
        .get("profit_quarters", {})
        .get(key)
    )

    if not isinstance(entry, dict):
        return pd.DataFrame()

    if entry.get("status") != QUERY_SUCCESS:
        return pd.DataFrame()

    return pd.DataFrame(
        entry.get("rows", [])
    )
