import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from market_snapshot_provider import (
    CacheDiagnostics,
    TushareBatchProvider,
    classify_historical_st_status,
    load_cached_or_fetch_market_daily,
    load_cached_or_fetch_market_snapshot,
    validate_market_daily,
)
from monthly_universe import (
    add_history_metrics,
    build_base_universe,
    print_probe,
    resolve_month_dates_from_calendar,
    run_full,
)


class FakeProvider:
    name = "fake"

    def __init__(self, daily_df=None, snapshot_df=None, fail=False):
        self.daily_df = daily_df
        self.snapshot_df = snapshot_df
        self.fail = fail

    def load_market_daily(self, trade_date):
        if self.fail:
            raise RuntimeError("download failed")
        return self.daily_df.copy()

    def load_market_snapshot(self, trade_date):
        if self.fail:
            raise RuntimeError("snapshot failed")
        return self.snapshot_df.copy()


class FakePro:
    def __init__(self, names=None):
        self.calls = {"daily": 0, "daily_basic": 0, "namechange": 0}
        self.names = names

    def daily(self, trade_date):
        self.calls["daily"] += 1
        return pd.DataFrame({
            "ts_code": ["000001.SZ", "600000.SH"],
            "close": [10.0, 11.0],
            "pct_chg": [1.0, -1.0],
            "amount": [100.0, 200.0],
        })

    def daily_basic(self, trade_date, fields):
        self.calls["daily_basic"] += 1
        return pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH"], "total_mv": [300000.0, 400000.0]})

    def namechange(self, start_date, end_date, fields):
        self.calls["namechange"] += 1
        if self.names is None:
            return pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date", "change_reason"])
        return self.names.copy()


def make_provider(pro):
    provider = TushareBatchProvider.__new__(TushareBatchProvider)
    provider.pro = pro
    provider._name_cache = {}
    return provider


class MonthlyUniverseTests(unittest.TestCase):
    def test_month_dates(self):
        cal = pd.DatetimeIndex(pd.to_datetime(["2022-12-30", "2023-01-03", "2023-01-04"]))
        screen, effective = resolve_month_dates_from_calendar(cal, "2023-01")
        self.assertEqual(screen.strftime("%Y%m%d"), "20230103")
        self.assertEqual(effective.strftime("%Y%m%d"), "20230104")

    def test_popularity_windows_exclude_current_day_baselines(self):
        dates = pd.bdate_range("2022-07-01", periods=121)
        amounts = [100.0] * 100 + [100.0] * 20 + [1000.0]
        pct = [0.0] * 120 + [5.0]
        df = pd.DataFrame({"trade_date": [d.strftime("%Y%m%d") for d in dates], "code": "000001", "close": 10, "pct_chg": pct, "amount": amounts, "total_market_cap": 3e9})
        metrics = add_history_metrics(df, dates[-1])
        row = metrics.iloc[0]
        self.assertEqual(row["abnormal_attention_count_20"], 1)
        self.assertEqual(row["active_days_20"], 1)

    def test_cache_date_validation(self):
        bad = pd.DataFrame({"trade_date": ["20230104"], "code": ["000001"], "close": [1], "pct_chg": [0], "amount": [1]})
        with self.assertRaises(ValueError):
            validate_market_daily(bad, "20230103", min_unique_codes=1)

    def test_cache_loader_rejects_wrong_date(self):
        good = pd.DataFrame({"trade_date": ["20230104"], "code": ["000001"], "close": [1], "pct_chg": [0], "amount": [1]})
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            good.to_csv(d / "20230103.csv", index=False)
            with self.assertRaises(ValueError):
                load_cached_or_fetch_market_daily(FakeProvider(good), "20230103", d, CacheDiagnostics("fake"), min_unique_codes=1)

    def test_base_uses_screen_date_snapshot_not_effective_date(self):
        dates = pd.bdate_range("2022-07-01", periods=121)
        rows = []
        for d in dates:
            rows.append({"trade_date": d.strftime("%Y%m%d"), "code": "000001", "close": 10, "pct_chg": 0, "amount": 60_000_000, "total_market_cap": 3_000_000_000})
        history = pd.DataFrame(rows)
        metrics = add_history_metrics(history, dates[-1])
        snapshot = history[history.trade_date == dates[-1].strftime("%Y%m%d")].copy()
        snapshot["name"] = "平安银行"
        snapshot["close"] = 10
        base = build_base_universe(snapshot, metrics, dates[-1], dates[-1] + pd.offsets.BDay(1))
        self.assertTrue(bool(base.iloc[0]["passed"]))
        self.assertEqual(base.iloc[0]["close"], 10)

    def test_daily_does_not_call_daily_basic_or_namechange(self):
        pro = FakePro()
        provider = make_provider(pro)
        df = provider.fetch_market_daily("20230103")
        self.assertEqual(pro.calls["daily"], 1)
        self.assertEqual(pro.calls["daily_basic"], 0)
        self.assertEqual(pro.calls["namechange"], 0)
        self.assertEqual(set(["trade_date", "code", "close", "pct_chg", "amount"]).issubset(df.columns), True)

    def test_snapshot_calls_daily_basic_and_namechange_once(self):
        names = pd.DataFrame({
            "ts_code": ["000001.SZ", "600000.SH"],
            "name": ["平安银行", "*ST浦发"],
            "start_date": ["20200101", "20200101"],
            "end_date": [None, None],
            "change_reason": ["", ""],
        })
        pro = FakePro(names)
        provider = make_provider(pro)
        df = provider.fetch_market_snapshot("20230103")
        self.assertEqual(pro.calls["daily_basic"], 1)
        self.assertEqual(pro.calls["namechange"], 1)
        self.assertIn("total_market_cap", df.columns)
        self.assertEqual(df.loc[df["code"] == "600000", "historical_st_status"].iloc[0], "ST")

    def test_name_missing_is_unknown_not_non_st(self):
        pro = FakePro(names=None)
        provider = make_provider(pro)
        df = provider.fetch_market_snapshot("20230103")
        self.assertEqual(set(df["historical_st_status"]), {"UNKNOWN"})
        self.assertEqual(classify_historical_st_status(None), "UNKNOWN")

    def test_snapshot_missing_total_market_cap_fails_validation(self):
        bad = pd.DataFrame({"trade_date": ["20230103"], "code": ["000001"], "name": ["平安银行"], "close": [1], "pct_chg": [0], "amount": [1]})
        with self.assertRaises(ValueError):
            validate_market_daily(bad, "20230103", require_snapshot=True, min_unique_codes=1)

    def test_daily_and_snapshot_cache_do_not_cross_read(self):
        daily = pd.DataFrame({"trade_date": ["20230103"], "code": ["000001"], "close": [1], "pct_chg": [0], "amount": [1]})
        snapshot = pd.DataFrame({"trade_date": ["20230103"], "code": ["000001"], "name": ["平安银行"], "close": [1], "pct_chg": [0], "amount": [1], "total_market_cap": [3e9]})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            load_cached_or_fetch_market_daily(FakeProvider(daily_df=daily), "20230103", root / "daily", CacheDiagnostics("fake"), min_unique_codes=1)
            snap = load_cached_or_fetch_market_snapshot(FakeProvider(snapshot_df=snapshot), "20230103", root / "snapshot", CacheDiagnostics("fake"), min_unique_codes=1)
            self.assertIn("total_market_cap", snap.columns)
            self.assertFalse((root / "daily" / "20230103.csv").read_text().find("total_market_cap") >= 0)

    def test_download_failure_saves_diagnostics(self):
        cal = pd.DatetimeIndex(pd.bdate_range("2022-06-01", periods=180))
        with tempfile.TemporaryDirectory() as tmp, \
                patch("monthly_universe.get_trade_dates", return_value=cal), \
                patch("monthly_universe.RESULT_DIR", Path(tmp) / "results"), \
                patch("monthly_universe.MARKET_DAILY_CACHE_DIR", Path(tmp) / "daily"), \
                patch("monthly_universe.MARKET_SNAPSHOT_CACHE_DIR", Path(tmp) / "snapshot"):
            with self.assertRaises(RuntimeError):
                run_full(FakeProvider(fail=True))
            diagnostics = pd.read_csv(Path(tmp) / "results" / "data_diagnostics.csv")
            self.assertIn("failed_trade_dates", diagnostics.columns)
            self.assertIn("messages", diagnostics.columns)
            self.assertIn("download failed", diagnostics.loc[0, "messages"])

    def test_probe_failure_saves_diagnostics(self):
        cal = pd.DatetimeIndex(pd.bdate_range("2022-06-01", periods=180))
        with tempfile.TemporaryDirectory() as tmp, \
                patch("monthly_universe.get_trade_dates", return_value=cal), \
                patch("monthly_universe.RESULT_DIR", Path(tmp) / "results"), \
                patch("monthly_universe.MARKET_DAILY_CACHE_DIR", Path(tmp) / "daily"), \
                patch("monthly_universe.MARKET_SNAPSHOT_CACHE_DIR", Path(tmp) / "snapshot"):
            with self.assertRaises(RuntimeError):
                print_probe(FakeProvider(fail=True))
            diagnostics = pd.read_csv(Path(tmp) / "results" / "data_diagnostics.csv")
            self.assertEqual(diagnostics.loc[0, "mode"], "probe")
            self.assertIn("failed_trade_dates", diagnostics.columns)
            self.assertIn("messages", diagnostics.columns)
            self.assertIn("download failed", diagnostics.loc[0, "messages"])


if __name__ == "__main__":
    unittest.main()
