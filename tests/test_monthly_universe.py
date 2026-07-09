import io
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
    validate_market_daily,
)
from monthly_universe import (
    add_history_metrics,
    build_base_universe,
    build_prefilter_candidates,
    build_enriched_snapshot,
    screen_daily_rows,
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


class FullFlowProvider:
    name = "full_flow_fake"

    def __init__(self):
        self.daily_calls = []
        self.daily_basic_calls = 0
        self.stk_premarket_calls = 0

    def load_market_daily(self, trade_date):
        self.daily_calls.append(trade_date)
        rows = [
            {"trade_date": trade_date, "code": "000001", "close": 10, "pct_chg": 0, "amount": 60_000_000},
            {"trade_date": trade_date, "code": "000002", "close": 50, "pct_chg": 0, "amount": 60_000_000},
        ]
        for i in range(1000):
            rows.append({"trade_date": trade_date, "code": f"300{i:03d}", "close": 10, "pct_chg": 0, "amount": 60_000_000})
        return pd.DataFrame(rows)

    def daily_basic(self, *args, **kwargs):
        self.daily_basic_calls += 1
        raise AssertionError("daily_basic must not be called")

    def stk_premarket(self, *args, **kwargs):
        self.stk_premarket_calls += 1
        raise AssertionError("stk_premarket must not be called")


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

    def test_full_flow_does_not_call_daily_basic_or_stk_premarket(self):
        pro = FakePro()
        provider = make_provider(pro)
        provider.fetch_market_daily("20230103")
        self.assertEqual(pro.calls["daily_basic"], 0)
        self.assertFalse(hasattr(pro, "stk_premarket"))

    def test_name_missing_is_unknown_not_non_st(self):
        pro = FakePro(names=None)
        provider = make_provider(pro)
        df = provider.fetch_market_snapshot("20230103")
        self.assertEqual(set(df["historical_st_status"]), {"UNKNOWN"})
        self.assertEqual(classify_historical_st_status(None), "UNKNOWN")

    def test_market_cap_close_times_total_share_units(self):
        pre = pd.DataFrame({"trade_date": ["20230103"], "code": ["600237"], "close": [6.74], "pct_chg": [0], "amount": [60_000_000], "listing_days": [121], "avg_amount_5": [1], "avg_amount_20": [60_000_000], "avg_amount_60": [1], "heat_ratio": [1], "active_days_20": [1], "abnormal_attention_count_20": [0]})
        enrich = pd.DataFrame({"code": ["600237"], "name": ["铜峰电子"], "name_source": ["BAOSTOCK_QUERY_ALL_STOCK"], "historical_st_status": ["NON_ST"], "st_status_source": ["BAOSTOCK_ISST_SCREEN_DATE"], "share_pub_date": ["2022-10-29"], "share_stat_date": ["2022-09-30"], "total_share": [564369565], "share_source": ["BAOSTOCK_QUERY_PROFIT_DATA"]})
        snap = build_enriched_snapshot(pre, enrich)
        self.assertEqual(snap.loc[0, "historical_market_cap"], 6.74 * 564369565)

    def test_unknown_share_excludes_with_reason(self):
        snapshot = pd.DataFrame({"trade_date": ["20230103"], "code": ["000001"], "name": ["平安银行"], "close": [10], "total_share": [float("nan")], "historical_market_cap": [float("nan")], "listing_days": [121], "avg_amount_20": [60_000_000], "historical_st_status": ["NON_ST"]})
        base = build_base_universe(snapshot, pd.DataFrame(), pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"))
        self.assertFalse(bool(base.iloc[0]["passed"]))
        self.assertIn("历史总股本未知", base.iloc[0]["exclude_reason"])

    def test_prefilter_before_baostock_only_candidates(self):
        dates = pd.bdate_range("2022-07-01", periods=121)
        rows = []
        for d in dates:
            rows += [
                {"trade_date": d.strftime("%Y%m%d"), "code": "000001", "close": 10, "pct_chg": 0, "amount": 60_000_000},
                {"trade_date": d.strftime("%Y%m%d"), "code": "300001", "close": 10, "pct_chg": 0, "amount": 60_000_000},
                {"trade_date": d.strftime("%Y%m%d"), "code": "000002", "close": 50, "pct_chg": 0, "amount": 60_000_000},
            ]
        hist = pd.DataFrame(rows)
        metrics = add_history_metrics(hist, dates[-1])
        pre = build_prefilter_candidates(screen_daily_rows(hist, dates[-1]), metrics)
        self.assertEqual(pre["code"].tolist(), ["000001"])

    def test_run_full_does_not_call_paid_tushare_and_enriches_only_prefilter_codes(self):
        cal = pd.DatetimeIndex(pd.bdate_range("2022-06-01", "2023-01-05"))
        provider = FullFlowProvider()
        captured = {}

        def fake_enrichment(codes, screen_date, cache_dir, provider=None, diagnostics=None):
            captured["codes"] = list(codes)
            diagnostics.messages.append("000009: simulated baostock detail")
            return pd.DataFrame({
                "code": codes,
                "name": codes,
                "name_source": ["CODE_FALLBACK"] * len(codes),
                "historical_st_status": ["NON_ST"] * len(codes),
                "st_status_source": ["BAOSTOCK_ISST_SCREEN_DATE"] * len(codes),
                "share_pub_date": ["2022-10-29"] * len(codes),
                "share_stat_date": ["2022-09-30"] * len(codes),
                "total_share": [300_000_000] * len(codes),
                "share_source": ["BAOSTOCK_QUERY_PROFIT_DATA"] * len(codes),
                "fetch_status": ["SUCCESS"] * len(codes),
            })

        with tempfile.TemporaryDirectory() as tmp, \
                patch("monthly_universe.get_trade_dates", return_value=cal), \
                patch("monthly_universe.RESULT_DIR", Path(tmp) / "results"), \
                patch("monthly_universe.MARKET_DAILY_CACHE_DIR", Path(tmp) / "daily"), \
                patch("monthly_universe.BAOSTOCK_ENRICHMENT_CACHE_DIR", Path(tmp) / "baostock"), \
                patch("monthly_universe.load_or_fetch_enrichment", side_effect=fake_enrichment):
            run_full(provider)
            diagnostics = pd.read_csv(Path(tmp) / "results" / "data_diagnostics.csv")

        self.assertEqual(provider.daily_basic_calls, 0)
        self.assertEqual(provider.stk_premarket_calls, 0)
        self.assertEqual(captured["codes"], ["000001"])
        self.assertIn("simulated baostock detail", diagnostics.loc[0, "messages"])

    def test_probe_requests_exact_daily_dates_and_fixed_baostock_codes(self):
        cal = pd.DatetimeIndex(pd.bdate_range("2022-06-01", "2023-01-05"))
        provider = FakeProvider(daily_df=pd.DataFrame({
            "trade_date": ["20221229"],
            "code": ["000001"],
            "close": [10],
            "pct_chg": [0],
            "amount": [1],
        }))
        daily_dates = []
        baostock_codes = []

        def fake_daily(provider_arg, trade_date, cache_dir, diagnostics, **kwargs):
            daily_dates.append(trade_date)
            return pd.DataFrame({"trade_date": [trade_date], "code": ["000001"], "close": [10], "pct_chg": [0], "amount": [1]})

        def fake_enrichment(codes, screen_date, cache_dir, provider=None, diagnostics=None):
            baostock_codes.extend(codes)
            return pd.DataFrame({"code": codes, "total_share": [1] * len(codes), "share_pub_date": ["2022-10-29"] * len(codes), "share_stat_date": ["2022-09-30"] * len(codes), "historical_st_status": ["NON_ST"] * len(codes)})

        with tempfile.TemporaryDirectory() as tmp, \
                patch("monthly_universe.get_trade_dates", return_value=cal), \
                patch("monthly_universe.RESULT_DIR", Path(tmp) / "results"), \
                patch("monthly_universe.MARKET_DAILY_CACHE_DIR", Path(tmp) / "daily"), \
                patch("monthly_universe.BAOSTOCK_ENRICHMENT_CACHE_DIR", Path(tmp) / "baostock"), \
                patch("monthly_universe.required_history_dates", return_value=["20221229", "20221230", "20230102"]), \
                patch("monthly_universe.load_cached_or_fetch_market_daily", side_effect=fake_daily), \
                patch("monthly_universe.load_or_fetch_enrichment", side_effect=fake_enrichment):
            print_probe(provider)

        self.assertEqual(daily_dates, ["20221229", "20221230"])
        self.assertEqual(baostock_codes, ["600237", "002559", "002962", "000751", "600520"])

    def test_probe_prints_name_map_stats_and_fallback_message(self):
        cal = pd.DatetimeIndex(pd.bdate_range("2022-06-01", "2023-01-05"))
        provider = FakeProvider(daily_df=pd.DataFrame({
            "trade_date": ["20221229"],
            "code": ["000001"],
            "close": [10],
            "pct_chg": [0],
            "amount": [1],
        }))

        def fake_daily(provider_arg, trade_date, cache_dir, diagnostics, **kwargs):
            return pd.DataFrame({"trade_date": [trade_date], "code": ["000001"], "close": [10], "pct_chg": [0], "amount": [1]})

        def fake_enrichment(codes, screen_date, cache_dir, provider=None, diagnostics=None):
            diagnostics.name_map_rows = 3
            diagnostics.name_map_unique_codes = 2
            diagnostics.name_map_non_empty_names = 1
            return pd.DataFrame({"code": codes, "total_share": [1] * len(codes), "share_pub_date": ["2022-10-29"] * len(codes), "share_stat_date": ["2022-09-30"] * len(codes), "historical_st_status": ["NON_ST"] * len(codes)})

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("monthly_universe.get_trade_dates", return_value=cal), \
                patch("monthly_universe.RESULT_DIR", Path(tmp) / "results"), \
                patch("monthly_universe.MARKET_DAILY_CACHE_DIR", Path(tmp) / "daily"), \
                patch("monthly_universe.BAOSTOCK_ENRICHMENT_CACHE_DIR", Path(tmp) / "baostock"), \
                patch("monthly_universe.required_history_dates", return_value=["20221229", "20221230", "20230102"]), \
                patch("monthly_universe.load_cached_or_fetch_market_daily", side_effect=fake_daily), \
                patch("monthly_universe.load_or_fetch_enrichment", side_effect=fake_enrichment), \
                patch("sys.stdout", out):
            print_probe(provider)

        text = out.getvalue()
        self.assertIn("BaoStock名称映射行数: 3", text)
        self.assertIn("BaoStock名称映射code去重数: 2", text)
        self.assertIn("BaoStock名称非空数量: 1", text)

    def test_probe_prints_name_map_fallback_when_mapping_fails(self):
        cal = pd.DatetimeIndex(pd.bdate_range("2022-06-01", "2023-01-05"))
        provider = FakeProvider(daily_df=pd.DataFrame({"trade_date": ["20221229"], "code": ["000001"], "close": [10], "pct_chg": [0], "amount": [1]}))

        def fake_daily(provider_arg, trade_date, cache_dir, diagnostics, **kwargs):
            return pd.DataFrame({"trade_date": [trade_date], "code": ["000001"], "close": [10], "pct_chg": [0], "amount": [1]})

        def fake_enrichment(codes, screen_date, cache_dir, provider=None, diagnostics=None):
            diagnostics.name_map_failed = True
            return pd.DataFrame({"code": codes, "total_share": [1] * len(codes), "share_pub_date": ["2022-10-29"] * len(codes), "share_stat_date": ["2022-09-30"] * len(codes), "historical_st_status": ["NON_ST"] * len(codes)})

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("monthly_universe.get_trade_dates", return_value=cal), \
                patch("monthly_universe.RESULT_DIR", Path(tmp) / "results"), \
                patch("monthly_universe.MARKET_DAILY_CACHE_DIR", Path(tmp) / "daily"), \
                patch("monthly_universe.BAOSTOCK_ENRICHMENT_CACHE_DIR", Path(tmp) / "baostock"), \
                patch("monthly_universe.required_history_dates", return_value=["20221229", "20221230", "20230102"]), \
                patch("monthly_universe.load_cached_or_fetch_market_daily", side_effect=fake_daily), \
                patch("monthly_universe.load_or_fetch_enrichment", side_effect=fake_enrichment), \
                patch("sys.stdout", out):
            print_probe(provider)

        self.assertIn("BaoStock名称映射失败，使用 CODE_FALLBACK", out.getvalue())

    def test_download_failure_saves_diagnostics(self):
        cal = pd.DatetimeIndex(pd.bdate_range("2022-06-01", periods=180))
        with tempfile.TemporaryDirectory() as tmp, \
                patch("monthly_universe.get_trade_dates", return_value=cal), \
                patch("monthly_universe.RESULT_DIR", Path(tmp) / "results"), \
                patch("monthly_universe.MARKET_DAILY_CACHE_DIR", Path(tmp) / "daily"), \
                patch("monthly_universe.BAOSTOCK_ENRICHMENT_CACHE_DIR", Path(tmp) / "baostock"):
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
                patch("monthly_universe.BAOSTOCK_ENRICHMENT_CACHE_DIR", Path(tmp) / "baostock"):
            with self.assertRaises(RuntimeError):
                print_probe(FakeProvider(fail=True))
            diagnostics = pd.read_csv(Path(tmp) / "results" / "data_diagnostics.csv")
            self.assertEqual(diagnostics.loc[0, "mode"], "probe")
            self.assertIn("failed_trade_dates", diagnostics.columns)
            self.assertIn("messages", diagnostics.columns)
            self.assertIn("download failed", diagnostics.loc[0, "messages"])


if __name__ == "__main__":
    unittest.main()
