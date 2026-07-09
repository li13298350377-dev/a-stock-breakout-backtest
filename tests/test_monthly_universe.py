import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_snapshot_provider import CacheDiagnostics, load_cached_or_fetch_market_daily, validate_market_daily
from monthly_universe import add_history_metrics, build_base_universe, resolve_month_dates_from_calendar


class FakeProvider:
    name = "fake"
    def __init__(self, df): self.df = df
    def load_market_daily(self, trade_date): return self.df.copy()


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


if __name__ == "__main__":
    unittest.main()
