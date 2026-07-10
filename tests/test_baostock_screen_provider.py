import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from baostock_screen_provider import (
    BaoStockDiagnostics,
    FETCH_DATA_UNKNOWN,
    FETCH_REQUEST_FAILED_RETRYABLE,
    FETCH_SUCCESS,
    BaoStockScreenProvider,
    load_or_fetch_enrichment,
    normalize_st_status,
    select_latest_published_total_share,
)


class FakeResultSet:
    error_code = "0"
    fields = ["code"]

    def __init__(self):
        self._used = False

    def next(self):
        if self._used:
            return False
        self._used = True
        return True

    def get_row_data(self):
        return ["sh.600000"]


class FakeBaoProvider:
    def __init__(self):
        self.calls = []
    def fetch_name_map(self, screen_date):
        return pd.DataFrame({"code": ["000001"], "name": ["平安银行"]})
    def fetch_one(self, code, screen_date, name_map=None):
        self.calls.append(code)
        return {"code": code, "name": name_map.get(code, code), "name_source": "BAOSTOCK_QUERY_ALL_STOCK" if code in name_map else "CODE_FALLBACK", "historical_st_status": "NON_ST", "st_status_source": "BAOSTOCK_ISST_SCREEN_DATE", "share_pub_date": "2022-10-29", "share_stat_date": "2022-09-30", "total_share": 100.0, "share_source": "BAOSTOCK_QUERY_PROFIT_DATA"}


class FlakyBaoProvider:
    def __init__(self):
        self.calls = []
        self.fail_once = {"000002"}
    def fetch_name_map(self, screen_date):
        return pd.DataFrame(columns=["code", "name"])
    def fetch_one(self, code, screen_date, name_map=None):
        self.calls.append(code)
        if code in self.fail_once:
            self.fail_once.remove(code)
            raise RuntimeError("temporary network")
        return {"code": code, "name": code, "name_source": "CODE_FALLBACK", "historical_st_status": "NON_ST", "st_status_source": "BAOSTOCK_ISST_SCREEN_DATE", "share_pub_date": "2022-10-29", "share_stat_date": "2022-09-30", "total_share": 100.0, "share_source": "BAOSTOCK_QUERY_PROFIT_DATA", "fetch_status": FETCH_SUCCESS}


class NameMapFailProvider(FakeBaoProvider):
    def fetch_name_map(self, screen_date):
        raise RuntimeError("name service down")


class UnknownBaoProvider(FakeBaoProvider):
    def fetch_one(self, code, screen_date, name_map=None):
        self.calls.append(code)
        return {"code": code, "name": code, "name_source": "CODE_FALLBACK", "historical_st_status": "UNKNOWN", "st_status_source": "UNKNOWN", "share_pub_date": "", "share_stat_date": "", "total_share": float("nan"), "share_source": "TOTAL_SHARE_UNKNOWN", "fetch_status": FETCH_DATA_UNKNOWN}


class BaoStockScreenProviderTests(unittest.TestCase):
    def test_call_sleeps_after_successful_baostock_request(self):
        provider = BaoStockScreenProvider(retry=2, request_interval=0.01)
        with patch("baostock_screen_provider.time.sleep") as sleep_mock:
            df = provider._call(lambda: FakeResultSet())
        self.assertEqual(df.loc[0, "code"], "sh.600000")
        sleep_mock.assert_called_once_with(0.01)

    def test_call_sleeps_for_failed_attempt_and_success_attempt(self):
        provider = BaoStockScreenProvider(retry=2, request_interval=0.01)
        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary")
            return FakeResultSet()

        with patch("baostock_screen_provider.time.sleep") as sleep_mock:
            provider._call(flaky)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_future_pubdate_not_used_and_latest_before_screen_selected(self):
        records = pd.DataFrame({
            "pubDate": ["2023-02-01", "2022-10-29", "2022-08-01"],
            "statDate": ["2022-12-31", "2022-09-30", "2022-06-30"],
            "totalShare": [999, 564369565, 100],
        })
        out = select_latest_published_total_share(records, "2023-01-03")
        self.assertEqual(out["total_share"], 564369565)
        self.assertEqual(out["share_pub_date"], "2022-10-29")

    def test_same_pubdate_prefers_latest_statdate(self):
        """
        Regression test based on historical BaoStock records for 001270.

        Multiple reporting periods can share the same pubDate.
        After enforcing pubDate <= screen_date, the newest statDate
        must be selected.
        """
        records = pd.DataFrame({
            "pubDate": [
                "2022-10-29",
                "2022-08-19",
                "2022-05-16",
                "2022-10-29",
            ],
            "statDate": [
                "2022-09-30",
                "2022-06-30",
                "2022-03-31",
                "2021-09-30",
            ],
            "totalShare": [
                111812946,
                111812946,
                83859446,
                83859446,
            ],
        })

        out = select_latest_published_total_share(
            records,
            "2023-01-03",
        )

        self.assertEqual(
            out["share_pub_date"],
            "2022-10-29",
        )
        self.assertEqual(
            out["share_stat_date"],
            "2022-09-30",
        )
        self.assertEqual(
            out["total_share"],
            111812946,
        )

    def test_total_share_unknown(self):
        out = select_latest_published_total_share(pd.DataFrame({"pubDate": ["2023-02-01"], "statDate": ["2022-12-31"], "totalShare": [100]}), "2023-01-03")
        self.assertEqual(out["share_source"], "TOTAL_SHARE_UNKNOWN")
        self.assertTrue(pd.isna(out["total_share"]))

    def test_fetch_one_stops_at_first_published_valid_quarter(self):
        provider = BaoStockScreenProvider()

        class FakeBS:
            def query_history_k_data_plus(self, *args, **kwargs):
                pass

            def query_profit_data(self, *args, **kwargs):
                pass

        provider.bs = FakeBS()

        profit_calls = []

        def fake_call(func, *args, **kwargs):
            if func.__name__ == "query_history_k_data_plus":
                return pd.DataFrame({
                    "date": ["2023-01-03"],
                    "code": ["sz.001270"],
                    "close": ["12.00"],
                    "isST": ["0"],
                })

            if func.__name__ == "query_profit_data":
                year = kwargs["year"]
                quarter = kwargs["quarter"]

                profit_calls.append((year, quarter))

                records = {
                    (2023, 1): {
                        "pubDate": "2023-04-27",
                        "statDate": "2023-03-31",
                        "totalShare": "111812946.00",
                    },
                    (2022, 4): {
                        "pubDate": "2023-03-29",
                        "statDate": "2022-12-31",
                        "totalShare": "111812946.00",
                    },
                    (2022, 3): {
                        "pubDate": "2022-10-29",
                        "statDate": "2022-09-30",
                        "totalShare": "111812946.00",
                    },
                }

                if (year, quarter) not in records:
                    self.fail(
                        f"early-stop failed: unexpected older quarter "
                        f"{year}Q{quarter} was requested"
                    )

                return pd.DataFrame([
                    records[(year, quarter)]
                ])

            raise AssertionError(
                f"unexpected BaoStock function: {func.__name__}"
            )

        provider._call = fake_call

        result = provider.fetch_one(
            code="001270",
            screen_date="20230103",
            name_map={},
        )

        self.assertEqual(
            profit_calls,
            [
                (2023, 1),
                (2022, 4),
                (2022, 3),
            ],
        )

        self.assertEqual(
            result["historical_st_status"],
            "NON_ST",
        )

        self.assertEqual(
            result["share_pub_date"],
            "2022-10-29",
        )

        self.assertEqual(
            result["share_stat_date"],
            "2022-09-30",
        )

        self.assertEqual(
            result["total_share"],
            111812946.0,
        )

        self.assertEqual(
            result["fetch_status"],
            FETCH_SUCCESS,
        )

    def test_st_mapping(self):
        self.assertEqual(normalize_st_status("1"), "ST")
        self.assertEqual(normalize_st_status("0"), "NON_ST")
        self.assertEqual(normalize_st_status(None), "UNKNOWN")

    def test_cache_resume_only_missing_codes_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            pd.DataFrame([{"code":"000001","name":"平安银行","name_source":"BAOSTOCK_QUERY_ALL_STOCK","historical_st_status":"NON_ST","st_status_source":"BAOSTOCK_ISST_SCREEN_DATE","share_pub_date":"2022-10-29","share_stat_date":"2022-09-30","total_share":100,"share_source":"BAOSTOCK_QUERY_PROFIT_DATA"}]).to_csv(cache / "20230103.csv", index=False)
            fake = FakeBaoProvider()
            diag = BaoStockDiagnostics()
            df = load_or_fetch_enrichment(["000001", "000002"], "20230103", cache, provider=fake, diagnostics=diag)
            self.assertEqual(fake.calls, ["000002"])
            self.assertEqual(diag.cache_hit_count, 1)
            self.assertEqual(len(df), 2)

    def test_request_failed_retryable_retried_but_success_cache_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            fake = FlakyBaoProvider()
            first = BaoStockDiagnostics()
            load_or_fetch_enrichment(["000001", "000002"], "20230103", cache, provider=fake, diagnostics=first)
            self.assertEqual(fake.calls, ["000001", "000002"])
            self.assertEqual(first.failed_count, 1)
            cached = pd.read_csv(cache / "20230103.csv", dtype={"code": str})
            self.assertEqual(cached.loc[cached["code"] == "000002", "fetch_status"].iloc[0], FETCH_REQUEST_FAILED_RETRYABLE)

            second = BaoStockDiagnostics()
            load_or_fetch_enrichment(["000001", "000002"], "20230103", cache, provider=fake, diagnostics=second)
            self.assertEqual(fake.calls, ["000001", "000002", "000002"])
            self.assertEqual(second.cache_hit_count, 1)

    def test_data_unknown_cache_is_completed_and_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            fake = UnknownBaoProvider()
            load_or_fetch_enrichment(["000003"], "20230103", cache, provider=fake, diagnostics=BaoStockDiagnostics())
            load_or_fetch_enrichment(["000003"], "20230103", cache, provider=fake, diagnostics=BaoStockDiagnostics())
            self.assertEqual(fake.calls, ["000003"])
            cached = pd.read_csv(cache / "20230103.csv", dtype={"code": str})
            self.assertEqual(cached.loc[0, "fetch_status"], FETCH_DATA_UNKNOWN)

    def test_name_map_failure_falls_back_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = NameMapFailProvider()
            diag = BaoStockDiagnostics()
            df = load_or_fetch_enrichment(["000001"], "20230103", Path(tmp), provider=fake, diagnostics=diag)
            self.assertEqual(fake.calls, ["000001"])
            self.assertEqual(df.loc[df["code"] == "000001", "name_source"].iloc[0], "CODE_FALLBACK")
            self.assertEqual(df.loc[df["code"] == "000001", "historical_st_status"].iloc[0], "NON_ST")
            self.assertIn("name_map", diag.messages[0])

if __name__ == "__main__":
    unittest.main()
