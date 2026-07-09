import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baostock_screen_provider import (
    BaoStockDiagnostics,
    load_or_fetch_enrichment,
    normalize_st_status,
    select_latest_published_total_share,
)


class FakeBaoProvider:
    def __init__(self):
        self.calls = []
    def fetch_name_map(self, screen_date):
        return pd.DataFrame({"code": ["000001"], "name": ["平安银行"]})
    def fetch_one(self, code, screen_date, name_map=None):
        self.calls.append(code)
        return {"code": code, "name": name_map.get(code, code), "name_source": "BAOSTOCK_QUERY_ALL_STOCK" if code in name_map else "CODE_FALLBACK", "historical_st_status": "NON_ST", "st_status_source": "BAOSTOCK_ISST_SCREEN_DATE", "share_pub_date": "2022-10-29", "share_stat_date": "2022-09-30", "total_share": 100.0, "share_source": "BAOSTOCK_QUERY_PROFIT_DATA"}


class BaoStockScreenProviderTests(unittest.TestCase):
    def test_future_pubdate_not_used_and_latest_before_screen_selected(self):
        records = pd.DataFrame({
            "pubDate": ["2023-02-01", "2022-10-29", "2022-08-01"],
            "statDate": ["2022-12-31", "2022-09-30", "2022-06-30"],
            "totalShare": [999, 564369565, 100],
        })
        out = select_latest_published_total_share(records, "2023-01-03")
        self.assertEqual(out["total_share"], 564369565)
        self.assertEqual(out["share_pub_date"], "2022-10-29")

    def test_total_share_unknown(self):
        out = select_latest_published_total_share(pd.DataFrame({"pubDate": ["2023-02-01"], "statDate": ["2022-12-31"], "totalShare": [100]}), "2023-01-03")
        self.assertEqual(out["share_source"], "TOTAL_SHARE_UNKNOWN")
        self.assertTrue(pd.isna(out["total_share"]))

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

if __name__ == "__main__":
    unittest.main()
