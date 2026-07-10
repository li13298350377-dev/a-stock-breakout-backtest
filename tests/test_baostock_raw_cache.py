
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baostock_raw_cache import (
    BaoStockRawCache,
    empty_bundle,
    has_profit_quarter,
    has_st_year,
    load_profit_quarter,
    lookup_st_raw,
    normalize_code,
    quarter_key,
    store_profit_quarter,
    store_st_year,
)


class BaoStockRawCacheTests(
    unittest.TestCase
):

    def test_code_and_quarter_normalization(self):
        self.assertEqual(
            normalize_code("sh.600238"),
            "600238",
        )

        self.assertEqual(
            normalize_code("1270"),
            "001270",
        )

        self.assertEqual(
            quarter_key(2023, 3),
            "2023Q3",
        )

        with self.assertRaises(ValueError):
            quarter_key(2023, 5)


    def test_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:

            cache = BaoStockRawCache(
                Path(tmp)
            )

            bundle = empty_bundle(
                "600238"
            )

            store_st_year(
                bundle,
                2023,
                pd.DataFrame({
                    "date": [
                        "2023-01-03",
                    ],
                    "isST": [
                        "0",
                    ],
                }),
            )

            cache.save(
                "600238",
                bundle,
            )

            loaded = cache.load(
                "600238"
            )

            self.assertTrue(
                has_st_year(
                    loaded,
                    2023,
                )
            )

            self.assertEqual(
                lookup_st_raw(
                    loaded,
                    "20230103",
                ),
                "0",
            )


    def test_st_lookup_is_exact_date_only(self):
        bundle = empty_bundle(
            "600238"
        )

        store_st_year(
            bundle,
            2023,
            pd.DataFrame({
                "date": [
                    "2023-01-03",
                    "2023-01-05",
                ],
                "isST": [
                    "0",
                    "1",
                ],
            }),
        )

        self.assertEqual(
            lookup_st_raw(
                bundle,
                "2023-01-03",
            ),
            "0",
        )

        self.assertIsNone(
            lookup_st_raw(
                bundle,
                "2023-01-04",
            )
        )


    def test_empty_profit_query_is_cached(self):
        bundle = empty_bundle(
            "001270"
        )

        store_profit_quarter(
            bundle,
            2023,
            1,
            pd.DataFrame(),
        )

        self.assertTrue(
            has_profit_quarter(
                bundle,
                2023,
                1,
            )
        )

        self.assertTrue(
            load_profit_quarter(
                bundle,
                2023,
                1,
            ).empty
        )


    def test_profit_rows_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:

            cache = BaoStockRawCache(
                Path(tmp)
            )

            bundle = empty_bundle(
                "001270"
            )

            df = pd.DataFrame({
                "pubDate": [
                    "2022-10-29",
                ],
                "statDate": [
                    "2022-09-30",
                ],
                "totalShare": [
                    "111812946",
                ],
            })

            store_profit_quarter(
                bundle,
                2022,
                3,
                df,
            )

            cache.save(
                "001270",
                bundle,
            )

            loaded = cache.load(
                "001270"
            )

            restored = (
                load_profit_quarter(
                    loaded,
                    2022,
                    3,
                )
            )

            self.assertEqual(
                restored.loc[
                    0,
                    "pubDate",
                ],
                "2022-10-29",
            )

            self.assertEqual(
                restored.loc[
                    0,
                    "totalShare",
                ],
                "111812946",
            )


    def test_corrupt_json_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:

            cache = BaoStockRawCache(
                Path(tmp)
            )

            path = cache.path_for(
                "600238"
            )

            path.write_text(
                "{not valid json",
                encoding="utf-8",
            )

            with self.assertRaises(
                RuntimeError
            ):
                cache.load(
                    "600238"
                )


if __name__ == "__main__":
    unittest.main()
