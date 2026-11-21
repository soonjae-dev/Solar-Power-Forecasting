"""Synthetic only: snowgate temporal provenance, station separation, metric path."""
import unittest
import numpy as np
import pandas as pd
from src.experiments.snowgate_review import rolling_snow, issued, gate, choose, compare, SnowError


def fixture():
    ts = pd.date_range("2023-11-28 01:00", periods=100, freq="h", tz="Asia/Seoul")
    frames = []
    for st in ("a", "b"):
        for r in (10, 17):
            frames.append(pd.DataFrame(dict(wa_station=st, ts=ts, round=r, wa_snow=1. if st == "a" else 0.)))
    return pd.concat(frames, ignore_index=True)


class SnowTests(unittest.TestCase):
    def test_rolling_includes_current_and_separates_stations(self):
        out, audit = rolling_snow(fixture(), "wa")
        self.assertEqual(out.loc["a"].wa_snow_sum24.iloc[-1], 24)
        self.assertEqual(out.loc["a"].wa_snow_sum72.iloc[-1], 72)
        self.assertEqual(out.loc["b"].wa_snow_sum72.sum(), 0)
        self.assertEqual(audit["72"]["issue_violations"], 0)

    def test_future_changes_do_not_change_past(self):
        f = fixture()
        boundary = f.ts.min()+pd.Timedelta(hours=50)
        a, _ = rolling_snow(f, "wa")
        f.loc[f.ts > boundary, "wa_snow"] = 99
        b, _ = rolling_snow(f, "wa")
        mask = a.index.get_level_values("ts") <= boundary
        pd.testing.assert_frame_equal(a.loc[mask], b.loc[mask])

    def test_fallback_and_all_missing(self):
        f = fixture()
        ts = f.ts.min()
        f.loc[(f.ts == ts) & (f["round"] == 17), "wa_snow"] = np.nan
        a, audit = rolling_snow(f, "wa")
        self.assertEqual(audit["fallback10_rows"], 2)
        self.assertEqual(a.loc[("a", ts), "wa_snow_sum24"], 1)
        f.wa_snow = np.nan
        a, _ = rolling_snow(f, "wa")
        self.assertTrue(a.isna().all().all())

    def test_midnight_issue_is_previous_target_day(self):
        ts = pd.Series(pd.DatetimeIndex(["2023-12-02 00:00", "2023-12-02 01:00"], tz="Asia/Seoul"))
        result = issued(ts, [17, 10])
        self.assertEqual(result.iloc[0], pd.Timestamp("2023-11-30 17:00", tz="Asia/Seoul"))
        self.assertEqual(result.iloc[1], pd.Timestamp("2023-12-01 10:00", tz="Asia/Seoul"))

    def test_gate_is_forecast_only_and_off_is_exact(self):
        values = pd.DataFrame(dict(wa_snow_sum72=[0, 1, np.nan, np.nan], wb_snow_sum72=[0, 0, 2, np.nan]))
        on = gate(values)
        np.testing.assert_array_equal(on, [False, True, True, False])
        np.testing.assert_array_equal(choose(np.ones(4), np.full(4, 9.), on), [1, 9, 9, 1])

    def test_invalid_source_stops(self):
        f = fixture()
        with self.assertRaises(SnowError):
            rolling_snow(pd.concat([f, f.iloc[:1]]), "wa")
        f.loc[0, "wa_snow"] = -1
        with self.assertRaises(SnowError):
            rolling_snow(f, "wa")

    def test_metric_entrypoint(self):
        ts = pd.date_range("2023-12-01 01:00", periods=48, freq="h", tz="Asia/Seoul")
        index = pd.MultiIndex.from_product([ts, ["A", "B"]], names=["ts", "group"])
        actual = pd.Series(10., index=index)
        smp = pd.DataFrame(dict(w=np.ones(48)), index=ts)
        result = compare(actual, actual+2, actual+1, smp)
        self.assertEqual(result["change_pct"], -50.)
        self.assertEqual(result["change_ci95_pct"], [-50., -50.])


if __name__ == "__main__":
    unittest.main()
