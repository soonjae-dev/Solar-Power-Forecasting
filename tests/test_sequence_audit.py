"""우측닫힘 슬롯과 정답mask 분리. 합성자료만 사용."""
import unittest
import numpy as np
import pandas as pd
from src.experiments.sequence_audit import coverage, day_slots, pack_day


class SequenceTest(unittest.TestCase):
    def test_right_closed_midnight_is_same_day_last_slot(self):
        ts = pd.Series(pd.date_range("2024-01-01 01:00", periods=24, freq="h", tz="Asia/Seoul"))
        days, slots = day_slots(ts)
        self.assertEqual(days.nunique(), 1)
        np.testing.assert_array_equal(slots, np.arange(24))

    def test_zero_days_retained_partial_days_counted(self):
        ts = pd.date_range("2024-01-01 01:00", periods=25, freq="h", tz="Asia/Seoul")
        r = coverage(pd.DataFrame(dict(site_id="synthetic", ts=ts, energy=0.)))
        self.assertEqual(r["complete_zero_site_days"], 1)
        self.assertEqual(r["partial_site_days"], 1)
        self.assertEqual(r["site_days"], 2)

    def test_target_changes_and_masks_cannot_change_inputs(self):
        values = np.array([[1., np.nan], [2., 3.]])
        a = pack_day(values, [1, 4], [0., 9.], [True, False])
        b = pack_day(values, [1, 4], [99., -999.], [False, True])
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertTrue(a[3][1])
        self.assertEqual(a[2][1], 0.)
        self.assertFalse(a[3][4])
        self.assertTrue(np.isnan(a[2][4]))
        self.assertEqual(a[0][1, 1], 0.)
        self.assertEqual(a[1][1, 1], 0.)

    def test_cancelling_values_are_not_zero_day(self):
        ts = pd.date_range("2024-01-01 01:00", periods=24, freq="h", tz="Asia/Seoul")
        energy = np.zeros(24)
        energy[:2] = [-1, 1]
        r = coverage(pd.DataFrame(dict(site_id="synthetic", ts=ts, energy=energy)))
        self.assertEqual(r["complete_zero_site_days"], 0)

    def test_duplicate_slots_rejected_and_inputs_copied(self):
        with self.assertRaises(ValueError):
            pack_day(np.ones((2, 3)), [1, 1], [0, 0], [True, True])
        x = np.ones((1, 3))
        a = pack_day(x, [1], [0], [True])
        x[:] = 9
        self.assertEqual(a[0][1, 0], 1.)


if __name__ == "__main__":
    unittest.main()
