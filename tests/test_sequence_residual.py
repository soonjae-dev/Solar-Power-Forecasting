"""실제 자료를 읽지 않는 입력 독립성/출력 경계/시간 마감 검사."""
import unittest
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from src.experiments.sequence_residual import Block, corrected, fit_normalizer, make_model, tensor_batch, stable_average
from src.solar import solar


def synthetic():
    ts = pd.date_range("2023-06-30 01:00", periods=24, freq="h", tz="Asia/Seoul")
    sites = pd.DataFrame(dict(site_id=["synthetic"], lat=[36.], lon=[127.], capacity_kw=[100.])).set_index("site_id")
    elev, _, cosz = solar(ts, np.full(24, 36.), np.full(24, 127.))
    weather = pd.DataFrame(dict(site_id="synthetic", ts=ts[elev > 0], cosz=cosz[elev > 0], cap=100.))
    for i in range(81):
        weather[f"x{i}"] = 1.
    cols = [c for c in weather if c not in ("site_id", "ts")]
    gfs = pd.DataFrame(np.ones((24, 60)), index=ts, columns=[f"g{i}" for i in range(60)])
    p = pd.DataFrame(dict(site_id="synthetic", ts=ts, energy=0., forecast_kwh=0.))
    return p, weather, cols, gfs, sites


class SequenceResidualTest(unittest.TestCase):
    def test_neighbor_average_matches_scalar_definition(self):
        values = np.arange(20, dtype="float32").reshape(5, 4)
        weights = np.array([[.25, .25, .25, .25], [0., .5, .5, 0.]])
        expected = np.array([[sum(float(v)*float(w) for v, w in zip(row, weight))
                              for weight in weights] for row in values])
        np.testing.assert_allclose(stable_average(values, weights), expected, rtol=0, atol=1e-12)

    def test_targets_and_observation_mask_do_not_change_input(self):
        p, w, cols, g, s = synthetic()
        a = Block(p, w, cols, g, s)
        p = p.drop(index=[10]).copy()  # 주간 관측 하나 삭제해도 입력은 동일
        p["energy"] = 999.
        p["forecast_kwh"] = 888.
        b = Block(p, w, cols, g, s)
        norm = fit_normalizer([a])
        np.testing.assert_array_equal(a.inputs([0], norm), b.inputs([0], norm))
        self.assertEqual(b.observed.sum(), 23)
        self.assertTrue(np.isfinite(b.inputs([0], norm)).all())

    def test_missing_daylight_input_is_rejected(self):
        p, w, cols, g, s = synthetic()
        with self.assertRaises(ValueError):
            Block(p, w.iloc[1:], cols, g, s)

    def test_cutoff_excludes_late_labels_but_preserves_input(self):
        p, w, cols, g, s = synthetic()
        b = Block(p, w, cols, g, s)
        norm = fit_normalizer([b])
        cutoff = pd.Timestamp("2023-06-30 17:00", tz="Asia/Seoul")
        a = tensor_batch(b, np.array([0]), norm, "cpu")
        c = tensor_batch(b, np.array([0]), norm, "cpu", cutoff)
        self.assertEqual(c[4].sum().item(), 17)
        torch.testing.assert_close(a[0], c[0])

    def test_zero_initialization_and_bound_and_night(self):
        p, w, cols, g, s = synthetic()
        p["forecast_kwh"] = np.where(p.ts.dt.hour.between(6, 19), 10., 0.)
        b = Block(p, w, cols, g, s)
        batch = tensor_batch(b, np.array([0]), fit_normalizer([b]), "cpu")
        model = make_model()
        torch.testing.assert_close(corrected(model, batch), batch[1])
        with torch.no_grad():
            model[-2].bias.fill_(100)
        pred = corrected(model, batch)
        self.assertTrue(torch.all((pred-batch[1]).abs() <= .200001*batch[2]))
        night = batch[2] == 0
        torch.testing.assert_close(pred[night], batch[1][night])


if __name__ == "__main__":
    unittest.main()
