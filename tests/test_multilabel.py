import numpy as np
import pytest
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R

from meshmetrics import compute_metrics, compute_metrics_multilabel, np2sitk

TAUS = (1.73,)  # avoids distances lying exactly on tau (box-shaped labels)


def label_maps(ndim):
    """Label maps with shared labels 1-3, label 4 only in ref and label 5 only in pred."""
    shape = (40, 36, 30)[:ndim]
    ref = np.zeros(shape, np.uint8)
    pred = np.zeros(shape, np.uint8)
    boxes = {
        1: ((2, 2, 2), (14, 12, 10)),
        2: ((18, 4, 6), (30, 16, 20)),
        3: ((5, 20, 12), (16, 33, 26)),
    }
    for label, (lo, hi) in boxes.items():
        lo, hi = lo[:ndim], hi[:ndim]
        ref[tuple(slice(a, b) for a, b in zip(lo, hi))] = label
        shift = (1, -1, 2)[:ndim]
        pred[tuple(slice(a + s, b + s) for a, b, s in zip(lo, hi, shift))] = label
    ref[tuple(slice(a, b) for a, b in [(32, 38), (2, 8), (24, 29)][:ndim])] = 4
    pred[tuple(slice(a, b) for a, b in [(33, 38), (20, 26), (2, 8)][:ndim])] = 5
    return ref, pred


def to_image(arr, ndim):
    img = np2sitk(arr)
    img.SetSpacing([0.8, 1.1, 1.3][:ndim])
    img.SetOrigin([5.0, -3.0, 2.0][:ndim])
    direction = R.random(random_state=4).as_matrix() if ndim == 3 else np.array([[0.6, -0.8], [0.8, 0.6]])
    img.SetDirection(direction.ravel().tolist())
    return img


def assert_same(res, expected):
    assert set(res) == set(expected)
    for k, v in expected.items():
        if isinstance(v, bool):
            assert res[k] == v, k
        else:
            assert res[k] == pytest.approx(v, rel=1e-6, abs=1e-6), k


@pytest.mark.parametrize("ndim", [2, 3])
def test_matches_per_label_loop(ndim):
    ref, pred = label_maps(ndim)
    ref_img, pred_img = to_image(ref, ndim), to_image(pred, ndim)
    results = compute_metrics_multilabel(ref_img, pred_img, taus=TAUS, verbose=False)
    assert list(results) == [1, 2, 3, 4, 5]
    for label, res in results.items():
        expected = compute_metrics(
            to_image((ref == label).astype(np.uint8), ndim),
            to_image((pred == label).astype(np.uint8), ndim),
            taus=TAUS,
            verbose=False,
        )
        assert_same(res, expected)
    assert results[4]["pred_is_empty"] and results[4]["HD_100"] == np.inf
    assert results[5]["ref_is_empty"] and results[5]["DSC"] == 0.0


def test_requested_label_missing_from_both():
    ref, pred = label_maps(3)
    results = compute_metrics_multilabel(to_image(ref, 3), to_image(pred, 3), labels=[2, 9], taus=TAUS, verbose=False)
    assert list(results) == [2, 9]
    assert results[9]["ref_is_empty"] and results[9]["pred_is_empty"]
    assert results[9]["HD_100"] == 0.0 and results[9]["DSC"] == 1.0


def test_numpy_input_matches_sitk():
    ref, pred = label_maps(3)
    spacing = (0.8, 1.1, 1.3)
    from_np = compute_metrics_multilabel(ref, pred, spacing=spacing, taus=TAUS, verbose=False)
    ref_img, pred_img = np2sitk(ref), np2sitk(pred)
    ref_img.SetSpacing(spacing)
    pred_img.SetSpacing(spacing)
    from_sitk = compute_metrics_multilabel(ref_img, pred_img, taus=TAUS, verbose=False)
    for label in from_sitk:
        assert_same(from_np[label], from_sitk[label])


def test_parallel_matches_sequential():
    ref, pred = label_maps(3)
    ref_img, pred_img = to_image(ref, 3), to_image(pred, 3)
    sequential = compute_metrics_multilabel(ref_img, pred_img, taus=TAUS, verbose=False)
    parallel = compute_metrics_multilabel(ref_img, pred_img, taus=TAUS, verbose=False, n_jobs=2)
    assert list(parallel) == list(sequential)
    for label in sequential:
        assert_same(parallel[label], sequential[label])


def test_input_checks():
    ref, pred = label_maps(3)
    ref_img, pred_img = to_image(ref, 3), to_image(pred, 3)
    pred_img.SetOrigin((0.0, 0.0, 0.0))
    with pytest.raises(AssertionError, match="origin"):
        compute_metrics_multilabel(ref_img, pred_img)
    with pytest.raises(AssertionError, match="spacing must be provided"):
        compute_metrics_multilabel(ref, pred)
    with pytest.raises(AssertionError, match="same shape"):
        compute_metrics_multilabel(ref, pred[:-1], spacing=(1.0, 1.0, 1.0))
