import gc
import warnings
import weakref

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R

from meshmetrics import DistanceMetrics, np2sitk, vtk_meshing

A = np.zeros((30, 30, 30), bool)
A[8:20, 10:22, 5:25] = True
B = np.zeros_like(A)
B[9:22, 9:20, 6:27] = True


def to_image(mask, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None):
    img = np2sitk(mask.astype(np.uint8))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(np.asarray(direction).ravel().tolist())
    return img


def all_metrics(m, tau=1.0):
    return {
        "hd": m.hd(),
        "hd95": m.hd(percentile=95),
        "masd": m.masd(),
        "assd": m.assd(),
        "nsd": m.nsd(tau),
        "biou": m.biou(tau),
        "dsc": m.dsc(),
        "iou": m.iou(),
    }


def test_numpy_and_sitk_inputs_agree():
    spacing = (0.8, 1.0, 1.3)
    m_np, m_sitk = DistanceMetrics(), DistanceMetrics()
    m_np.set_input(A, B, spacing=spacing)
    m_sitk.set_input(to_image(A, spacing), to_image(B, spacing))
    np_res, sitk_res = all_metrics(m_np), all_metrics(m_sitk)
    for k in np_res:
        assert np_res[k] == pytest.approx(sitk_res[k]), k


def test_metrics_invariant_to_direction():
    """Rigidly rotating the image grid must not change grid-based metrics.

    Surface-distance metrics (HD, NSD, ...) are excluded: vtkSurfaceNets3D picks the
    quad-splitting diagonal from floating-point ties, so triangle centroids (and
    hence those metrics) depend slightly on grid orientation.
    """
    origin = (10.0, -5.0, 3.0)
    ref = DistanceMetrics()
    ref.set_input(to_image(A, origin=origin), to_image(B, origin=origin))
    expected = all_metrics(ref)
    for direction in R.random(3, random_state=0).as_matrix():
        m = DistanceMetrics()
        m.set_input(
            to_image(A, origin=origin, direction=direction),
            to_image(B, origin=origin, direction=direction),
        )
        res = all_metrics(m)
        for k in ("dsc", "iou"):
            assert res[k] == pytest.approx(expected[k]), k
        assert res["biou"] == pytest.approx(expected["biou"], abs=0.01)


@pytest.mark.parametrize("mesh_side", ["ref", "pred"])
def test_mixed_input_keeps_image_grid(mesh_side):
    img_a, img_b = to_image(A), to_image(B)
    both_sitk = DistanceMetrics()
    both_sitk.set_input(img_a, img_b)
    expected = all_metrics(both_sitk)

    m = DistanceMetrics()
    if mesh_side == "pred":
        m.set_input(img_a, vtk_meshing(img_b))
        assert m.ref_sitk is img_a
    else:
        m.set_input(vtk_meshing(img_a), img_b)
        assert m.pred_sitk is img_b
    res = all_metrics(m)
    for k in expected:
        assert res[k] == pytest.approx(expected[k]), k


def test_mixed_input_warns_when_mesh_outside_image():
    small = to_image(A[:15, :15, :15])
    m = DistanceMetrics()
    with pytest.warns(UserWarning, match="extends beyond"):
        m.set_input(small, vtk_meshing(to_image(B)))


def test_verbose_preserved():
    m = DistanceMetrics(verbose=False)
    m.set_input(to_image(A), to_image(np.zeros_like(B)))
    assert m.verbose is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert m.hd() == np.inf


def test_cache_is_per_instance():
    m1, m2 = DistanceMetrics(), DistanceMetrics()
    m1.set_input(to_image(A), to_image(B))
    m2.set_input(to_image(A), to_image(A))
    assert m1.hd() > 0 and m2.hd() == pytest.approx(0, abs=1e-9)

    # set_input on the same instance recomputes
    m1.set_input(to_image(A), to_image(A))
    assert m1.hd() == pytest.approx(0, abs=1e-9)


def test_instance_released_after_del():
    m = DistanceMetrics()
    m.set_input(to_image(A), to_image(B))
    m.hd()
    ref = weakref.ref(m)
    del m
    gc.collect()
    assert ref() is None


def test_spacing_float_noise_accepted():
    m = DistanceMetrics()
    m.set_input(to_image(A, (0.8, 1.0, 1.2)), to_image(B, (0.8 + 1e-9, 1.0, 1.2)))
    assert np.isfinite(m.hd())


def test_tau_accepts_numpy_floats():
    m = DistanceMetrics()
    m.set_input(to_image(A), to_image(B))
    assert m.nsd(np.float32(1.0)) == pytest.approx(m.nsd(1.0))
    assert m.biou(np.float32(1.0)) == pytest.approx(m.biou(1.0))


def test_invalid_input_message():
    m = DistanceMetrics()
    with pytest.raises(ValueError, match="got str and Image"):
        m.set_input("ref", to_image(B))


def test_both_empty_meshes():
    import vtk

    m = DistanceMetrics(verbose=False)
    m.set_input(vtk.vtkPolyData(), vtk.vtkPolyData(), spacing=(1.0, 1.0, 1.0))
    assert m.hd() == 0.0 and m.dsc() == 1.0


def test_old_import_alias():
    with pytest.warns(DeprecationWarning):
        import importlib
        import sys

        sys.modules.pop("MeshMetrics", None)
        old = importlib.import_module("MeshMetrics")
    import meshmetrics

    assert old.DistanceMetrics is meshmetrics.DistanceMetrics
    from MeshMetrics.utils import vtk_voxelizer

    assert vtk_voxelizer is meshmetrics.vtk_voxelizer


def full_field_biou(m, tau):
    ref_mask, ref_field, pred_mask, pred_field = m.img_dist_field
    ref_hollow = (ref_field < tau) & ref_mask.astype(bool)
    pred_hollow = (pred_field < tau) & pred_mask.astype(bool)
    with np.errstate(invalid="ignore"):
        return (ref_hollow & pred_hollow).sum() / (ref_hollow | pred_hollow).sum()


def _inputs(kind):
    direction = R.random(random_state=5).as_matrix()
    if kind == "numpy_3d":
        return dict(ref=A, pred=B, spacing=(0.8, 1.0, 1.3))
    if kind == "numpy_2d":
        return dict(ref=A[:, :, 12], pred=B[:, :, 12], spacing=(0.7, 1.1))
    if kind == "sitk_rotated":
        return dict(
            ref=to_image(A, (0.8, 1.0, 1.3), (5.0, -3.0, 2.0), direction),
            pred=to_image(B, (0.8, 1.0, 1.3), (5.0, -3.0, 2.0), direction),
        )
    if kind == "mixed":
        return dict(ref=to_image(A), pred=vtk_meshing(to_image(B)))
    if kind == "meshes":
        return dict(ref=vtk_meshing(to_image(A)), pred=vtk_meshing(to_image(B)), spacing=(1.0, 1.0, 1.0))


@pytest.mark.parametrize("kind", ["numpy_3d", "numpy_2d", "sitk_rotated", "mixed", "meshes"])
def test_narrow_band_biou_matches_full_field(kind):
    taus = [3.0, 1.0, 2.0, 1.5]  # largest first: smaller taus reuse the cached band
    m = DistanceMetrics()
    m.set_input(**_inputs(kind))
    band = [m.biou(t) for t in taus]

    full = DistanceMetrics()
    full.set_input(**_inputs(kind))
    expected = [full_field_biou(full, t) for t in taus]
    assert band == pytest.approx(expected)


def test_narrow_band_cache_grows_with_tau():
    m = DistanceMetrics()
    m.set_input(to_image(A), to_image(B))
    m.biou(2.0)
    assert m._dist_fields[0] == 2.0
    m.biou(1.0)  # reuses the cached band
    assert m._dist_fields[0] == 2.0
    m.biou(3.0)
    assert m._dist_fields[0] == 3.0
    m.set_input(to_image(A), to_image(A))
    assert m._dist_fields is None


def test_biou_raises_for_too_small_tau():
    m = DistanceMetrics()
    m.set_input(to_image(A), to_image(B))  # 1 mm voxels: centres are >= 0.5 from the surface
    with pytest.raises(ValueError, match="BIoU is undefined for tau=0.5"):
        m.biou(0.5)
    assert 0 < m.biou(1.0) < 1
