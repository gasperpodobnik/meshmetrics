import gc
import warnings
import weakref

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R

from meshmetrics import DistanceMetrics, np2sitk, vtk_meshing, vtk_voxelizer

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
    """Rigidly rotating the image grid must not change any metric."""
    origin = (10.0, -5.0, 3.0)
    ref = DistanceMetrics()
    ref.set_input(to_image(A, origin=origin), to_image(B, origin=origin))
    expected = all_metrics(ref, tau=1.5)
    for direction in R.random(3, random_state=0).as_matrix():
        m = DistanceMetrics()
        m.set_input(
            to_image(A, origin=origin, direction=direction),
            to_image(B, origin=origin, direction=direction),
        )
        res = all_metrics(m, tau=1.5)
        for k in expected:
            assert res[k] == pytest.approx(expected[k], rel=1e-6), k


@pytest.mark.parametrize("mesh_side", ["ref", "pred"])
def test_mixed_input_keeps_image_grid(mesh_side):
    img_a, img_b = to_image(A), to_image(B)
    both_sitk = DistanceMetrics()
    both_sitk.set_input(img_a, img_b)
    expected = all_metrics(both_sitk)

    m = DistanceMetrics()
    if mesh_side == "pred":
        m.set_input(img_a, vtk_meshing(img_b))
        given, kept = img_a, m.ref_sitk
    else:
        m.set_input(vtk_meshing(img_a), img_b)
        given, kept = img_b, m.pred_sitk
    # the image grid is kept (cropped): same spacing/direction, voxel centres on the given grid
    assert kept.GetSpacing() == given.GetSpacing()
    assert kept.GetDirection() == given.GetDirection()
    start = given.TransformPhysicalPointToContinuousIndex(kept.GetOrigin())
    np.testing.assert_allclose(start, np.round(start), atol=1e-9)
    assert sitk.GetArrayFromImage(kept).sum() == sitk.GetArrayFromImage(given).sum()
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
    # 1 mm voxels: centres are >= 0.5/sqrt(3) (corner cut of marching cubes) from the surface
    m.set_input(to_image(A), to_image(B))
    with pytest.raises(ValueError, match="BIoU is undefined for tau=0.2"):
        m.biou(0.2)
    assert 0 < m.biou(1.0) < 1


def test_compute_metrics_matches_class():
    from meshmetrics import compute_metrics

    ref, pred = to_image(A, (0.8, 1.0, 1.3)), to_image(B, (0.8, 1.0, 1.3))
    res = compute_metrics(ref, pred, taus=(1.0, 2.5))
    assert list(res) == [
        "ref_is_empty", "pred_is_empty", "HD_100", "HD_95", "MASD", "ASSD",
        "NSD_1.0", "NSD_2.5", "BIoU_1.0", "BIoU_2.5", "DSC", "IoU",
    ]  # fmt: skip
    m = DistanceMetrics()
    m.set_input(ref, pred)
    expected = {
        "HD_100": m.hd(), "HD_95": m.hd(95), "MASD": m.masd(), "ASSD": m.assd(),
        "NSD_1.0": m.nsd(1.0), "NSD_2.5": m.nsd(2.5), "BIoU_1.0": m.biou(1.0),
        "BIoU_2.5": m.biou(2.5), "DSC": m.dsc(), "IoU": m.iou(),
    }  # fmt: skip
    for k, v in expected.items():
        assert res[k] == pytest.approx(v), k
    assert res["ref_is_empty"] is False and res["pred_is_empty"] is False


def test_compute_metrics_defaults_subset_and_numpy_2d():
    from meshmetrics import compute_metrics

    res = compute_metrics(A[:, :, 12], B[:, :, 12], spacing=(0.7, 1.1))
    assert not any(k.startswith(("NSD", "BIoU")) for k in res)  # no taus given
    assert {"HD_100", "HD_95", "MASD", "ASSD", "DSC", "IoU"} <= set(res)

    res = compute_metrics(
        A[:, :, 12], B[:, :, 12], spacing=(0.7, 1.1), metrics=["HD", "nsd"], taus=[2], percentiles=[50]
    )
    assert list(res) == ["ref_is_empty", "pred_is_empty", "HD_50", "NSD_2.0"]


def test_compute_metrics_errors_and_empty():
    from meshmetrics import compute_metrics

    with pytest.raises(ValueError, match="Unknown metrics"):
        compute_metrics(to_image(A), to_image(B), metrics=["hd", "dice"])
    with pytest.raises(ValueError, match="need at least one tolerance"):
        compute_metrics(to_image(A), to_image(B), metrics=["biou"])
    res = compute_metrics(to_image(A), to_image(np.zeros_like(B)), taus=[1.0], verbose=False)
    assert res["pred_is_empty"] and res["HD_100"] == np.inf and res["NSD_1.0"] == 0.0


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("touch", ["inside", "one_edge", "corner"])
def test_identical_masks_touching_image_border(ndim, touch):
    """Identical masks must give perfect scores, also when they touch the image border."""
    a = np.zeros((20,) * ndim, np.uint8)
    if touch == "inside":
        a[(slice(5, 15),) * ndim] = 1
    elif touch == "one_edge":
        a[(slice(0, 12),) + (slice(5, 15),) * (ndim - 1)] = 1
    else:
        a[(slice(0, 12),) * ndim] = 1
    img = sitk.GetImageFromArray(a)
    img.SetSpacing([0.8, 1.3, 1.1][:ndim])
    m = DistanceMetrics()
    m.set_input(img, img)
    assert m.hd() == pytest.approx(0, abs=1e-5)  # float32 mesh points
    assert m.masd() == pytest.approx(0, abs=1e-5)  # float32 mesh points
    assert m.assd() == pytest.approx(0, abs=1e-5)  # float32 mesh points
    assert m.nsd(1.0) == pytest.approx(1.0)
    assert m.biou(1.0) == pytest.approx(1.0)
    assert m.dsc() == pytest.approx(1.0)


def _embed(mask, shape, offset):
    big = np.zeros(shape, mask.dtype)
    big[tuple(slice(o, o + n) for o, n in zip(offset, mask.shape))] = mask
    return big


@pytest.mark.parametrize("ndim", [2, 3])
def test_results_independent_of_image_size(ndim):
    """Masks are cropped internally: embedding them in a larger image changes nothing."""
    a = A if ndim == 3 else A[:, :, 12]
    b = B if ndim == 3 else B[:, :, 12]
    spacing = (0.8, 1.0, 1.3)[:ndim]
    direction = R.random(random_state=1).as_matrix() if ndim == 3 else np.array([[0.6, -0.8], [0.8, 0.6]])
    results = []
    for shape, offset in [(a.shape, (0,) * ndim), ((70, 55, 64)[:ndim], (21, 7, 30)[:ndim])]:
        imgs = [to_image(_embed(x, shape, offset), spacing, (0.0,) * ndim, direction) for x in (a, b)]
        m = DistanceMetrics()
        m.set_input(*imgs)
        # tau=1.73: box masks on these grids have many distances exactly at round values
        # (e.g. 1.0, 1.5), where rounding decides whether `distance <= tau`
        results.append(all_metrics(m, tau=1.73))
    for k in results[0]:
        assert results[1][k] == pytest.approx(results[0][k], abs=1e-5), k


def test_crop_to_foreground():
    from meshmetrics.utils import crop_np_to_foreground, crop_to_foreground

    big = _embed(A.astype(np.uint8), (70, 55, 64), (21, 7, 30))
    img = to_image(big, (0.8, 1.0, 1.3), (5.0, -3.0, 2.0), R.random(random_state=2).as_matrix())
    (cropped,) = crop_to_foreground([img], margin=1)
    # sitk2np order (x, y, z): A's foreground spans [8,20) x [10,22) x [5,25), shifted by the offset
    assert cropped.GetSize() == (12 + 2, 12 + 2, 20 + 2)
    assert sitk.GetArrayFromImage(cropped).sum() == big.sum()
    # physical position of the voxels is preserved
    np.testing.assert_allclose(cropped.GetOrigin(), img.TransformIndexToPhysicalPoint((21 + 8 - 1, 7 + 10 - 1, 30 + 5 - 1)))
    # empty masks and masks filling the image are returned unchanged
    empty = to_image(np.zeros((5, 6, 7), bool))
    assert crop_to_foreground([empty])[0] is empty
    full = to_image(np.ones((5, 6, 7), bool))
    assert crop_to_foreground([full])[0] is full
    # numpy: margin is clipped at the array border
    arr = np.zeros((10, 10), bool)
    arr[0:3, 4:6] = True
    (c,) = crop_np_to_foreground([arr], margin=2)
    assert c.shape == (5, 6) and c.sum() == arr.sum()


def _circle_contour(radius, n=720, polyline=False):
    import vtk

    pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        pts.InsertNextPoint(radius * np.cos(t), radius * np.sin(t), 0.0)
    if polyline:  # one closed polyline instead of 2-point segments
        lines.InsertNextCell(n + 1, list(range(n)) + [0])
    else:
        for i in range(n):
            lines.InsertNextCell(2, [i, (i + 1) % n])
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    return poly


@pytest.mark.parametrize("spacing", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("polyline", [False, True])
def test_2D_contour_inputs_exact(spacing, polyline):
    """Concentric circles (r = 10, 12): all distances are 2, independent of the grid spacing."""
    m = DistanceMetrics()
    m.set_input(
        _circle_contour(10, polyline=polyline), _circle_contour(12, polyline=polyline), spacing=(spacing, spacing)
    )
    for value in (m.hd(), m.hd(95), m.masd(), m.assd()):
        assert value == pytest.approx(2.0, abs=1e-3)  # inscribed 720-gons
    assert m.nsd(2.5) == pytest.approx(1.0)
    assert m.nsd(1.5) == pytest.approx(0.0)


def test_2D_contour_biou_close_to_analytic():
    """BIoU of concentric circles: annuli (r - tau, r], computed on a fine grid."""
    tau, r1, r2 = 2.5, 10.0, 12.0
    inter = np.pi * (r1**2 - (r2 - tau) ** 2)
    union = np.pi * (r1**2 - (r1 - tau) ** 2) + np.pi * (r2**2 - (r2 - tau) ** 2) - inter
    m = DistanceMetrics()
    m.set_input(_circle_contour(r1), _circle_contour(r2), spacing=(0.1, 0.1))
    assert m.biou(tau) == pytest.approx(inter / union, abs=0.005)


def test_2D_mixed_image_and_contour():
    """Image + contour input measures to the given contour."""
    spacing = 0.25
    contour = _circle_contour(12)
    img = vtk_voxelizer(contour, spacing=(spacing, spacing))  # defines the grid
    exact = DistanceMetrics()
    exact.set_input(_circle_contour(10), contour, spacing=(spacing, spacing))
    m = DistanceMetrics()
    m.set_input(vtk_voxelizer(_circle_contour(10), img), contour)
    # the image side is a voxelized circle, so it deviates from r = 10 by up to half a pixel
    assert m.hd() == pytest.approx(exact.hd(), abs=spacing)
    assert m.masd() == pytest.approx(exact.masd(), abs=spacing / 2)
