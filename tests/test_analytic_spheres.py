"""Validation against the closed-form solution for two non-concentric spheres.

Sphere A (radius 30 mm) is centred at the origin and sphere B (radius 20 mm) at (0, 0, 9),
so B lies entirely inside A. The derivation is given in the Supplementary materials of the
MeshMetrics paper (Note "Analytical case study").
"""

import numpy as np
import pytest
import SimpleITK as sitk
import vtk

from meshmetrics import DistanceMetrics, vtk_voxelizer

R_A, R_B, D_Z = 30.0, 20.0, 9.0


# ---- closed-form solution ----
def dist_a_to_b(cos_theta):
    """Distance from a point on dA (polar angle theta) to dB."""
    return np.sqrt(R_A**2 + D_Z**2 - 2 * D_Z * R_A * cos_theta) - R_B


def dist_b_to_a(cos_theta):
    """Distance from a point on dB (polar angle theta) to dA."""
    return R_A - np.sqrt(R_B**2 + D_Z**2 + 2 * D_Z * R_B * cos_theta)


def analytic_hd(percentile):
    # both distances increase with theta, and the area fraction of the cap up to theta_p
    # is (1 - cos(theta_p)) / 2
    cos_theta_p = 1 - 2 * percentile / 100
    return max(dist_a_to_b(cos_theta_p), dist_b_to_a(cos_theta_p))


def accumulated_distances():
    u = lambda x: 2 / 3 * x**3 - R_B * x**2  # antiderivative in terms of sqrt(u) = x
    v = lambda x: R_A * x**2 - 2 / 3 * x**3
    acc_a = np.pi * R_A / D_Z * (u(R_A + D_Z) - u(R_A - D_Z))
    acc_b = np.pi * R_B / D_Z * (v(R_B + D_Z) - v(R_B - D_Z))
    return acc_a, acc_b


AREA_A, AREA_B = 4 * np.pi * R_A**2, 4 * np.pi * R_B**2


def analytic_masd():
    acc_a, acc_b = accumulated_distances()
    return (acc_a / AREA_A + acc_b / AREA_B) / 2


def analytic_assd():
    acc_a, acc_b = accumulated_distances()
    return (acc_a + acc_b) / (AREA_A + AREA_B)


def analytic_nsd(tau):
    cos_a = np.clip((R_A**2 + D_Z**2 - (R_B + tau) ** 2) / (2 * D_Z * R_A), -1, 1)
    cos_b = np.clip(((R_A - tau) ** 2 - R_B**2 - D_Z**2) / (2 * D_Z * R_B), -1, 1)
    cap_a = 2 * np.pi * R_A**2 * (1 - cos_a)
    cap_b = 2 * np.pi * R_B**2 * (1 - cos_b)
    return (cap_a + cap_b) / (AREA_A + AREA_B)


def ball_intersection_volume(r1, r2, d):
    if d <= abs(r1 - r2):
        return 4 / 3 * np.pi * min(r1, r2) ** 3
    if d >= r1 + r2:
        return 0.0
    return np.pi * (r1 + r2 - d) ** 2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2) / (12 * d)


def analytic_biou(tau):
    shell = lambda r: 4 / 3 * np.pi * (r**3 - (r - tau) ** 3)
    inter = (
        ball_intersection_volume(R_A, R_B, D_Z)
        - ball_intersection_volume(R_A, R_B - tau, D_Z)
        - ball_intersection_volume(R_A - tau, R_B, D_Z)
        + ball_intersection_volume(R_A - tau, R_B - tau, D_Z)
    )
    return inter / (shell(R_A) + shell(R_B) - inter)


ANALYTIC = {
    "HD_100": analytic_hd(100),
    "HD_95": analytic_hd(95),
    "MASD": analytic_masd(),
    "ASSD": analytic_assd(),
    "NSD_2": analytic_nsd(2.0),
    "NSD_3": analytic_nsd(3.0),
    "BIoU_2": analytic_biou(2.0),
    "BIoU_3": analytic_biou(3.0),
}


def all_metrics(m):
    return {
        "HD_100": m.hd(),
        "HD_95": m.hd(percentile=95),
        "MASD": m.masd(),
        "ASSD": m.assd(),
        "NSD_2": m.nsd(2.0),
        "NSD_3": m.nsd(3.0),
        "BIoU_2": m.biou(2.0),
        "BIoU_3": m.biou(3.0),
    }


def sphere_mesh(center, radius, resolution=128):
    src = vtk.vtkSphereSource()
    src.SetCenter(*center)
    src.SetRadius(radius)
    src.SetThetaResolution(resolution)
    src.SetPhiResolution(resolution)
    src.Update()
    return src.GetOutput()


def test_analytic_solution_matches_paper():
    """Values reported in the Supplementary materials."""
    assert ANALYTIC["HD_100"] == pytest.approx(19.0)
    assert dist_a_to_b(-0.9) == pytest.approx(18.3014, abs=1e-4)
    assert dist_b_to_a(-0.9) == pytest.approx(17.4700, abs=1e-4)
    assert ANALYTIC["HD_95"] == pytest.approx(18.3014, abs=1e-4)
    assert accumulated_distances() == pytest.approx((39240 * np.pi, 13840 * np.pi))
    assert ANALYTIC["MASD"] == pytest.approx(9.775)
    assert ANALYTIC["ASSD"] == pytest.approx(10.2077, abs=1e-4)
    assert ANALYTIC["NSD_2"] == pytest.approx(0.0519, abs=1e-4)
    assert ANALYTIC["NSD_3"] == pytest.approx(0.1043, abs=1e-4)
    assert ANALYTIC["BIoU_2"] == pytest.approx(194.52 / 30031.79, abs=1e-5)


def test_mesh_input_matches_analytic():
    """Native mesh input reproduces the closed-form solution (up to the sphere triangulation)."""
    m = DistanceMetrics()
    m.set_input(
        sphere_mesh((0, 0, 0), R_A), sphere_mesh((0, 0, D_Z), R_B), spacing=(1.0, 1.0, 1.0)
    )
    res = all_metrics(m)
    for k, expected in ANALYTIC.items():
        tol = 0.02 if k.startswith(("HD", "MASD", "ASSD")) else 0.002  # mm | fraction
        assert res[k] == pytest.approx(expected, abs=tol), k


def test_mask_input_close_to_analytic():
    """Mask input (spheres voxelized on a 1 mm grid) stays close to the closed-form solution."""
    spacing = (1.0, 1.0, 1.0)
    meta = sitk.Image([66, 66, 66], sitk.sitkUInt8)
    meta.SetSpacing(spacing)
    meta.SetOrigin((-33.0, -33.0, -33.0))
    ref = vtk_voxelizer(sphere_mesh((0, 0, 0), R_A, resolution=256), meta)
    pred = vtk_voxelizer(sphere_mesh((0, 0, D_Z), R_B, resolution=256), meta)

    m = DistanceMetrics()
    m.set_input(ref, pred)
    res = all_metrics(m)
    for k, expected in ANALYTIC.items():
        tol = 0.5 if k.startswith(("HD", "MASD", "ASSD")) else 0.02  # mm | fraction
        assert res[k] == pytest.approx(expected, abs=tol), k
