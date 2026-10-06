"""Validation against the closed-form solution for two non-concentric spheres.

Sphere A (radius 30 mm) is centred at the origin and sphere B (radius 20 mm) at (0, 0, 9),
so B lies entirely inside A. The derivation is given in the Supplementary materials of the
MeshMetrics paper (Note "Analytical case study"); the closed-form solution is in
`analytical_metrics_spheres.py`, the implementation used for the paper.
"""

import numpy as np
import pytest
import SimpleITK as sitk
import vtk

from analytical_metrics_spheres import (
    TwoSpheres,
    accumulated_A_to_B,
    accumulated_B_to_A,
    assd,
    biou,
    hd_directed,
    hd_percentile,
    masd,
    nsd,
)
from meshmetrics import DistanceMetrics, vtk_voxelizer

R_A, R_B, D_Z = 30.0, 20.0, 9.0
SPHERES = TwoSpheres(R_A=R_A, R_B=R_B, d_z=D_Z)

ANALYTIC = {
    "HD_100": hd_percentile(SPHERES, 1.0),
    "HD_95": hd_percentile(SPHERES, 0.95),
    "MASD": masd(SPHERES),
    "ASSD": assd(SPHERES),
    "NSD_2": nsd(SPHERES, 2.0),
    "NSD_3": nsd(SPHERES, 3.0),
    "BIoU_2": biou(SPHERES, 2.0),
    "BIoU_3": biou(SPHERES, 3.0),
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
    assert hd_directed(SPHERES, 0.95) == pytest.approx((18.3014, 17.4700), abs=1e-4)
    assert ANALYTIC["HD_95"] == pytest.approx(18.3014, abs=1e-4)
    assert accumulated_A_to_B(SPHERES) == pytest.approx(39240 * np.pi)
    assert accumulated_B_to_A(SPHERES) == pytest.approx(13840 * np.pi)
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
