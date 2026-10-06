"""Closed-form distance-based segmentation metrics for two non-concentric spheres.

Reference configuration (as derived in the response to Reviewer #1, Comment R1-1):

    sphere A: center O_A = (0, 0, 0),   radius R_A
    sphere B: center O_B = (0, 0, d_z), radius R_B

Sphere B is assumed to be strictly contained in sphere A, i.e. d_z < R_A - R_B.
Under that assumption the point-to-surface distances are

    d(a, dB) = sqrt(R_A^2 + d_z^2 - 2 d_z R_A cos(theta)) - R_B     for a in dA
    d(b, dA) = R_A - sqrt(R_B^2 + d_z^2 + 2 d_z R_B cos(theta))     for b in dB

and both are monotonically increasing in the polar angle theta on [0, pi], which is
what makes the percentile and tolerance-based metrics analytically tractable.

Every metric below is an exact closed form -- no meshing, no voxelization, no
numerical integration.

Usage:
    python analytical_metrics_spheres.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, isclose, pi, sqrt

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoSpheres:
    """Two non-concentric spheres, B contained in A, offset along the z-axis."""

    R_A: float
    R_B: float
    d_z: float

    def __post_init__(self) -> None:
        if not (self.R_A > 0 and self.R_B > 0):
            raise ValueError("radii must be positive")
        if self.d_z < 0:
            raise ValueError("d_z must be non-negative")
        if self.d_z >= self.R_A - self.R_B:
            raise ValueError(
                f"containment requires d_z < R_A - R_B "
                f"({self.d_z} < {self.R_A - self.R_B} is violated)"
            )

    # --- surface areas and volumes -----------------------------------------

    @property
    def area_A(self) -> float:
        """|dA| = 4 pi R_A^2."""
        return 4.0 * pi * self.R_A**2

    @property
    def area_B(self) -> float:
        """|dB| = 4 pi R_B^2."""
        return 4.0 * pi * self.R_B**2

    # --- point-to-surface distances ----------------------------------------

    def d_A_to_B(self, theta: float) -> float:
        """Distance from a point on dA at polar angle theta to dB."""
        return sqrt(self.R_A**2 + self.d_z**2 - 2 * self.d_z * self.R_A * cos(theta)) - self.R_B

    def d_B_to_A(self, theta: float) -> float:
        """Distance from a point on dB at polar angle theta to dA."""
        return self.R_A - sqrt(self.R_B**2 + self.d_z**2 + 2 * self.d_z * self.R_B * cos(theta))


# ---------------------------------------------------------------------------
# accumulated (area-integrated) directed distances -- eqs. (1) and (2)
# ---------------------------------------------------------------------------


def accumulated_A_to_B(s: TwoSpheres) -> float:
    """Integral of d(a, dB) over dA, evaluated in closed form.

    Substitution u = R_A^2 + d_z^2 - 2 d_z R_A cos(theta) turns the surface
    integral into (pi R_A / d_z) * int (sqrt(u) - R_B) du over
    [(R_A - d_z)^2, (R_A + d_z)^2].
    """

    def F(u: float) -> float:
        return 2.0 * u**1.5 / 3.0 - s.R_B * u

    lo, hi = (s.R_A - s.d_z) ** 2, (s.R_A + s.d_z) ** 2
    return pi * s.R_A / s.d_z * (F(hi) - F(lo))


def accumulated_B_to_A(s: TwoSpheres) -> float:
    """Integral of d(b, dA) over dB, evaluated in closed form.

    Substitution v = R_B^2 + d_z^2 + 2 d_z R_B cos(theta).
    """

    def F(v: float) -> float:
        return s.R_A * v - 2.0 * v**1.5 / 3.0

    lo, hi = (s.R_B - s.d_z) ** 2, (s.R_B + s.d_z) ** 2
    return pi * s.R_B / s.d_z * (F(hi) - F(lo))


def mean_A_to_B(s: TwoSpheres) -> float:
    """Area-weighted mean of d(a, dB) over dA."""
    return accumulated_A_to_B(s) / s.area_A


def mean_B_to_A(s: TwoSpheres) -> float:
    """Area-weighted mean of d(b, dA) over dB."""
    return accumulated_B_to_A(s) / s.area_B


# ---------------------------------------------------------------------------
# HD_p -- percentile Hausdorff distance
# ---------------------------------------------------------------------------


def _theta_at_percentile(p: float) -> float:
    """Polar angle enclosing the fraction p of a sphere's surface area.

    The spherical cap up to theta_p covers a fraction F(theta_p) = (1 - cos
    theta_p) / 2 of the total area, hence cos theta_p = 1 - 2 p.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return acos(max(-1.0, min(1.0, 1.0 - 2.0 * p)))


def hd_directed(s: TwoSpheres, p: float = 0.95) -> tuple[float, float]:
    """The p-th percentile of each directed point-to-surface distance."""
    theta_p = _theta_at_percentile(p)
    return s.d_A_to_B(theta_p), s.d_B_to_A(theta_p)


def hd_percentile(s: TwoSpheres, p: float = 0.95) -> float:
    """HD_p = max over the two directions (p = 1.0 gives the exact HD)."""
    return max(hd_directed(s, p))


# ---------------------------------------------------------------------------
# MASD and ASSD
# ---------------------------------------------------------------------------


def masd(s: TwoSpheres) -> float:
    """Mean of the two mean directed distances."""
    return (mean_A_to_B(s) + mean_B_to_A(s)) / 2.0


def assd(s: TwoSpheres) -> float:
    """Accumulated distances divided by the combined surface area."""
    return (accumulated_A_to_B(s) + accumulated_B_to_A(s)) / (s.area_A + s.area_B)


# ---------------------------------------------------------------------------
# NSD -- normalized surface distance at tolerance tau
# ---------------------------------------------------------------------------


def _cos_theta_A_within(s: TwoSpheres, tau: float) -> float:
    """cos(theta_A) bound for d(a, dB) <= tau."""
    return (s.R_A**2 + s.d_z**2 - (s.R_B + tau) ** 2) / (2 * s.d_z * s.R_A)


def _cos_theta_B_within(s: TwoSpheres, tau: float) -> float:
    """cos(theta_B) bound for d(b, dA) <= tau."""
    return ((s.R_A - tau) ** 2 - s.R_B**2 - s.d_z**2) / (2 * s.d_z * s.R_B)


def _cap_area(R: float, cos_theta: float) -> float:
    """Area of the spherical cap up to theta, clamped to [0, 4 pi R^2]."""
    return 2.0 * pi * R**2 * (1.0 - max(-1.0, min(1.0, cos_theta)))


def nsd(s: TwoSpheres, tau: float) -> float:
    """Fraction of the combined boundary within tolerance tau of the other."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    within = _cap_area(s.R_A, _cos_theta_A_within(s, tau)) + _cap_area(
        s.R_B, _cos_theta_B_within(s, tau)
    )
    return within / (s.area_A + s.area_B)


# ---------------------------------------------------------------------------
# BIoU -- boundary intersection over union at tolerance tau
# ---------------------------------------------------------------------------


def _cap_volume(R: float, h: float) -> float:
    """Volume of a spherical cap of height h on a ball of radius R."""
    return pi * h**2 * (3.0 * R - h) / 3.0


def _ball_intersection(r_1: float, r_2: float, d: float) -> float:
    """Volume of the intersection of two balls whose centers are d apart.

    Three regimes:
      * disjoint     (d >= r_1 + r_2)      -> 0
      * containment  (d <= |r_1 - r_2|)    -> the smaller ball
      * partial      otherwise             -> lens = sum of two spherical caps,
        whose base plane sits at (d^2 + r_1^2 - r_2^2) / (2 d) from center 1
    """
    if d >= r_1 + r_2:
        return 0.0
    if d <= abs(r_1 - r_2):
        return 4.0 / 3.0 * pi * min(r_1, r_2) ** 3
    h_1 = r_1 - (d**2 + r_1**2 - r_2**2) / (2 * d)
    h_2 = r_2 - (d**2 + r_2**2 - r_1**2) / (2 * d)
    return _cap_volume(r_1, h_1) + _cap_volume(r_2, h_2)


def _shell_volume(R: float, tau: float) -> float:
    """Volume of the shell of thickness tau measured inwards from radius R."""
    return 4.0 / 3.0 * pi * (R**3 - max(0.0, R - tau) ** 3)


def biou_volumes(s: TwoSpheres, tau: float) -> tuple[float, float, float]:
    """Return (intersection, union, |A_tau| + |B_tau|) of the two shells.

    Each shell is the difference of two concentric balls, so the intersection
    follows by inclusion-exclusion over the four resulting balls.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    rA_in, rB_in = max(0.0, s.R_A - tau), max(0.0, s.R_B - tau)
    inter = (
        _ball_intersection(s.R_A, s.R_B, s.d_z)
        - _ball_intersection(s.R_A, rB_in, s.d_z)
        - _ball_intersection(rA_in, s.R_B, s.d_z)
        + _ball_intersection(rA_in, rB_in, s.d_z)
    )
    vol_A, vol_B = _shell_volume(s.R_A, tau), _shell_volume(s.R_B, tau)
    return inter, vol_A + vol_B - inter, vol_A + vol_B


def biou(s: TwoSpheres, tau: float) -> float:
    """Boundary IoU of the two inward shells of thickness tau."""
    inter, union, _ = biou_volumes(s, tau)
    return inter / union


# ---------------------------------------------------------------------------
# convenience wrapper
# ---------------------------------------------------------------------------


def all_metrics(
    R_A: float, R_B: float, d_z: float, tau: float, p: float = 0.95
) -> dict[str, float]:
    """Every metric for the given configuration.

    Distances are in the same length unit as the radii; NSD and BIoU are
    fractions in [0, 1].
    """
    s = TwoSpheres(R_A=R_A, R_B=R_B, d_z=d_z)
    d_ab, d_ba = hd_directed(s, p)
    inter, union, _ = biou_volumes(s, tau)
    return {
        "HD": hd_percentile(s, 1.0),
        f"HD_{p:g}_A_to_B": d_ab,
        f"HD_{p:g}_B_to_A": d_ba,
        f"HD_{p:g}": max(d_ab, d_ba),
        "mean_A_to_B": mean_A_to_B(s),
        "mean_B_to_A": mean_B_to_A(s),
        "MASD": masd(s),
        "ASSD": assd(s),
        f"NSD_{tau:g}": nsd(s, tau),
        f"BIoU_{tau:g}": biou(s, tau),
        "shell_intersection": inter,
        "shell_union": union,
    }


# ---------------------------------------------------------------------------
# reference configuration from the manuscript
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    R_A, R_B, d_z = 30.0, 20.0, 9.0
    s = TwoSpheres(R_A=R_A, R_B=R_B, d_z=d_z)

    print(f"R_A = {R_A} mm, R_B = {R_B} mm, d_z = {d_z} mm\n")

    print("accumulated directed distances")
    print(f"  d_A->B = {accumulated_A_to_B(s):14.4f} mm^3 = {accumulated_A_to_B(s) / pi:.1f} pi")
    print(f"  d_B->A = {accumulated_B_to_A(s):14.4f} mm^3 = {accumulated_B_to_A(s) / pi:.1f} pi")
    print(f"  |dA|   = {s.area_A:14.4f} mm^2 = {s.area_A / pi:.1f} pi")
    print(f"  |dB|   = {s.area_B:14.4f} mm^2 = {s.area_B / pi:.1f} pi\n")

    print("metrics")
    d_ab, d_ba = hd_directed(s, 0.95)
    print(f"  HD           = {hd_percentile(s, 1.0):10.4f} mm")
    print(f"  HD_95 A->B   = {d_ab:10.4f} mm")
    print(f"  HD_95 B->A   = {d_ba:10.4f} mm")
    print(f"  HD_95        = {hd_percentile(s, 0.95):10.4f} mm")
    print(f"  mean A->B    = {mean_A_to_B(s):10.4f} mm")
    print(f"  mean B->A    = {mean_B_to_A(s):10.4f} mm")
    print(f"  MASD         = {masd(s):10.4f} mm")
    print(f"  ASSD         = {assd(s):10.4f} mm")
    for tau in (2.0, 3.0):
        print(f"  NSD_{tau:g}        = {100 * nsd(s, tau):10.4f} %")
    for tau in (2.0, 3.0):
        inter, union, _ = biou_volumes(s, tau)
        print(
            f"  BIoU_{tau:g}       = {100 * biou(s, tau):10.4f} %"
            f"   (inter {inter:.4f}, union {union:.4f} mm^3)"
        )

    # values reported in the response to Reviewer #1, Comment R1-1
    print("\nagreement with the reported values")
    checks = [
        ("d_A->B / pi", accumulated_A_to_B(s) / pi, 39240.0, 1e-6),
        ("d_B->A / pi", accumulated_B_to_A(s) / pi, 13840.0, 1e-6),
        ("mean A->B", mean_A_to_B(s), 10.9, 1e-9),
        ("mean B->A", mean_B_to_A(s), 8.65, 1e-9),
        ("HD_95", hd_percentile(s, 0.95), 18.3014, 1e-4),
        ("MASD", masd(s), 9.775, 1e-9),
        ("ASSD", assd(s), 10.2077, 1e-4),
        ("NSD_2 [%]", 100 * nsd(s, 2.0), 5.19, 1e-2),
        ("NSD_3 [%]", 100 * nsd(s, 3.0), 10.43, 1e-2),
        ("shell inter", biou_volumes(s, 2.0)[0], 194.52, 1e-2),
        ("shell union", biou_volumes(s, 2.0)[1], 30031.79, 1e-2),
        ("BIoU_2 [%]", 100 * biou(s, 2.0), 0.65, 1e-2),
    ]
    for name, got, want, tol in checks:
        ok = isclose(got, want, rel_tol=0, abs_tol=tol)
        print(f"  {'OK ' if ok else 'FAIL'} {name:<12} {got:14.6f}  (reported {want})")
