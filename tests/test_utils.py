import numpy as np
import pytest
import SimpleITK as sitk
import vtk
from scipy.spatial.transform import Rotation as R

from meshmetrics import (
    meshio_to_vtk,
    np2sitk,
    sitk2np,
    vtk_2D_meshing,
    vtk_3D_meshing,
    vtk_meshing,
    vtk_read_polydata,
    vtk_voxelizer,
    vtk_write_polydata,
)
from meshmetrics.utils import (
    compute_distance_field,
    index2world,
    mask_boundary_region,
    mesh_distance_bounds,
    mesh_distance_lower_bound,
    vtk_compute_cell_sizes,
    vtk_2D_mask_surface,
    vtk_mask_distance_field,
    vtk_points_outside_image,
)

CENTER = np.array([12.0, -7.0, 20.0])
AXES = np.array([9.0, 5.0, 3.0])


def make_image(size, spacing, direction, origin=None):
    direction = np.asarray(direction, dtype=float)
    size, spacing = np.asarray(size), np.asarray(spacing, dtype=float)
    if origin is None:  # centre the grid on CENTER
        origin = CENTER[: len(size)] - direction @ (spacing * (size - 1) / 2)
    img = sitk.Image([int(s) for s in size], sitk.sitkUInt8)
    img.SetSpacing(spacing.tolist())
    img.SetOrigin(np.asarray(origin, dtype=float).tolist())
    img.SetDirection(direction.ravel().tolist())
    return img


def voxel_centers_world(img):
    ndim = img.GetDimension()
    idx = np.stack(
        np.meshgrid(*[np.arange(n) for n in img.GetSize()], indexing="ij"), -1
    ).reshape(-1, ndim)
    direction = np.array(img.GetDirection()).reshape(ndim, ndim)
    return index2world(idx, np.array(img.GetSpacing()), np.array(img.GetOrigin()), direction)


def ellipsoid_mesh():
    src = vtk.vtkSphereSource()
    src.SetRadius(1)
    src.SetThetaResolution(128)
    src.SetPhiResolution(128)
    tr = vtk.vtkTransform()
    tr.Translate(*CENTER)
    tr.Scale(*AXES)
    f = vtk.vtkTransformFilter()
    f.SetInputConnection(src.GetOutputPort())
    f.SetTransform(tr)
    f.Update()
    return f.GetOutput()


def ellipse_contour(n=512):
    pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        pts.InsertNextPoint(CENTER[0] + AXES[0] * np.cos(t), CENTER[1] + AXES[1] * np.sin(t), 0)
    for i in range(n):
        lines.InsertNextCell(2, [i, (i + 1) % n])
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    return poly


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def rotation_2d(deg, flip=1):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([1, flip])


DIRECTIONS_3D = {
    "identity": np.eye(3),
    "flip_xy": np.diag([-1.0, -1.0, 1.0]),
    "perm_xy": np.eye(3)[[1, 0, 2]],
    "rot_z_90": R.from_euler("z", 90, degrees=True).as_matrix(),
    "rot_z_30": R.from_euler("z", 30, degrees=True).as_matrix(),
    "rot_xyz": R.from_euler("xyz", [20, 35, -50], degrees=True).as_matrix(),
    "rot_flip": R.from_euler("xyz", [10, -25, 40], degrees=True).as_matrix() @ np.diag([1, 1, -1]),
    **{f"random_{i}": m for i, m in enumerate(R.random(5, random_state=1).as_matrix())},
}


@pytest.mark.parametrize("direction", DIRECTIONS_3D.values(), ids=DIRECTIONS_3D.keys())
def test_voxelizer_3d_direction(direction):
    img = make_image((48, 56, 64), (0.7, 0.9, 1.1), direction)
    out = sitk.GetArrayFromImage(vtk_voxelizer(ellipsoid_mesh(), img)).astype(bool)
    pts = voxel_centers_world(img)
    gt = ((((pts - CENTER) / AXES) ** 2).sum(1) <= 1).reshape(img.GetSize()).T
    assert dice(out, gt) > 0.99


@pytest.mark.parametrize("deg", [0, 30, 90, 180, -65])
@pytest.mark.parametrize("flip", [1, -1])
def test_voxelizer_2d_direction(deg, flip):
    img = make_image((80, 80), (0.6, 0.8), rotation_2d(deg, flip))
    out = sitk.GetArrayFromImage(vtk_voxelizer(ellipse_contour(), img)).astype(bool)
    pts = voxel_centers_world(img)
    gt = ((((pts - CENTER[:2]) / AXES[:2]) ** 2).sum(1) <= 1).reshape(img.GetSize()).T
    assert dice(out, gt) > 0.99


def random_blob(shape, rng):
    a = np.zeros(shape, np.uint8)
    for _ in range(3):
        lo = rng.integers(3, np.array(shape) // 2)
        hi = lo + rng.integers(4, np.array(shape) // 2)
        a[tuple(slice(l, h) for l, h in zip(lo, hi))] = 1
    return a


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("ndim", [2, 3])
def test_meshing_voxelizer_roundtrip(ndim, seed):
    rng = np.random.default_rng(seed)
    if ndim == 2:
        direction, spacing = rotation_2d(rng.uniform(0, 360), rng.choice([1, -1])), [0.6, 0.8]
    else:
        direction, spacing = R.random(random_state=seed).as_matrix(), [0.7, 0.9, 1.3]
    a = random_blob((40, 52, 30)[:ndim], rng)
    img = sitk.GetImageFromArray(a)
    img.SetSpacing(spacing)
    img.SetOrigin(rng.uniform(-50, 50, ndim).tolist())
    img.SetDirection(np.asarray(direction).ravel().tolist())
    mesher = vtk_2D_meshing if ndim == 2 else vtk_3D_meshing
    out = sitk.GetArrayFromImage(vtk_voxelizer(mesher(img), img))
    assert (out.astype(bool) == a.astype(bool)).all()


def test_index2world_matches_simpleitk():
    direction = R.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    img = make_image((10, 11, 12), (0.8, 1.0, 1.2), direction, origin=(10.0, -5.0, 3.0))
    idx = np.array([[0, 0, 0], [3, 4, 5], [9, 10, 11]])
    expected = [img.TransformIndexToPhysicalPoint(i.tolist()) for i in idx]
    np.testing.assert_allclose(
        index2world(idx, np.array(img.GetSpacing()), np.array(img.GetOrigin()), direction),
        expected,
    )


def test_points_outside_image():
    direction = R.from_euler("z", 30, degrees=True).as_matrix()
    big = make_image((64, 64, 64), (1.0, 1.0, 1.0), direction)
    small = make_image((10, 10, 10), (1.0, 1.0, 1.0), direction)
    assert not vtk_points_outside_image(ellipsoid_mesh(), big)
    assert vtk_points_outside_image(ellipsoid_mesh(), small)


def test_np2sitk_roundtrip():
    a = np.random.default_rng(0).integers(0, 2, (5, 6, 7)).astype(np.uint8)
    img = np2sitk(a)
    assert img.GetSize() == a.shape
    assert img.GetSpacing() == (1.0, 1.0, 1.0)
    assert (sitk2np(img) == a).all()


@pytest.mark.parametrize("ext", [".obj", ".vtk", ".stl", ".vtp", ".OBJ"])
def test_read_write_polydata(tmp_path, ext):
    a = np.zeros((20, 20, 20), np.uint8)
    a[5:15, 6:14, 4:16] = 1
    mesh = vtk_meshing(np2sitk(a))
    pth = tmp_path / f"mesh{ext}"
    vtk_write_polydata(mesh, pth)
    back = vtk_read_polydata(pth)
    assert back.GetNumberOfCells() == mesh.GetNumberOfCells()
    np.testing.assert_allclose(back.GetBounds(), mesh.GetBounds(), atol=1e-5)


def test_read_polydata_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        vtk_read_polydata(tmp_path / "missing.vtk")
    with pytest.raises(ValueError, match="Unsupported"):
        vtk_read_polydata(tmp_path / "mesh.ply")
    # legacy .vtk that is not polydata
    grid = vtk.vtkImageData()
    grid.SetDimensions(2, 2, 2)
    w = vtk.vtkDataSetWriter()
    w.SetInputData(grid)
    w.SetFileName(str(tmp_path / "image.vtk"))
    w.Write()
    with pytest.raises(ValueError, match="not a legacy VTK polydata"):
        vtk_read_polydata(tmp_path / "image.vtk")


def test_meshio_mixed_cells_and_lines():
    meshio = pytest.importorskip("meshio")
    pts = np.random.default_rng(0).random((6, 3))
    m = meshio.Mesh(
        pts,
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("quad", np.array([[2, 3, 4, 5]])),
            ("line", np.array([[0, 1], [1, 2]])),
        ],
    )
    poly = meshio_to_vtk(m)
    assert poly.GetNumberOfPolys() == 2
    assert poly.GetNumberOfLines() == 2
    # vtkPolyData orders cells as verts, lines, polys
    assert poly.GetCell(3).GetNumberOfPoints() == 4


def random_mask_image(ndim, seed, touch_border=False):
    rng = np.random.default_rng(seed)
    shape = (24, 28, 20)[:ndim]
    a = random_blob(shape, rng)
    if touch_border:
        a[(slice(0, 6),) * ndim] = 1
    img = sitk.GetImageFromArray(a)
    img.SetSpacing(rng.uniform(0.5, 1.5, ndim).tolist())
    img.SetOrigin(rng.uniform(-20, 20, ndim).tolist())
    direction = rotation_2d(rng.uniform(0, 360)) if ndim == 2 else R.random(random_state=seed).as_matrix()
    img.SetDirection(np.asarray(direction).ravel().tolist())
    return img


def slotted_boxes_mesh():
    """Two boxes separated by a 0.1 wide slot that contains no voxel centre (1 mm grid).

    The voxelized mask is one solid block, but the surface reaches deep inside it, so
    distances to the surface cannot be bounded using the mask alone.
    """
    append = vtk.vtkAppendPolyData()
    for x0, x1 in [(2.2, 10.45), (10.55, 18.8)]:
        cube = vtk.vtkCubeSource()
        cube.SetBounds(x0, x1, 2.2, 12.8, 2.2, 12.8)
        append.AddInputConnection(cube.GetOutputPort())
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(append.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def surface_and_mask(kind, seed):
    if kind == "mask":
        img = random_mask_image(3, seed, touch_border=seed % 2 == 0)
        return vtk_meshing(img), img
    if kind == "mask_2d":
        img = random_mask_image(2, seed, touch_border=seed % 2 == 0)
        return None, img
    meta = sitk.Image([22, 16, 16], sitk.sitkUInt8)
    if kind == "slot":
        mesh = slotted_boxes_mesh()
    else:  # ellipsoid on a rotated, anisotropic grid
        mesh = ellipsoid_mesh()
        meta = make_image((40, 30, 24), (0.7, 0.9, 1.1), DIRECTIONS_3D["rot_xyz"])
    return mesh, vtk_voxelizer(mesh, meta)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["mask", "mask_2d", "slot", "ellipsoid"])
def test_mesh_distance_lower_bound(kind, seed):
    mesh, mask = surface_and_mask(kind, seed)
    exact = vtk_mask_distance_field(mask, mesh)
    fg = sitk2np(mask) > 0
    if mask.GetDimension() == 3:
        sources = [mesh]
    else:  # extruded surface (subdivided) and the contour itself (no subdivision)
        sources = [vtk_2D_mask_surface(mask), vtk_2D_meshing(mask)]
    for source in sources:
        for max_dist in [0.5, 1.0, 3.0]:
            bound = mesh_distance_lower_bound(source, mask, max_dist)
            near = fg & (exact < max_dist)
            assert np.all(bound[near] <= exact[near] + 1e-6)


def test_slot_mesh_needs_mesh_based_bound():
    """Voxels next to the slot are close to the surface although deep inside the mask."""
    mesh, mask = surface_and_mask("slot", 0)
    exact = vtk_mask_distance_field(mask, mesh)
    mask_np = sitk2np(mask)
    assert mask_np[10, 7, 7] and mask_np[11, 7, 7]  # solid across the slot
    assert exact[10, 7, 7] == pytest.approx(0.45)
    assert exact[11, 7, 7] == pytest.approx(0.45)


@pytest.mark.parametrize("kind", ["mask", "mask_2d", "slot", "ellipsoid"])
def test_mask_distance_field_max_dist(kind):
    mesh, mask = surface_and_mask(kind, seed=7)
    full = vtk_mask_distance_field(mask, mesh)
    for max_dist in [0.5, 1.0, 2.5]:
        band = vtk_mask_distance_field(mask, mesh, max_dist=max_dist)
        assert np.array_equal(full < max_dist, band < max_dist)
        near = full < max_dist
        np.testing.assert_allclose(band[near], full[near])


def test_compute_distance_field_matches_per_point():
    mesh = ellipsoid_mesh()
    pts = CENTER + np.random.default_rng(0).normal(scale=6, size=(200, 3))
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(mesh)
    expected = [abs(ipd.FunctionValue(p)) for p in pts]
    np.testing.assert_allclose(compute_distance_field(pts, mesh), expected)
    assert compute_distance_field(np.zeros((0, 3)), mesh).shape == (0,)


def test_cell_sizes_match_per_cell():
    mesh = vtk_meshing(random_mask_image(3, seed=1))
    expected = [mesh.GetCell(i).ComputeArea() for i in range(mesh.GetNumberOfCells())]
    np.testing.assert_allclose(vtk_compute_cell_sizes(mesh), expected)
    assert vtk_compute_cell_sizes(vtk.vtkPolyData()).shape == (0,)


@pytest.mark.parametrize(
    "spacing, mesh_fn, ndim",
    [((0.7, 0.9, 1.1), ellipsoid_mesh, 3), ((1.0, 1.0, 1.0), ellipsoid_mesh, 3), ((0.6, 0.8), ellipse_contour, 2)],
)
def test_voxelizer_with_spacing(spacing, mesh_fn, ndim):
    mesh = mesh_fn()
    img = vtk_voxelizer(mesh, spacing=spacing)
    assert img.GetDimension() == ndim
    np.testing.assert_allclose(img.GetSpacing(), spacing)
    np.testing.assert_allclose(img.GetDirection(), np.eye(ndim).ravel())
    # grid starts 2 pixels/voxels before the mesh bounds
    bounds_min = np.array(mesh.GetBounds())[0 : 2 * ndim : 2]
    np.testing.assert_allclose(img.GetOrigin(), bounds_min - 2 * np.array(spacing))

    out = sitk2np(img).astype(bool)
    # the mesh is fully covered: the outermost pixels/voxels are background on every side
    for axis in range(ndim):
        assert not out.take(0, axis=axis).any() and not out.take(-1, axis=axis).any()
    # matches the exact shape
    pts = voxel_centers_world(img)
    gt = ((((pts - CENTER[:ndim]) / AXES[:ndim]) ** 2).sum(1) <= 1).reshape(img.GetSize())
    assert dice(out, gt) > 0.99


def test_voxelizer_arguments():
    mesh = ellipsoid_mesh()
    meta = make_image((40, 30, 24), (1.0, 1.0, 1.0), np.eye(3))
    with pytest.raises(ValueError, match="exactly one"):
        vtk_voxelizer(mesh)
    with pytest.raises(ValueError, match="exactly one"):
        vtk_voxelizer(mesh, meta, spacing=(1.0, 1.0, 1.0))
    # positional reference image still works
    assert vtk_voxelizer(mesh, meta).GetSize() == meta.GetSize()
    # empty mesh with spacing gives an empty image
    assert np.prod(vtk_voxelizer(vtk.vtkPolyData(), spacing=(1.0, 1.0, 1.0)).GetSize()) == 0


def test_meshes_bbox_image():
    from meshmetrics.utils import vtk_meshes_bbox_sitk_image

    def sphere(center, radius):
        src = vtk.vtkSphereSource()
        src.SetCenter(*center)
        src.SetRadius(radius)
        src.Update()
        return src.GetOutput()

    meshes = [sphere((0, 0, 0), 2), sphere((10, -5, 3), 1), sphere((-4, 6, -8), 3)]
    spacing = (0.5, 1.0, 2.0)
    img = vtk_meshes_bbox_sitk_image(meshes, spacing, tolerance=(1.0, 1.0, 1.0))
    bounds = np.array([m.GetBounds() for m in meshes])
    lo, hi = bounds[:, 0::2].min(0), bounds[:, 1::2].max(0)
    np.testing.assert_allclose(img.GetOrigin(), lo - 1.0)
    last_centre = np.array(img.GetOrigin()) + (np.array(img.GetSize()) - 1) * spacing
    assert np.all(last_centre >= hi + 1.0 - np.array(spacing) - 1e-9)
    np.testing.assert_allclose(img.GetDirection(), np.eye(3).ravel())

    # a single mesh, and empty meshes are ignored
    single = vtk_meshes_bbox_sitk_image(meshes[0], spacing)
    with_empty = vtk_meshes_bbox_sitk_image([vtk.vtkPolyData(), meshes[0]], spacing)
    assert single.GetSize() == with_empty.GetSize()
    assert single.GetOrigin() == with_empty.GetOrigin()
    assert np.prod(vtk_meshes_bbox_sitk_image([vtk.vtkPolyData()] * 2, spacing).GetSize()) == 0


def _segment_distances_reference(contour, surface):
    """Per-segment loop (the original implementation) as reference."""
    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(surface)
    lines, ids = contour.GetLines(), vtk.vtkIdList()
    lines.InitTraversal()
    dists, lengths = [], []
    while lines.GetNextCell(ids):
        p0 = np.array(contour.GetPoint(ids.GetId(0)))
        p1 = np.array(contour.GetPoint(ids.GetId(1)))
        lengths.append(np.linalg.norm(p0 - p1))
        dists.append(abs(ipd.FunctionValue((p0 + p1) / 2)))
    return np.array(dists), np.array(lengths)


@pytest.mark.parametrize("seed", range(4))
def test_2D_segment_distances_match_reference(seed):
    from meshmetrics.utils import vtk_2D_centroid2surface_dist_length

    ref_img = random_mask_image(2, seed, touch_border=seed % 2 == 0)
    pred_img = random_mask_image(2, seed + 10, touch_border=seed % 2 == 1)
    pred_img.CopyInformation(ref_img)  # same grid (rotated, anisotropic)
    contour, surface = vtk_2D_meshing(ref_img), vtk_2D_mask_surface(pred_img)

    dists, lengths = vtk_2D_centroid2surface_dist_length(contour, surface)
    exp_dists, exp_lengths = _segment_distances_reference(contour, surface)
    assert len(dists) == contour.GetNumberOfLines() > 0
    np.testing.assert_allclose(dists, exp_dists, atol=1e-9)
    np.testing.assert_allclose(lengths, exp_lengths, atol=1e-9)

    # empty contour
    empty_d, empty_l = vtk_2D_centroid2surface_dist_length(vtk.vtkPolyData(), surface)
    assert empty_d.shape == empty_l.shape == (0,)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["mask", "mask_2d", "slot", "ellipsoid"])
def test_mesh_distance_upper_bound(kind, seed):
    mesh, mask = surface_and_mask(kind, seed)
    exact = vtk_mask_distance_field(mask, mesh)
    fg = sitk2np(mask) > 0
    source = mesh if mask.GetDimension() == 3 else vtk_2D_meshing(mask)
    for max_dist in [0.5, 1.0, 3.0]:
        lower, upper = mesh_distance_bounds(source, mask, max_dist)
        near = fg & (exact < max_dist)
        assert np.all(lower[near] <= exact[near] + 1e-6)
        assert np.all(upper[fg] >= exact[fg] - 1e-6)


@pytest.mark.parametrize("kind", ["mask", "mask_2d", "slot", "ellipsoid"])
def test_mask_boundary_region_matches_exact_field(kind):
    """The region from the bounds equals thresholding the exact distance field, also when the
    exact distances computed for one tau are reused for another."""
    mesh, mask = surface_and_mask(kind, seed=5)
    if mask.GetDimension() == 2:
        source = vtk_2D_meshing(mask)
        surface, full = vtk_2D_mask_surface(mask), vtk_mask_distance_field(mask)
    else:
        source = surface = mesh
        full = vtk_mask_distance_field(mask, mesh)
    fg = sitk2np(mask) > 0
    exact = None
    for tau in [2.5, 0.7, 1.3, 4.0]:
        region, exact = mask_boundary_region(mask, tau, surface, bound_source=source, exact=exact)
        assert np.array_equal(region, fg & (full < tau)), tau
