from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtkIdTypeArray, numpy_to_vtk


def np2sitk(img_np: np.ndarray, swapaxes=True) -> sitk.Image:
    """Converts a numpy array to a SimpleITK image (default geometry).

    With ``swapaxes=True`` the array axes map to the image x, y(, z) axes, i.e.
    the inverse of ``sitk2np``. Set spacing/origin/direction on the result.
    """
    if swapaxes:
        assert img_np.ndim in [2, 3], "Unsupported number of dimensions"
        img_np = np.swapaxes(img_np, 0, -1)
    return sitk.GetImageFromArray(img_np)


def sitk2np(sitk_img: sitk.Image) -> np.ndarray:
    assert sitk_img.GetDimension() in [2, 3], "Unsupported number of dimensions"
    return np.swapaxes(sitk.GetArrayFromImage(sitk_img), 0, -1)


def sitk_add_axis_to_end(sitk_img: sitk.Image) -> sitk.Image:
    """Adds a new axis as the last dimension to a SimpleITK image.

    Note:
        The new spacing element is set to 1.0, and 0.0 is appended to the origin.
    """
    return sitk.JoinSeries([sitk_img])


def to_sitk(img: Union[str, Path, sitk.Image]) -> sitk.Image:
    if isinstance(img, (str, Path)):
        img = sitk.ReadImage(str(img))
    elif isinstance(img, sitk.Image):
        pass
    else:
        raise NotImplementedError(f"Unknown image type: {type(img)}")
    return img


_POLYDATA_READERS = {
    ".obj": vtk.vtkOBJReader,
    ".vtk": vtk.vtkPolyDataReader,
    ".stl": vtk.vtkSTLReader,
    ".vtp": vtk.vtkXMLPolyDataReader,
}
_POLYDATA_WRITERS = {
    ".obj": vtk.vtkOBJWriter,
    ".vtk": vtk.vtkPolyDataWriter,
    ".stl": vtk.vtkSTLWriter,
    ".vtp": vtk.vtkXMLPolyDataWriter,
}


def _polydata_io_class(pth: Path, classes: dict):
    suffix = pth.suffix.lower()
    if suffix not in classes:
        raise ValueError(
            f"Unsupported mesh file extension '{suffix}', supported: {', '.join(classes)}"
        )
    return classes[suffix]


def vtk_read_polydata(pth: Union[str, Path]) -> vtk.vtkPolyData:
    """Read a mesh from a .obj, .vtk (legacy polydata), .stl or .vtp file."""
    pth = Path(pth)
    reader_cls = _polydata_io_class(pth, _POLYDATA_READERS)
    if not pth.is_file():
        raise FileNotFoundError(pth)

    reader = reader_cls()
    reader.SetFileName(str(pth))
    if isinstance(reader, vtk.vtkPolyDataReader) and not reader.IsFilePolyData():
        raise ValueError(f"{pth} is not a legacy VTK polydata file")
    reader.Update()
    return reader.GetOutput()


def vtk_write_polydata(vtk_polydata: vtk.vtkPolyData, pth: Union[str, Path]):
    """Write a mesh to a .obj, .vtk (legacy polydata), .stl or .vtp file."""
    assert isinstance(vtk_polydata, vtk.vtkPolyData), "Unknown mesh type"
    pth = Path(pth)
    writer = _polydata_io_class(pth, _POLYDATA_WRITERS)()
    writer.SetInputData(vtk_polydata)
    writer.SetFileName(str(pth))
    if not writer.Write():
        raise IOError(f"Failed to write mesh to {pth}")


def to_vtk(src_mesh: Union[str, Path, vtk.vtkPolyData]) -> vtk.vtkPolyData:
    if isinstance(src_mesh, (str, Path)):
        src_mesh = vtk_read_polydata(src_mesh)
    elif isinstance(src_mesh, vtk.vtkPolyData):
        pass
    else:
        raise NotImplementedError(f"Unknown mesh type: {type(src_mesh)}")
    return src_mesh


def vtk_is_mesh_closed(polydata):
    """
    Check if a surface is closed (i.e., it has no boundary edges).

    Parameters:
    polydata (vtk.vtkPolyData): The input surface mesh.

    Returns:
    bool: True if the surface is closed, False otherwise.
    """
    polydata = to_vtk(polydata)
    # Initialize the vtkFeatureEdges filter
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()

    # Get the number of boundary edges
    boundary_edges = feature_edges.GetOutput().GetNumberOfCells()

    return boundary_edges == 0


def vtk_is_mesh_manifold(polydata):
    """
    Check if a surface is manifold
    (i.e., every mesh edge is shared by at most two faces).

    Parameters:
    polydata (vtk.vtkPolyData): The input surface mesh.

    Returns:
    bool: True if the surface is closed, False otherwise.
    """

    polydata = to_vtk(polydata)
    # Initialize the vtkFeatureEdges filter
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(polydata)
    feature_edges.BoundaryEdgesOff()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOn()
    feature_edges.Update()

    # Get the number of boundary edges
    boundary_edges = feature_edges.GetOutput().GetNumberOfCells()

    return boundary_edges == 0


def _vtk_discrete_meshing(src_img: sitk.Image, pad: bool) -> vtk.vtkPolyData:
    """Mesh the foreground (> 0) of a 2D/3D mask with discrete flying edges / marching cubes.

    The mask is meshed in index space and the points are then mapped to world coordinates
    (spacing, origin and direction), independent of VTK's support for image direction.
    """
    n_dim = src_img.GetDimension()
    if not sitk.GetArrayViewFromImage(src_img).any():
        return vtk.vtkPolyData()

    # pad to avoid potential open boundary related issues
    if pad:
        src_img = sitk.ConstantPad(src_img, (1,) * n_dim, (1,) * n_dim, 0)

    mask_np = (sitk.GetArrayViewFromImage(src_img) > 0).astype(np.uint8)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(*src_img.GetSize(), *(1,) * (3 - n_dim))
    vtk_img.GetPointData().SetScalars(numpy_to_vtk(mask_np.ravel(), deep=True))

    if n_dim == 2:
        meshing_alg = vtk.vtkDiscreteFlyingEdges2D()
    else:
        meshing_alg = vtk.vtkDiscreteMarchingCubes()
        meshing_alg.ComputeNormalsOff()
        meshing_alg.ComputeGradientsOff()
    meshing_alg.SetValue(0, 1)
    meshing_alg.ComputeScalarsOff()
    meshing_alg.SetInputData(vtk_img)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()
    if mesh.GetNumberOfPoints() == 0:  # e.g. a single-slice 3D volume without padding
        return vtk.vtkPolyData()

    # index -> world coordinates
    pts_idx = vtk_to_numpy(mesh.GetPoints().GetData())[:, :n_dim]
    pts_world = index2world(
        pts_idx,
        np.array(src_img.GetSpacing()),
        np.array(src_img.GetOrigin()),
        np.array(src_img.GetDirection()).reshape(n_dim, n_dim),
    )
    if n_dim == 2:
        pts_world = np.concatenate([pts_world, np.zeros((len(pts_world), 1))], axis=1)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(pts_world, dtype=float), deep=True))
    mesh.SetPoints(points)
    return mesh


def vtk_2D_meshing(
    src_img: Union[str, Path, sitk.Image], pad: bool = True
) -> vtk.vtkPolyData:
    """Contour (line segments in the z=0 plane) of a 2D mask, using discrete flying edges."""
    src_img = to_sitk(src_img)
    assert src_img.GetDimension() == 2, "Only 2D images are supported for discrete flying edges"
    return _vtk_discrete_meshing(src_img, pad)


def vtk_3D_meshing(
    src_img: Union[str, Path, sitk.Image], pad: bool = True
) -> vtk.vtkPolyData:
    """Triangle surface mesh of a 3D mask, using discrete marching cubes."""
    src_img = to_sitk(src_img)
    assert src_img.GetDimension() == 3, "Only 3D images are supported for discrete marching cubes"
    return _vtk_discrete_meshing(src_img, pad)


def vtk_meshing(src_img: Union[str, Path, sitk.Image]):
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    if n_dim == 2:
        mesh = vtk_2D_meshing(src_img)
    elif n_dim == 3:
        mesh = vtk_3D_meshing(src_img)
    else:
        raise ValueError("Only 2D or 3D images are supported")
    return mesh


def vtk_2D_mask_surface(mask_sitk: sitk.Image) -> vtk.vtkPolyData:
    """Open 3D surface (vertical walls) of a 2D mask's contour (see `vtk_2D_meshing`).

    `vtk.vtkImplicitPolyDataDistance` needs a surface, so the 2D contour (in the z=0 plane)
    is extruded along z from -0.5 to 0.5. For points in the z=0 plane, distances to this
    surface equal the in-plane distances to the contour. The contour is created with padding,
    as in `vtk_meshing`, so masks touching the image border have a closed boundary (as in 3D).
    """
    return _vtk_extrude_contour(vtk_2D_meshing(mask_sitk, pad=True))


def _vtk_extrude_contour(contour: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Extrude a 2D contour (z=0) along z from -0.5 to 0.5 into a triangulated open surface."""
    if contour.GetNumberOfCells() == 0:
        return vtk.vtkPolyData()

    shift = vtk.vtkTransform()
    shift.Translate(0.0, 0.0, -0.5)
    shift_filter = vtk.vtkTransformFilter()
    shift_filter.SetInputData(contour)
    shift_filter.SetTransform(shift)

    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection(shift_filter.GetOutputPort())
    extrude.SetExtrusionTypeToVectorExtrusion()
    extrude.SetVector(0.0, 0.0, 1.0)
    extrude.SetScaleFactor(1.0)
    extrude.CappingOff()

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(extrude.GetOutputPort())
    triangulate.PassLinesOff()
    triangulate.PassVertsOff()
    triangulate.Update()
    return triangulate.GetOutput()


def vtk_2D_centroid2surface_dist_length(
    pts_contour: vtk.vtkPolyData,
    surface_mesh: vtk.vtkPolyData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from the centroid (midpoint) of each contour segment to `surface_mesh`, and
    the length of each segment."""
    lines = pts_contour.GetLines()
    if lines.GetNumberOfCells() == 0:
        return np.zeros(0), np.zeros(0)
    assert (
        lines.GetMaxCellSize() == 2
    ), "This function supports segment lines that have 2 points"

    pts = vtk_to_numpy(pts_contour.GetPoints().GetData()).astype(float)
    segments = vtk_to_numpy(lines.GetConnectivityArray()).reshape(-1, 2)
    pt0, pt1 = pts[segments[:, 0]], pts[segments[:, 1]]

    segment_lengths = np.linalg.norm(pt1 - pt0, axis=1)
    dists_pts2surface = compute_distance_field((pt0 + pt1) / 2, surface_mesh)
    return dists_pts2surface, segment_lengths


def sort_dists_and_bsizes(dists, boundary_sizes) -> Tuple[np.ndarray, np.ndarray]:
    reidx = np.argsort(np.abs(dists))
    dists_s = dists[reidx]
    boundary_sizes_s = boundary_sizes[reidx]
    return dists_s, boundary_sizes_s


def vtk_measurements_2D(
    ref_contour: vtk.vtkPolyData,
    pred_contour: vtk.vtkPolyData,
    ref_sitk: sitk.Image,
    pred_sitk: sitk.Image,
    ref_surface: vtk.vtkPolyData = None,
    pred_surface: vtk.vtkPolyData = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bidirectional distances between contour centroids and the opposing surface.

    This function measures distances from the centroids of one contour (reference or
    prediction) to the surface generated from the opposing contour. Distances are computed
    in both directions (ref→pred and pred→ref), along with the lengths of the corresponding
    line segments. Distances and the corresponding segment lengths are then sorted.

    Note:
        Since `vtk.vtkImplicitPolyDataDistance` requires a surface, the contours of the
        2D masks are extruded along z into open surfaces (see `vtk_2D_mask_surface`).

    Args:
        ref_contour (vtk.vtkPolyData): Contour created from reference segmentation.
        pred_contour (vtk.vtkPolyData): Contour created from predicted segmentation.
        ref_sitk (SimpleITK.Image): Reference segmentation mask.
        pred_sitk (SimpleITK.Image): Predicted segmentation mask.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: numpy vector with distances
        from ref to pred mesh and ref segment lengths, and vice-versa
    """
    # fmt: off
    # extrude 2D contours of the masks into open surfaces (unless given)
    if ref_surface is None:
        ref_surface = vtk_2D_mask_surface(ref_sitk)
    if pred_surface is None:
        pred_surface = vtk_2D_mask_surface(pred_sitk)

    # compute distances between contour centroids and opposing surface
    dists_ref2pred, segment_lengths_ref = vtk_2D_centroid2surface_dist_length(ref_contour, pred_surface)
    dists_pred2ref, segment_lengths_pred = vtk_2D_centroid2surface_dist_length(pred_contour, ref_surface)

    # sort distances and boundary sizes
    dists_ref2pred, segment_lengths_ref = sort_dists_and_bsizes(dists_ref2pred, segment_lengths_ref)
    dists_pred2ref, segment_lengths_pred = sort_dists_and_bsizes(dists_pred2ref, segment_lengths_pred)
    # fmt: on

    return dists_ref2pred, segment_lengths_ref, dists_pred2ref, segment_lengths_pred


def vtk_compute_cell_sizes(mesh: vtk.vtkPolyData) -> np.ndarray:
    """Area of each (polygonal) cell of the mesh."""
    if mesh.GetNumberOfCells() == 0:
        return np.zeros(0)
    cell_size = vtk.vtkCellSizeFilter()
    cell_size.SetInputData(mesh)
    cell_size.ComputeAreaOn()
    cell_size.ComputeLengthOff()
    cell_size.ComputeVolumeOff()
    cell_size.ComputeVertexCountOff()
    cell_size.ComputeSumOff()
    cell_size.Update()
    return vtk_to_numpy(cell_size.GetOutput().GetCellData().GetArray("Area")).astype(float)


def vtk_measurements_3D(
    ref_mesh: vtk.vtkPolyData,
    pred_mesh: vtk.vtkPolyData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bidirectional distances between triangle centroids and the opposing surface.

    This function measures distances from the centroids of one mesh (reference or
    prediction) to the opposing mesh. Distances are computed in both directions
    (ref→pred and pred→ref), along with the areas of the corresponding
    surface elements (surfels). Distances and the corresponding surfel areas are then sorted.

    Args:
        ref_mesh (vtk.vtkPolyData): Mesh created from reference segmentation.
        pred_mesh (vtk.vtkPolyData): Mesh created from predicted segmentation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: numpy vector with distances
        from ref to pred mesh and ref surfel areas, and vice-versa
    """
    # fmt: off
    # compute distances between triangle centroids and opposing surface
    vtk_p2v_dist = vtk.vtkDistancePolyDataFilter()
    vtk_p2v_dist.SetInputData(0, ref_mesh)
    vtk_p2v_dist.SetInputData(1, pred_mesh)
    vtk_p2v_dist.SignedDistanceOn()
    vtk_p2v_dist.ComputeCellCenterDistanceOn()
    vtk_p2v_dist.ComputeSecondDistanceOn()
    vtk_p2v_dist.Update()
    dists_ref2pred = vtk_to_numpy(vtk_p2v_dist.GetOutput().GetCellData().GetArray("Distance"))
    dists_pred2ref = vtk_to_numpy(vtk_p2v_dist.GetSecondDistanceOutput().GetCellData().GetArray("Distance"))

    # compute surfel areas
    surfel_areas_ref = vtk_compute_cell_sizes(ref_mesh)
    surfel_areas_pred = vtk_compute_cell_sizes(pred_mesh)

    # sort distances and surfel areas
    dists_ref2pred, surfel_areas_ref = sort_dists_and_bsizes(dists_ref2pred, surfel_areas_ref)
    dists_pred2ref, surfel_areas_pred = sort_dists_and_bsizes(dists_pred2ref, surfel_areas_pred)
    # fmt: on

    return dists_ref2pred, surfel_areas_ref, dists_pred2ref, surfel_areas_pred


def index2world(
    inds: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    return origin + (inds * spacing) @ direction.T


def compute_distance_field(pts_np: np.ndarray, mesh_vtk: vtk.vtkPolyData) -> np.ndarray:
    """Unsigned distance from each point (N x 3) to the mesh surface."""
    if len(pts_np) == 0:
        return np.zeros(0)
    vtk_p2s_dist = vtk.vtkImplicitPolyDataDistance()
    vtk_p2s_dist.SetInput(mesh_vtk)
    dists = vtk.vtkDoubleArray()
    vtk_p2s_dist.FunctionValue(
        numpy_to_vtk(np.ascontiguousarray(pts_np, dtype=float), deep=True), dists
    )
    return np.abs(vtk_to_numpy(dists))


def mesh_distance_lower_bound(
    surface: vtk.vtkPolyData, meta_sitk: sitk.Image, max_dist: float
) -> np.ndarray:
    """Lower bound on the distance from each voxel centre of `meta_sitk` to `surface`.

    Valid for any surface, or for a 2D contour (line segments in the z=0 plane). Every
    surface point is within ``L`` of a mesh vertex: for a contour, ``L`` is half the longest
    segment; for a surface, the surface is subdivided until all triangle edges are at most
    ``L`` (the smallest voxel spacing). Each vertex is snapped to its nearest voxel centre
    (moving it by at most half a voxel diagonal), and a Euclidean distance transform gives the
    distance from every voxel centre to the nearest snapped vertex. Hence::

        exact distance >= EDT - voxel_diagonal / 2 - L

    Surface points farther than ``max_dist`` from the image grid are ignored, so the bound
    is only meaningful for deciding whether a distance is below ``max_dist``. For 2D images,
    the bound refers to in-plane (x, y) distances. Returned in ``sitk2np`` axis order.
    """
    ndim = meta_sitk.GetDimension()
    spacing = np.array(meta_sitk.GetSpacing())
    origin = np.array(meta_sitk.GetOrigin())
    direction = np.array(meta_sitk.GetDirection()).reshape(ndim, ndim)
    size = np.array(meta_sitk.GetSize())
    if surface.GetNumberOfCells() == 0:
        return np.full(size, np.inf)

    if surface.GetNumberOfPolys() == 0 and surface.GetNumberOfLines() > 0:
        # contour: every point of a segment is within half its length of an endpoint
        pts = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
        segments = vtk_to_numpy(surface.GetLines().GetConnectivityArray()).reshape(-1, 2)
        sample_error = np.linalg.norm(pts[segments[:, 1]] - pts[segments[:, 0]], axis=1).max() / 2
        pts = pts[:, :ndim]
    else:
        sample_error = spacing.min()
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(surface)
        triangulate.PassLinesOff()
        triangulate.PassVertsOff()
        subdivide = vtk.vtkAdaptiveSubdivisionFilter()
        subdivide.SetInputConnection(triangulate.GetOutputPort())
        subdivide.SetMaximumEdgeLength(sample_error)
        subdivide.SetMaximumTriangleArea(np.inf)
        subdivide.SetMaximumNumberOfPasses(1000)
        subdivide.Update()
        pts = vtk_to_numpy(subdivide.GetOutput().GetPoints().GetData())[:, :ndim]
    margin = np.linalg.norm(spacing) / 2 + sample_error

    # world -> nearest voxel index on a grid padded by `pad` voxels; vertices outside the
    # padded grid are farther than max_dist from every voxel centre of the image
    pad = int(np.ceil(max_dist / spacing.min())) + 1
    idx = np.rint(((pts - origin) @ direction) / spacing).astype(np.int64) + pad
    padded_size = size + 2 * pad
    idx = idx[np.all((idx >= 0) & (idx < padded_size), axis=1)]
    if len(idx) == 0:
        return np.full(size, np.inf)

    marked = np.zeros(padded_size[::-1], np.uint8)  # SimpleITK array order
    marked[tuple(idx[:, ::-1].T)] = 1
    marked_sitk = sitk.GetImageFromArray(marked)
    marked_sitk.SetSpacing(spacing.tolist())
    edt = sitk.SignedMaurerDistanceMap(
        marked_sitk, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    edt_np = np.maximum(sitk2np(edt), 0)[(slice(pad, -pad),) * ndim]
    return edt_np - margin


def vtk_mask_distance_field(
    mask_sitk: sitk.Image,
    mesh_vtk: vtk.vtkPolyData = None,
    max_dist: float = None,
    contour_2d: vtk.vtkPolyData = None,
    surface_2d: vtk.vtkPolyData = None,
) -> np.ndarray:
    """Distance from each foreground voxel centre of a binary mask to its surface.

    Args:
        mask_sitk: Binary mask.
        mesh_vtk: Surface of the mask (3D only). If None, or for 2D masks, the surface is
            created from the mask with `vtk_meshing`.
        max_dist: If given, exact distances are only computed for voxels that may be closer
            than ``max_dist`` to the surface (see ``mesh_distance_lower_bound``); the
            remaining foreground voxels are set to ``np.inf``.
        contour_2d, surface_2d: For 2D masks, the precomputed contour of the mask
            (``vtk_2D_meshing(mask_sitk)``) and its extrusion (``vtk_2D_mask_surface``), to
            avoid recomputing them.

    Returns:
        np.ndarray (``sitk2np`` axis order): distances for foreground voxels, 0 for background.
    """
    ndim = mask_sitk.GetDimension()
    mask_np = sitk2np(mask_sitk) > 0

    if ndim == 2:
        # distances to the contour, measured in the z=0 plane of an extruded open surface;
        # the lower bound uses the contour directly (no subdivision of the extrusion)
        bound_source = vtk_2D_meshing(mask_sitk, pad=True) if contour_2d is None else contour_2d
        surface = _vtk_extrude_contour(bound_source) if surface_2d is None else surface_2d
    else:
        surface = vtk_meshing(mask_sitk) if mesh_vtk is None else mesh_vtk
        bound_source = surface

    candidates = mask_np.copy()
    if max_dist is not None and np.isfinite(max_dist):
        candidates &= mesh_distance_lower_bound(bound_source, mask_sitk, max_dist) < max_dist

    pts = index2world(
        np.argwhere(candidates),
        np.array(mask_sitk.GetSpacing()),
        np.array(mask_sitk.GetOrigin()),
        np.array(mask_sitk.GetDirection()).reshape(ndim, ndim),
    )
    if ndim == 2:
        pts = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)

    dist_field = np.where(mask_np, np.inf, 0).astype(np.float32)
    dist_field[candidates] = compute_distance_field(pts, surface)
    return dist_field


def vtk_distance_field(
    ref_mesh: vtk.vtkPolyData,
    pred_mesh: vtk.vtkPolyData,
    ref_sitk: sitk.Image,
    pred_sitk: sitk.Image,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute voxel-wise distance fields between binary segmentation masks and their corresponding surfaces.

    For each foreground voxel in the reference and prediction masks, the function computes the
    shortest distance (in world coordinates) to the corresponding surface mesh. The output is
    two distance fields aligned with the input images, where distance values are only assigned
    inside the foreground region.

    Notes:
        - Works for both 2D and 3D segmentations. In 2D, an artificial third axis (z=0) is added,
          and the segmentation is extruded into 3D for surface meshing.
        - Distances are computed in physical space using image spacing, origin, and direction.

    Args:
        ref_mesh (vtk.vtkPolyData):
            Surface mesh corresponding to the reference segmentation (used for 3D inputs).
        pred_mesh (vtk.vtkPolyData):
            Surface mesh corresponding to the predicted segmentation (used for 3D inputs).
        ref_sitk (sitk.Image):
            Reference segmentation as a SimpleITK image (binary mask).
        pred_sitk (sitk.Image):
            Predicted segmentation as a SimpleITK image (binary mask).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - ref_dist_field (np.ndarray): Distance values for each foreground voxel in the
              reference segmentation relative to the reference surface.
            - pred_dist_field (np.ndarray): Distance values for each foreground voxel in the
              predicted segmentation relative to the predicted surface.
    """

    # 2D: surfaces are always created from the masks (meshes are unused)
    ref_dist_field = vtk_mask_distance_field(ref_sitk, ref_mesh if ref_sitk.GetDimension() == 3 else None)
    pred_dist_field = vtk_mask_distance_field(pred_sitk, pred_mesh if pred_sitk.GetDimension() == 3 else None)
    return ref_dist_field, pred_dist_field


def _vtk_image_values_to_sitk(vtk_img: vtk.vtkImageData) -> sitk.Image:
    """Copy the voxel values of a (3D) vtkImageData into a SimpleITK image (geometry not set).

    VTK stores values with x varying fastest, which matches a SimpleITK array of shape
    (nz, ny, nx).
    """
    nx, ny, nz = vtk_img.GetDimensions()
    values = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    return sitk.GetImageFromArray(values.reshape(nz, ny, nx))


def vtk_voxelizer(
    mesh_vtk: vtk.vtkPolyData,
    meta_sitk: sitk.Image = None,
    spacing: Union[tuple, list, np.ndarray] = None,
) -> sitk.Image:
    """Voxelize a closed mesh (3D) or closed contour (2D, in the z=0 plane).

    Pixels/voxels whose centre lies inside the mesh are set to 1. The output grid is given by
    exactly one of `meta_sitk` or `spacing`.

    Parameters
    ----------
    mesh_vtk : vtk.vtkPolyData
        Input mesh, in world coordinates.
    meta_sitk : sitk.Image, optional
        Reference image whose grid (size, spacing, origin and direction) is used.
    spacing : tuple, optional
        Pixel/voxel size (2 or 3 values) of a new axis-aligned grid (identity direction) that
        covers the mesh bounds plus a padding of 2 pixels/voxels on each side.

    Returns
    -------
    sitk.Image
        Binary mask on the reference grid, or on the new grid when `spacing` is given.
    """
    assert isinstance(mesh_vtk, vtk.vtkPolyData), "Mesh must be vtkPolyData"
    if (meta_sitk is None) == (spacing is None):
        raise ValueError("Provide exactly one of `meta_sitk` or `spacing`")
    if spacing is not None:
        spacing = np.asarray(spacing, dtype=float)
        assert len(spacing) in (2, 3), "spacing must have 2 or 3 values"
        assert np.all(spacing > 0), "spacing must be positive"
        meta_sitk = vtk_meshes_bbox_sitk_image(mesh_vtk, spacing=tuple(spacing), tolerance=2 * spacing)
    assert isinstance(meta_sitk, sitk.Image), "meta_sitk must be a SimpleITK image"

    # check for empty image (e.g. an empty mesh with `spacing`)
    if np.prod(meta_sitk.GetSize()) == 0:
        return meta_sitk

    ndim = meta_sitk.GetDimension()
    direction = np.array(meta_sitk.GetDirection()).reshape(ndim, ndim)
    origin = np.array(meta_sitk.GetOrigin())
    spacing = np.array(meta_sitk.GetSpacing())
    size = np.array(meta_sitk.GetSize())
    
    # Check if we need to handle non-standard direction
    has_non_standard_direction = not np.allclose(direction, np.eye(ndim))
    
    if has_non_standard_direction:
        # Create the full 4x4 transformation matrix
        # This transforms from world space to voxel space aligned with axes
        T = np.eye(4)
        T[:ndim, :ndim] = direction  # world = origin + direction @ local
        T[:ndim, 3] = origin
        
        # Create VTK transform (inverse to transform mesh into aligned space)
        vtk_transform = vtk.vtkTransform()
        vtk_matrix = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, T[i, j])
        vtk_transform.SetMatrix(vtk_matrix)
        vtk_transform.Inverse()
        
        # Transform the mesh
        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(mesh_vtk)
        transform_filter.SetTransform(vtk_transform)
        transform_filter.Update()
        mesh_vtk = transform_filter.GetOutput()
        
        # Use axis-aligned origin and extent for voxelization
        voxel_origin = np.zeros(3)
        voxel_spacing = spacing
        voxel_extent = [0, size[0] - 1, 0, size[1] - 1, 0, (size[2] - 1 if ndim == 3 else 0)]
    else:
        # Standard direction - use original parameters
        voxel_origin = origin
        voxel_spacing = spacing
        extent_max = size - 1
        if ndim == 2:
            extent_max = np.append(extent_max, 0)
        voxel_extent = [0, extent_max[0], 0, extent_max[1], 0, extent_max[2]]

    # VTK expects 3D origin/spacing; pad for 2D images
    if ndim == 2:
        voxel_origin = np.append(voxel_origin[:2], 0.0)
        voxel_spacing = np.append(voxel_spacing, 1.0)

    # Voxelize: polygonal data --> image stencil
    poly2stenc = vtk.vtkPolyDataToImageStencil()
    poly2stenc.SetInputData(mesh_vtk)
    poly2stenc.SetOutputOrigin(list(map(float, voxel_origin)))
    poly2stenc.SetOutputSpacing(list(map(float, voxel_spacing)))
    poly2stenc.SetOutputWholeExtent(voxel_extent)
    poly2stenc.Update()
    
    # Convert stencil to image
    stencil_to_image = vtk.vtkImageStencilToImage()
    stencil_to_image.SetInputData(poly2stenc.GetOutput())
    stencil_to_image.SetOutsideValue(0)
    stencil_to_image.SetInsideValue(1)
    stencil_to_image.Update()
    
    # Convert VTK image to SimpleITK
    voxelized_sitk = _vtk_image_values_to_sitk(stencil_to_image.GetOutput())
    
    # For 2D cases, remove the added axial dimension
    if ndim == 2:
        voxelized_sitk = voxelized_sitk[:, :, 0]
    
    # Set the correct metadata
    voxelized_sitk.SetOrigin(meta_sitk.GetOrigin())
    voxelized_sitk.SetSpacing(meta_sitk.GetSpacing())
    voxelized_sitk.SetDirection(meta_sitk.GetDirection())
    
    return voxelized_sitk

def vtk_points_outside_image(mesh_vtk: vtk.vtkPolyData, meta_sitk: sitk.Image) -> bool:
    """Check whether any mesh point lies outside the image grid (incl. half-voxel border)."""
    if mesh_vtk.GetNumberOfPoints() == 0:
        return False
    ndim = meta_sitk.GetDimension()
    pts = vtk_to_numpy(mesh_vtk.GetPoints().GetData())[:, :ndim]
    direction = np.array(meta_sitk.GetDirection()).reshape(ndim, ndim)
    origin, spacing = np.array(meta_sitk.GetOrigin()), np.array(meta_sitk.GetSpacing())
    # world -> continuous index (direction is orthonormal)
    idx = ((pts - origin) @ direction) / spacing
    size = np.array(meta_sitk.GetSize())
    return bool(np.any(idx < -0.5) or np.any(idx > size - 0.5))


def sitk_foreground_index_bounds(img: sitk.Image) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """Index bounds (lo, hi; inclusive, in (x, y[, z]) order) of the foreground (> 0), or None
    if it is empty."""
    fg = sitk.GetArrayViewFromImage(img) > 0  # (z, y, x) order
    if not fg.any():
        return None
    lo, hi = [], []
    for axis in range(fg.ndim):
        nonzero = np.flatnonzero(fg.any(axis=tuple(i for i in range(fg.ndim) if i != axis)))
        lo.append(nonzero[0])
        hi.append(nonzero[-1])
    return np.array(lo[::-1]), np.array(hi[::-1])


def vtk_mesh_index_bounds(
    mesh_vtk: vtk.vtkPolyData, meta_sitk: sitk.Image
) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """Index bounds (lo, hi; inclusive) on the grid of `meta_sitk` that enclose all mesh
    points, or None for an empty mesh. Bounds may lie outside the image."""
    if mesh_vtk.GetNumberOfPoints() == 0:
        return None
    ndim = meta_sitk.GetDimension()
    pts = vtk_to_numpy(mesh_vtk.GetPoints().GetData())[:, :ndim]
    direction = np.array(meta_sitk.GetDirection()).reshape(ndim, ndim)
    origin, spacing = np.array(meta_sitk.GetOrigin()), np.array(meta_sitk.GetSpacing())
    idx = ((pts - origin) @ direction) / spacing  # world -> continuous index
    return np.floor(idx.min(axis=0)).astype(int), np.ceil(idx.max(axis=0)).astype(int)


def crop_to_foreground(
    images: Sequence[sitk.Image],
    margin: int = 1,
    extra_bounds: Sequence[Tuple[np.ndarray, np.ndarray]] = (),
) -> list:
    """Crop images on the same grid to the union of their foreground (and `extra_bounds`),
    plus `margin` pixels/voxels, clipped to the image. The physical position is preserved.

    Images are returned unchanged if everything is empty or the crop would cover the image.
    """
    bounds = [b for b in (sitk_foreground_index_bounds(img) for img in images) if b is not None]
    bounds += [b for b in extra_bounds if b is not None]
    if not bounds:
        return list(images)
    size = np.array(images[0].GetSize())
    lo = np.maximum(np.min([b[0] for b in bounds], axis=0) - margin, 0)
    hi = np.minimum(np.max([b[1] for b in bounds], axis=0) + margin, size - 1)
    if np.all(lo == 0) and np.all(hi == size - 1):
        return list(images)
    roi_size = (hi - lo + 1).tolist()
    return [sitk.RegionOfInterest(img, roi_size, lo.tolist()) for img in images]


def crop_np_to_foreground(arrays: Sequence[np.ndarray], margin: int = 1) -> list:
    """Crop numpy masks of the same shape to the union of their foreground plus `margin`."""
    union = np.zeros(arrays[0].shape, bool)
    for a in arrays:
        union |= a.astype(bool)
    if not union.any():
        return list(arrays)
    slices = []
    for axis in range(union.ndim):
        nonzero = np.flatnonzero(union.any(axis=tuple(i for i in range(union.ndim) if i != axis)))
        slices.append(slice(max(nonzero[0] - margin, 0), min(nonzero[-1] + margin + 1, union.shape[axis])))
    return [a[tuple(slices)] for a in arrays]


def vtk_meshes_bbox_sitk_image(
    meshes: Union[vtk.vtkPolyData, Sequence[vtk.vtkPolyData]],
    spacing: tuple,
    tolerance: tuple | None = None,
) -> sitk.Image:
    """Empty axis-aligned image (identity direction) covering the bounds of one or more meshes.

    The first pixel/voxel centre lies at the minimum of the combined bounds minus `tolerance`,
    and the grid extends to at least the maximum of the bounds plus `tolerance` minus one
    pixel/voxel. Empty meshes are ignored; if all meshes are empty, an image of size 0 is
    returned. For 2D (two `spacing` values), only the x and y bounds are used.
    """
    if isinstance(meshes, vtk.vtkPolyData):
        meshes = [meshes]
    meshes = list(meshes)
    assert len(meshes) > 0, "At least one mesh is required"
    assert all(isinstance(m, vtk.vtkPolyData) for m in meshes), "Meshes must be vtkPolyData"
    ndim = len(spacing)

    meshes = [m for m in meshes if m.GetNumberOfPoints() > 0]
    if not meshes:
        meta_sitk = sitk.GetImageFromArray(np.zeros((0,) * ndim))
        meta_sitk.SetSpacing(spacing)
        return meta_sitk

    bounds = np.array([m.GetBounds() for m in meshes])  # (xmin, xmax, ymin, ymax, zmin, zmax)
    origin = bounds[:, 0::2].min(axis=0)[:ndim]
    diagonal = bounds[:, 1::2].max(axis=0)[:ndim]

    if tolerance is not None:
        tolerance = np.array(tolerance)
        assert np.all(tolerance >= 0), "Tolerance must be positive"
        origin -= tolerance
        diagonal += tolerance

    sitk_size = np.ceil((diagonal - origin) / np.array(spacing)).astype(int)

    meta_sitk = np2sitk(np.zeros(sitk_size, dtype=np.uint8))
    meta_sitk.SetSpacing(spacing)
    meta_sitk.SetOrigin(origin)

    return meta_sitk


def vtk_create_sphere(radius: float) -> vtk.vtkPolyData:
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(radius)
    sphere_source.SetThetaResolution(64)
    sphere_source.SetPhiResolution(64)
    sphere_source.Update()
    return sphere_source.GetOutput()


def extract_cirle_from_sphere(vtk_sphere):
    # Define the cutting plane (z = 0)
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)  # Origin of the plane
    plane.SetNormal(0.0, 0.0, 1.0)  # Normal vector (perpendicular to the Z-axis)

    # Use vtkCutter to slice the sphere with the plane
    cutter = vtk.vtkCutter()
    cutter.SetInputData(vtk_sphere)
    cutter.SetCutFunction(plane)  # Set the cutting plane
    cutter.Update()
    return cutter.GetOutput()


def create_synthetic_examples_3d(
    r1: float, r2: float, spacing: tuple
) -> vtk.vtkPolyData:
    # 3D
    vtk_mesh1 = vtk_create_sphere(r1)
    vtk_mesh2 = vtk_create_sphere(r2)

    # create a meta image SimpleITK that encompasses both masks
    meta_sitk = vtk_meshes_bbox_sitk_image(
        [vtk_mesh1, vtk_mesh2], spacing, tolerance=5 * np.array(spacing)
    )

    sitk_mask1 = vtk_voxelizer(vtk_mesh1, meta_sitk)
    sitk_mask2 = vtk_voxelizer(vtk_mesh2, meta_sitk)

    return vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2


def create_synthetic_examples_2d(
    r1: float, r2: float, spacing: tuple
) -> vtk.vtkPolyData:
    # 3D
    vtk_mesh1_3d = vtk_create_sphere(r1)
    vtk_mesh2_3d = vtk_create_sphere(r2)

    # 2D
    spacing = spacing[:2]
    vtk_mesh1 = extract_cirle_from_sphere(vtk_mesh1_3d)
    vtk_mesh2 = extract_cirle_from_sphere(vtk_mesh2_3d)

    # create a meta image SimpleITK that encompasses both masks
    meta_sitk = vtk_meshes_bbox_sitk_image(
        [vtk_mesh1, vtk_mesh2], spacing, tolerance=5 * np.array(spacing)
    )
    sitk_mask1 = vtk_voxelizer(vtk_mesh1, meta_sitk)
    sitk_mask2 = vtk_voxelizer(vtk_mesh2, meta_sitk)

    return vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2


# ------------------------------
def trimesh_to_vtk(mesh: "trimesh.Trimesh") -> vtk.vtkPolyData:
    """
    Convert a trimesh.Trimesh object (2D polygon or 3D mesh) to vtk.vtkPolyData.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh. Can be 2D (Nx2) or 3D (Nx3).

    Returns
    -------
    vtk.vtkPolyData
        VTK PolyData object representing the mesh.
    """
    # Ensure vertices are Nx3
    vertices = np.array(mesh.vertices)
    if vertices.shape[1] == 2:
        vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1))])

    # Convert vertices to vtkPoints
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Handle faces if present
    if mesh.faces is not None and len(mesh.faces) > 0:
        faces = np.array(mesh.faces, dtype=np.int64)
        faces_flat = np.hstack(
            [np.full((faces.shape[0], 1), faces.shape[1], dtype=np.int64), faces]
        ).ravel()
        vtk_faces = numpy_to_vtkIdTypeArray(faces_flat, deep=True)
        cells = vtk.vtkCellArray()
        cells.ImportLegacyFormat(vtk_faces)
        polydata.SetPolys(cells)
    else:
        # If no faces, treat as a set of points (use vertices only)
        polydata.SetVerts(vtk.vtkCellArray())

    polydata.BuildCells()
    polydata.BuildLinks()
    return polydata


def meshio_to_vtk(mesh: "meshio.Mesh") -> vtk.vtkPolyData:
    """
    Convert a meshio.Mesh object to vtk.vtkPolyData.
    Works with 2D (polygonal) and 3D meshes.

    Parameters
    ----------
    mesh : meshio.Mesh
        The input mesh.

    Returns
    -------
    vtk.vtkPolyData
        VTK PolyData object.
    """

    # Get points
    points = np.array(mesh.points)
    if points.shape[1] == 2:
        points = np.hstack([points, np.zeros((points.shape[0], 1))])  # pad z for 2D

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Collect polygon (3D surfaces) and line (2D contours) cells.
    # mesh.cells is a list of cell blocks; blocks may have different cell sizes
    # (e.g. triangles and quads), so each block is flattened separately.
    def to_vtk_cells(blocks):
        flat = [
            np.hstack(
                [np.full((len(b), 1), b.shape[1], dtype=np.int64), b.astype(np.int64)]
            ).ravel()
            for b in blocks
        ]
        cells = vtk.vtkCellArray()
        cells.ImportLegacyFormat(numpy_to_vtkIdTypeArray(np.concatenate(flat), deep=True))
        return cells

    polys = [b.data for b in mesh.cells if b.type in ("triangle", "quad", "polygon")]
    lines = [b.data for b in mesh.cells if b.type == "line"]
    if polys:
        polydata.SetPolys(to_vtk_cells(polys))
    if lines:
        polydata.SetLines(to_vtk_cells(lines))
    if not polys and not lines:
        # fallback: only vertices
        polydata.SetVerts(vtk.vtkCellArray())

    polydata.BuildCells()
    polydata.BuildLinks()
    return polydata
