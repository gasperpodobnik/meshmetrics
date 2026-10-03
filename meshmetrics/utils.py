from pathlib import Path
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtkIdTypeArray, numpy_to_vtk
from SimpleITK.utilities.vtk import sitk2vtk, vtk2sitk


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


def vtk_read_polydata(pth: Union[str, Path]) -> vtk.vtkPolyData:
    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(pth))
    reader.Update()
    polydata = reader.GetOutput()
    return polydata


def vtk_write_polydata(vtk_polydata: vtk.vtkPolyData, dst_pth: Union[str, Path]):
    writer = vtk.vtkOBJWriter()
    writer.SetInputData(vtk_polydata)
    writer.SetFileName(str(dst_pth))
    writer.Write()


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


def vtk_2D_meshing(
    src_img: Union[str, Path, sitk.Image], pad: bool = True
) -> vtk.vtkPolyData:
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    assert n_dim == 2, "Only 2D images are supported for marching squares"

    if sitk.GetArrayFromImage(src_img).sum() == 0:
        return vtk.vtkPolyData()

    # pad to avoid potential open boundary related issues
    if pad:
        src_img = sitk.ConstantPad(src_img, (1, 1), (1, 1), 0)

    vtkImage = sitk2vtk(src_img > 0)
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    meshing_alg = vtk.vtkSurfaceNets2D()
    meshing_alg.SmoothingOff()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

    return mesh


def vtk_3D_meshing(
    src_img: Union[str, Path, sitk.Image], pad: bool = True
) -> vtk.vtkPolyData:
    src_img = to_sitk(src_img)

    n_dim = src_img.GetDimension()
    assert n_dim == 3, "Only 3D images are supported for marching cubes"

    if sitk.GetArrayViewFromImage(src_img).sum() == 0:
        return vtk.vtkPolyData()

    # pad to avoid potential open boundary related issues
    if pad:
        src_img = sitk.ConstantPad(src_img, (1, 1, 1), (1, 1, 1), 0)

    vtkImage = sitk2vtk(src_img > 0)
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    meshing_alg = vtk.vtkSurfaceNets3D()
    meshing_alg.SmoothingOff()
    meshing_alg.SetOutputMeshTypeToTriangles()
    meshing_alg.SetInputData(vtkImage)
    meshing_alg.Update()
    mesh = meshing_alg.GetOutput()

    return mesh


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


def vtk_2D_centroid2surface_dist_length(
    pts_contour: vtk.vtkPolyData,
    surface_mesh: vtk.vtkPolyData,
) -> Tuple[np.ndarray, np.ndarray]:

    lines = pts_contour.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    assert (
        lines.GetMaxCellSize() == 2
    ), "This function supports segment lines that have 2 points"

    N = lines.GetNumberOfCells()
    dists_pts2surface, segment_lengths = np.zeros(N), np.zeros(N)

    vtk_p2s_dist = vtk.vtkImplicitPolyDataDistance()
    vtk_p2s_dist.SetInput(surface_mesh)

    # Iterate over each polyline in vtkPolyData
    cnt = 0
    while lines.GetNextCell(id_list):
        point_id1 = id_list.GetId(0)
        point_id2 = id_list.GetId(1)

        # Get the coordinates of the two points
        pt0 = np.array(pts_contour.GetPoint(point_id1))
        pt1 = np.array(pts_contour.GetPoint(point_id2))

        segment_lengths[cnt] = np.linalg.norm(pt0 - pt1)
        dists_pts2surface[cnt] = abs(vtk_p2s_dist.FunctionValue((pt0 + pt1) / 2))
        cnt += 1
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bidirectional distances between contour centroids and the opposing surface.

    This function measures distances from the centroids of one contour (reference or
    prediction) to the surface generated from the opposing contour. Distances are computed
    in both directions (ref→pred and pred→ref), along with the lengths of the corresponding
    line segments. Distances and the corresponding segment lengths are then sorted.

    Note:
        Since `vtk.vtkImplicitPolyDataDistance` requires a 3D `vtkPolyData`, the input
        2D `sitk.Image` contours are lifted into 3D by adding a singleton axis and then
        meshed into open 3D surfaces.

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
    # lift 2D contours into 3D by adding a singleton axis and mesh into open surfaces
    ref_sitk_3D, pred_sitk_3D = sitk_add_axis_to_end(ref_sitk), sitk_add_axis_to_end(pred_sitk)
    # note that the surfaces should be created using surface nets
    ref_surface, pred_surface = vtk_3D_meshing(ref_sitk_3D, pad=False), vtk_3D_meshing(pred_sitk_3D, pad=False)

    # compute distances between contour centroids and opposing surface
    dists_ref2pred, segment_lengths_ref = vtk_2D_centroid2surface_dist_length(ref_contour, pred_surface)
    dists_pred2ref, segment_lengths_pred = vtk_2D_centroid2surface_dist_length(pred_contour, ref_surface)

    # sort distances and boundary sizes
    dists_ref2pred, segment_lengths_ref = sort_dists_and_bsizes(dists_ref2pred, segment_lengths_ref)
    dists_pred2ref, segment_lengths_pred = sort_dists_and_bsizes(dists_pred2ref, segment_lengths_pred)
    # fmt: on

    return dists_ref2pred, segment_lengths_ref, dists_pred2ref, segment_lengths_pred


def vtk_compute_cell_sizes(mesh: vtk.vtkPolyData) -> np.ndarray:
    N_faces = mesh.GetNumberOfCells()
    cell_sizes = np.zeros(N_faces)
    for enum in range(N_faces):
        cell_sizes[enum] = mesh.GetCell(enum).ComputeArea()
    return cell_sizes


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
    return (inds * spacing + origin) @ direction.T


def compute_distance_field(pts_np: np.ndarray, mesh_vtk: vtk.vtkPolyData):
    vtk_p2s_dist = vtk.vtkImplicitPolyDataDistance()
    vtk_p2s_dist.SetInput(mesh_vtk)
    return np.array([abs(vtk_p2s_dist.FunctionValue(pt)) for pt in pts_np])


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

    n_dim = ref_sitk.GetDimension()
    spacing = np.array(ref_sitk.GetSpacing())
    origin = np.array(ref_sitk.GetOrigin())
    direction = np.array(ref_sitk.GetDirection()).reshape(n_dim, n_dim)

    # get foreground pixel/voxel coordinates in world space
    ref_np, pred_np = sitk2np(ref_sitk), sitk2np(pred_sitk)
    ref_world = index2world(
        np.stack(np.nonzero(ref_np), axis=1), spacing, origin, direction
    )
    pred_world = index2world(
        np.stack(np.nonzero(pred_np), axis=1), spacing, origin, direction
    )

    if n_dim == 2:
        # add z axis with 0 values
        ref_world = np.concatenate(
            [ref_world, np.zeros((ref_world.shape[0], 1))], axis=1
        )
        pred_world = np.concatenate(
            [pred_world, np.zeros((pred_world.shape[0], 1))], axis=1
        )

        ref_sitk_3D, pred_sitk_3D = sitk_add_axis_to_end(
            ref_sitk
        ), sitk_add_axis_to_end(pred_sitk)
        ref_surface, pred_surface = vtk_3D_meshing(
            ref_sitk_3D, pad=False
        ), vtk_3D_meshing(pred_sitk_3D, pad=False)
    else:
        ref_surface, pred_surface = ref_mesh, pred_mesh

    # initialize distance fields with zeros, then compute distances only for foreground pixels/voxels
    ref_dist_field, pred_dist_field = np.copy(ref_np).astype(np.float32), np.copy(
        pred_np
    ).astype(np.float32)
    ref_dist_field[ref_dist_field > 0] = compute_distance_field(ref_world, ref_surface)
    pred_dist_field[pred_dist_field > 0] = compute_distance_field(
        pred_world, pred_surface
    )
    return ref_dist_field, pred_dist_field


def vtk_voxelizer(mesh_vtk: vtk.vtkPolyData, meta_sitk: sitk.Image):
    """
    Voxelize a VTK mesh using SimpleITK image metadata, correctly handling image direction.
    
    Parameters:
    -----------
    mesh_vtk : vtk.vtkPolyData
        Input mesh to voxelize
    meta_sitk : sitk.Image
        Reference SimpleITK image providing spacing, origin, direction, and extent
    
    Returns:
    --------
    sitk.Image
        Voxelized binary mask matching the reference image geometry
    """
    assert isinstance(mesh_vtk, vtk.vtkPolyData), "Mesh must be vtkPolyData"
    assert isinstance(meta_sitk, sitk.Image), "Segmentation must be SimpleITK image"

    # check for empty mesh or empty image
    if mesh_vtk is None or np.prod(meta_sitk.GetSize()) == 0:
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
        transform_filter = vtk.vtkTransformPolyDataFilter()
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
    voxelized_sitk = vtk2sitk(stencil_to_image.GetOutput())
    
    # For 2D cases, remove the added axial dimension
    if ndim == 2:
        voxelized_sitk = voxelized_sitk[:, :, 0]
    
    # Set the correct metadata
    voxelized_sitk.SetOrigin(meta_sitk.GetOrigin())
    voxelized_sitk.SetSpacing(meta_sitk.GetSpacing())
    voxelized_sitk.SetDirection(meta_sitk.GetDirection())
    
    return voxelized_sitk

def get_mesh_bounds(mesh: vtk.vtkPolyData) -> np.ndarray:
    bounds = np.array(mesh.GetBounds())
    if np.allclose(bounds, (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)):
        bounds[:] = np.nan
    return bounds


def vtk_meshes_bbox_sitk_image(
    mesh1: vtk.vtkPolyData,
    mesh2: vtk.vtkPolyData,
    spacing: tuple,
    tolerance: tuple = None,
) -> sitk.Image:
    ndim = len(spacing)

    # if both meshes are empty, return an empty sitk image
    if mesh1.GetNumberOfPoints() == 0 and mesh2.GetNumberOfPoints() == 0:
        meta_sitk = sitk.GetImageFromArray(np.zeros((0,) * ndim))
        meta_sitk.SetSpacing(spacing)
        return meta_sitk

    # create a meta image SimpleITK that encompasses both masks
    ref_b = get_mesh_bounds(mesh1)
    pred_b = get_mesh_bounds(mesh2)
    ref_origin, ref_diagonal = ref_b[::2], ref_b[1::2]
    pred_origin, pred_diagonal = pred_b[::2], pred_b[1::2]

    # find element-wise minimum and maximum
    origin = np.nanmin((ref_origin, pred_origin), axis=0)[:ndim]
    diagonal = np.nanmax((ref_diagonal, pred_diagonal), axis=0)[:ndim]

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
        vtk_mesh1, vtk_mesh2, spacing, tolerance=5 * np.array(spacing)
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
        vtk_mesh1, vtk_mesh2, spacing, tolerance=5 * np.array(spacing)
    )
    sitk_mask1 = vtk_voxelizer(vtk_mesh1, meta_sitk)
    sitk_mask2 = vtk_voxelizer(vtk_mesh2, meta_sitk)

    return vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2


def vtk_write_polydata(vtk_polydata: vtk.vtkPolyData, pth: Union[str, Path]):
    assert isinstance(vtk_polydata, vtk.vtkPolyData), "Unknown mesh type"

    writer = vtk.vtkOBJWriter()
    writer.SetInputData(vtk_polydata)
    writer.SetFileName(str(pth))
    writer.Write()


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
        cells.SetCells(faces.shape[0], vtk_faces)
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

    # Find cells that can be treated as polygons/triangles
    # mesh.cells is a list of (cell_type, array_of_indices)
    polys = []
    for cell_block in mesh.cells:
        cell_type, data = cell_block.type, cell_block.data
        if cell_type in ("triangle", "quad", "polygon"):
            polys.append(data)
    if polys:
        polys = np.vstack(polys)
        faces_flat = np.hstack(
            [np.full((polys.shape[0], 1), polys.shape[1], dtype=np.int64), polys]
        ).ravel()
        vtk_faces = numpy_to_vtkIdTypeArray(faces_flat, deep=True)
        cells = vtk.vtkCellArray()
        cells.SetCells(polys.shape[0], vtk_faces)
        polydata.SetPolys(cells)
    else:
        # fallback: only vertices
        polydata.SetVerts(vtk.vtkCellArray())

    polydata.BuildCells()
    polydata.BuildLinks()
    return polydata
