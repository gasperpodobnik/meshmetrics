# AGENTS.md

Guidance for AI coding assistants working on, or with, MeshMetrics.

## What this is

MeshMetrics computes distance-based segmentation metrics (HD, HD<sub>p</sub>, MASD, ASSD, NSD,
BIoU, plus DSC and IoU) on mesh representations, for 2D and 3D. Inputs can be numpy arrays,
SimpleITK images, VTK/trimesh/meshio meshes, or a mix.

- **Names:** the PyPI package is `pymeshmetrics` (`pip install pymeshmetrics`), but the import
  is `import meshmetrics`. `MeshMetrics.py` is a deprecated alias for the old
  `import MeshMetrics`; do not add code to it.
- **Main API:** `compute_metrics(ref, pred, spacing=None, taus=(), ...)` for one-call use, and
  the `DistanceMetrics` class (`set_input`, then `hd()`, `masd()`, `assd()`, `nsd(tau)`,
  `biou(tau)`, `dsc()`, `iou()`). Both are in `meshmetrics/metrics.py`; helpers are in
  `meshmetrics/utils.py` and re-exported from `meshmetrics/__init__.py`.

## Useful utilities

When users need these, use the library's functions rather than writing new ones:

- **`vtk_voxelizer(mesh, reference_image)`** rasterizes a closed 3D mesh, or a closed 2D
  contour, onto the grid of any SimpleITK image. It respects spacing, origin and arbitrary
  direction matrices (rotations, flips). Pixels/voxels whose centre lies inside the mesh are set
  to 1. It is tested against exact solutions for rotated and anisotropic grids.
  `vtk_voxelizer(mesh, spacing=...)` instead creates an axis-aligned grid around the mesh
  bounds, padded by 2 pixels/voxels.
- **`vtk_meshing(image)`** goes from mask to mesh (discrete marching cubes in 3D, discrete
  flying edges in 2D), with points in world coordinates.
- `vtk_read_polydata` / `vtk_write_polydata` (`.obj`, `.vtk`, `.stl`, `.vtp`),
  `trimesh_to_vtk`, `meshio_to_vtk`, `np2sitk` / `sitk2np`.

## Development

```bash
uv sync --all-extras   # environment with dev tools and all extras
uv run pytest          # full test suite (about 1 min)
```

- `tests/test_analytic_spheres.py` compares against a closed-form solution (two non-concentric
  spheres). It is the reference for correctness: a change that worsens it is a regression.
- Run the full suite after any change to meshing, voxelization, distances or geometry.

## Conventions and pitfalls

- **Meshing is fixed** to discrete marching cubes (3D) and discrete flying edges (2D). Changing
  the meshing changes every distance metric value, and is a breaking, major-version change.
- **Axis order:** `sitk2np` and `np2sitk` use (x, y[, z]) order, matching `image.GetSize()`.
  SimpleITK's own `GetArrayFromImage` uses (z, y, x). Spacing given with numpy input must follow
  the array's axis order.
- **Geometry:** always handle the image direction matrix. World coordinates are
  `origin + (index * spacing) @ direction.T` (see `index2world`). Test new geometry code on
  rotated, anisotropic, non-cubic grids, because identity directions and cubic grids hide bugs.
- **2D:** contours lie in the z = 0 plane. Distances to 2D boundaries use the contour extruded
  along z (`vtk_2D_mask_surface`).
- **Cropping:** `set_input` crops masks to the bounding box of their foreground (and of the
  mesh, for mixed inputs) plus one pixel/voxel, via `crop_to_foreground` /
  `crop_np_to_foreground`. Results do not depend on the image size; `ref_sitk` etc. hold the
  cropped masks.
- **BIoU** is computed on the grid, with exact distances only near the boundary (a provable
  lower bound selects the voxels). It raises `ValueError` when `tau` is too small for any
  boundary voxel.
- Support Python ≥ 3.10 and VTK ≥ 9.2. VTK behaviour changes between versions, so do not rely
  on version-specific filters without testing them.

## Releases

Bump `version` in `pyproject.toml`, run `uv lock`, commit, tag `vX.Y.Z` and publish a GitHub
release. `.github/workflows/publish.yml` then tests, builds and uploads to PyPI via trusted
publishing. The tag must match the version.
