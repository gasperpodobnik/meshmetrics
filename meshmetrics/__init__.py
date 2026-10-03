from importlib.metadata import PackageNotFoundError, version

from .metrics import DistanceMetrics
from .utils import (
    # conversion
    np2sitk,
    sitk2np,
    to_sitk,
    to_vtk,
    trimesh_to_vtk,
    meshio_to_vtk,
    # mesh I/O
    vtk_read_polydata,
    vtk_write_polydata,
    # meshing & voxelization
    vtk_meshing,
    vtk_2D_meshing,
    vtk_3D_meshing,
    vtk_voxelizer,
    # mesh checks
    vtk_is_mesh_closed,
    vtk_is_mesh_manifold,
    # synthetic examples
    vtk_create_sphere,
    create_synthetic_examples_2d,
    create_synthetic_examples_3d,
)

try:
    __version__ = version("meshmetrics")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "DistanceMetrics",
    "np2sitk",
    "sitk2np",
    "to_sitk",
    "to_vtk",
    "trimesh_to_vtk",
    "meshio_to_vtk",
    "vtk_read_polydata",
    "vtk_write_polydata",
    "vtk_meshing",
    "vtk_2D_meshing",
    "vtk_3D_meshing",
    "vtk_voxelizer",
    "vtk_is_mesh_closed",
    "vtk_is_mesh_manifold",
    "vtk_create_sphere",
    "create_synthetic_examples_2d",
    "create_synthetic_examples_3d",
]
