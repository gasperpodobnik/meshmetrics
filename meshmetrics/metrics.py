from functools import cached_property
import numbers
import warnings
from typing import Tuple, Union

import numpy as np
import vtk
import SimpleITK as sitk
from .utils import (
    vtk_meshing,
    np2sitk,
    sitk2np,
    vtk_measurements_2D,
    vtk_measurements_3D,
    vtk_mask_distance_field,
    vtk_voxelizer,
    vtk_is_mesh_closed,
    vtk_meshes_bbox_sitk_image,
    vtk_points_outside_image,
    trimesh_to_vtk,
    meshio_to_vtk,
)

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import meshio
except ImportError:
    meshio = None

MeshTypes = Union[vtk.vtkPolyData, "trimesh.Trimesh", "meshio.Mesh"]
RUNTIME_MESH_TYPES = tuple(
    t
    for t in (
        vtk.vtkPolyData,
        getattr(trimesh, "Trimesh", None),
        getattr(meshio, "Mesh", None),
    )
    if t is not None
)


class DistanceMetrics:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.clear_cache()

    def set_input(
        self,
        ref: Union[np.ndarray, sitk.Image, MeshTypes],
        pred: Union[np.ndarray, sitk.Image, MeshTypes],
        spacing: Union[tuple, list, np.ndarray] = None,
    ):
        """General input setter method that automatically detects the input type.

        Note:
            - If `ref` is numpy array, the `pred` must also be numpy array and vice versa. Also spacing must be set.
            This is to avoid potential issues with different spatial positions of segmentation masks
            (i.e. mesh in world coordinate system and mask in local coordinate system).

            - If both `ref` and `pred` are mesh types, the `spacing` must be set.
            This is because some calculations are performed in grid space (see BIoU).

            - If both `ref` and `pred` are SimpleITK images, the spacing is automatically set from the images and should not be provided.

            - For all other combinations (`ref` is SimpleITK image and `pred` is mesh type or vice versa),
            the spacing will be inferred from the SimpleITK image input and should not be provided.
            The input mesh should be in world coordinates.
        """
        self.clear_cache()

        # both np.ndarray
        if isinstance(ref, np.ndarray) or isinstance(pred, np.ndarray):
            assert isinstance(ref, np.ndarray) and isinstance(
                pred, np.ndarray
            ), "if `ref` is numpy array, pred must also be a numpy array and vice versa"
            assert (
                spacing is not None
            ), "spacing must be provided if either `ref` or `pred` are numpy arrays"
            self._set_input_numpy(ref, pred, spacing)
        # both sitk.Image
        elif isinstance(ref, sitk.Image) and isinstance(pred, sitk.Image):
            assert (
                spacing is None
            ), "spacing must not be provided if both `ref` and `pred` are SimpleITK images"
            self.spacing = ref.GetSpacing()
            self._set_input_SimpleITK(ref, pred)
        # both MeshTypes
        elif isinstance(ref, RUNTIME_MESH_TYPES) and isinstance(
            pred, RUNTIME_MESH_TYPES
        ):
            assert (
                spacing is not None
            ), "spacing must be provided if both `ref` and `pred` are mesh types (vtk.vtkPolyData, trimesh.Trimesh or meshio.Mesh)"
            self._set_input_vtk(ref, pred, spacing)
        # one is sitk.Image and the other is MeshTypes
        elif isinstance(ref, (sitk.Image, *RUNTIME_MESH_TYPES)) and isinstance(
            pred, (sitk.Image, *RUNTIME_MESH_TYPES)
        ):
            assert (
                spacing is None
            ), "spacing will be inferred from the SimpleITK image input"
            if isinstance(ref, sitk.Image):
                self._set_input_mixed(img_name="ref", img=ref, mesh_name="pred", mesh=pred)
            else:
                self._set_input_mixed(img_name="pred", img=pred, mesh_name="ref", mesh=ref)
        else:
            raise ValueError(
                "`ref` and `pred` must be numpy.ndarray, SimpleITK.Image, vtk.vtkPolyData, "
                f"trimesh.Trimesh or meshio.Mesh (got {type(ref).__name__} and {type(pred).__name__})"
            )

    def _set_input_numpy(
        self,
        ref: np.ndarray,
        pred: np.ndarray,
        spacing: Union[tuple, list, np.ndarray],
    ):
        self.clear_cache()
        assert (
            ref.ndim == pred.ndim == len(spacing)
        ), "masks and spacing must all have the same dimensionality"
        assert ref.shape == pred.shape, "masks must have the same shape"

        self.ref_np = ref
        self.pred_np = pred
        self.spacing = spacing

        ## set other representations
        self.ref_sitk = self.ref_np
        self.pred_sitk = self.pred_np
        self.ref_vtk = self.ref_np
        self.pred_vtk = self.pred_np

    def _set_input_SimpleITK(
        self,
        ref: sitk.Image,
        pred: sitk.Image,
    ):
        self.clear_cache()
        assert ref.GetSize() == pred.GetSize(), "input mask size must be the same"
        assert np.allclose(
            ref.GetOrigin(), pred.GetOrigin()
        ), "input mask origin must be the same"
        assert np.allclose(
            ref.GetSpacing(), pred.GetSpacing()
        ), "input mask spacing must be the same"
        assert np.allclose(
            ref.GetDirection(), pred.GetDirection()
        ), "input mask direction must be the same"

        self.ref_sitk = ref
        self.pred_sitk = pred

        ## set other representations
        self.ref_np = self.ref_sitk
        self.pred_np = self.pred_sitk
        self.ref_vtk = self.ref_sitk
        self.pred_vtk = self.pred_sitk

    def _set_input_vtk(
        self,
        ref: MeshTypes,
        pred: MeshTypes,
        spacing: Union[tuple, list, np.ndarray],
    ):
        """
        Set the input VTK polydata for reference and prediction meshes along with the spacing.
        Parameters
        ----------
        ref : vtk.vtkPolyData, trimesh.Trimesh, meshio.Mesh
            The reference mesh.
        pred : vtk.vtkPolyData, trimesh.Trimesh, meshio.Mesh
            The prediction mesh.
        spacing : Union[tuple, list, np.ndarray]
            The spacing is required, because some calculations are performed in grid space (see BIoU).
        Raises
        ------
        AssertionError
            If `ref` or `pred` are not instances of vtk.vtkPolyData, trimesh.Trimesh or meshio.Mesh.
            If `ref` or `pred` are not closed meshes.
        """
        self.clear_cache()

        self.ref_vtk = ref
        self.pred_vtk = pred
        self.spacing = spacing

        ## set other representations
        # create a meta image SimpleITK that encompasses both masks
        meta_sitk = vtk_meshes_bbox_sitk_image(
            self.ref_vtk,
            self.pred_vtk,
            spacing=self.spacing,
            tolerance=5 * np.array(self.spacing),
        )

        self.ref_sitk = vtk_voxelizer(self.ref_vtk, meta_sitk)
        self.pred_sitk = vtk_voxelizer(self.pred_vtk, meta_sitk)

        self.ref_np = self.ref_sitk
        self.pred_np = self.pred_sitk

    def _set_input_mixed(
        self,
        img_name: str,
        img: sitk.Image,
        mesh_name: str,
        mesh: MeshTypes,
    ):
        """One input is a SimpleITK image and the other a mesh (in world coordinates).

        The image grid is kept as is and the mesh is voxelized onto it.
        """
        self.clear_cache()
        self.spacing = img.GetSpacing()

        setattr(self, f"{img_name}_sitk", img)
        setattr(self, f"{img_name}_np", img)
        setattr(self, f"{img_name}_vtk", img)

        setattr(self, f"{mesh_name}_vtk", mesh)
        mesh_vtk = getattr(self, f"{mesh_name}_vtk")
        if vtk_points_outside_image(mesh_vtk, img):
            warnings.warn(
                f"`{mesh_name}` mesh extends beyond the `{img_name}` image grid; "
                "the part outside is clipped in grid-based metrics (DSC, IoU, BIoU)"
            )
        mesh_sitk = vtk_voxelizer(mesh_vtk, img)
        setattr(self, f"{mesh_name}_sitk", mesh_sitk)
        setattr(self, f"{mesh_name}_np", mesh_sitk)

    @property
    def n_dim(self):
        return len(self.spacing)

    def clear_cache(self):
        """Reset inputs and all cached results of this instance."""
        self._spacing = None
        for name in ("ref", "pred"):
            for kind in ("np", "sitk", "vtk"):
                setattr(self, f"_{name}_{kind}", None)
        # BIoU distance fields: (max_dist, ref_field, pred_field)
        self._dist_fields = None
        for name, attr in vars(type(self)).items():
            if isinstance(attr, cached_property):
                self.__dict__.pop(name, None)

    @property
    def spacing(self) -> tuple:
        return self._spacing

    @spacing.setter
    def spacing(self, value):
        if self._spacing is not None:
            assert len(value) == len(self._spacing) and np.allclose(
                value, self._spacing
            ), "spacing must be the same as the previously set spacing"
        else:
            assert isinstance(
                value, (list, tuple, np.ndarray)
            ), "spacing must be a list, tuple or numpy array"
            assert len(value) in [2, 3], "only 2D or 3D calculations are supported"
            self._spacing = tuple(value)

    def _set_np(self, name: str, value: Union[np.ndarray, sitk.Image, MeshTypes]):
        """Internal helper for ref_np and pred_np setters.
        name: 'ref' or 'pred'
        """
        attr = f"_{name}_np"
        sitk_attr = getattr(self, f"{name}_sitk", None)

        if getattr(self, attr) is not None:
            return

        if isinstance(value, np.ndarray):
            assert value.dtype == bool, f"{name}_np mask must be a boolean array"
            setattr(self, attr, value.astype("uint8"))

        elif isinstance(value, sitk.Image):
            assert (
                sitk_attr is not None
            ), f"{name}_sitk must exist before assigning {name}_np"
            assert id(value) == id(
                sitk_attr
            ), f"mask must be the same object as `{name}_sitk`"
            setattr(self, attr, sitk2np(value))

        elif isinstance(value, RUNTIME_MESH_TYPES):
            raise NotImplementedError(
                f"Conversion from {type(value)} to numpy.ndarray is not implemented for {name}_np"
            )

        else:
            raise ValueError(
                f"{name}_np mask must be a numpy.ndarray or SimpleITK.Image"
            )

    def _set_sitk(self, name: str, value: Union[np.ndarray, sitk.Image]):
        """Internal helper for ref_sitk and pred_sitk setters.
        name: 'ref' or 'pred'
        """
        attr = f"_{name}_sitk"
        np_attr = getattr(self, f"{name}_np")

        if getattr(self, attr) is not None:
            return

        if isinstance(value, np.ndarray):
            assert id(value) == id(
                np_attr
            ), f"mask must be the same object as `{name}_np`"
            img_sitk = np2sitk(value)
            img_sitk.SetSpacing(self.spacing)
            setattr(self, attr, img_sitk)

        elif isinstance(value, sitk.Image):
            # optional asserts about pixel type, number of labels, etc.
            setattr(self, attr, value)
            self.spacing = value.GetSpacing()

        elif isinstance(value, RUNTIME_MESH_TYPES):
            raise NotImplementedError(
                f"Conversion from {type(value)} to SimpleITK.Image needs "
                f"to happen in the {name}_sitk setter input"
            )

        else:
            raise ValueError("mask must be a numpy.ndarray or SimpleITK.Image")

    def _set_vtk(self, name: str, value: Union[np.ndarray, sitk.Image, MeshTypes]):
        """Internal helper for ref_vtk and pred_vtk setters.
        name: 'ref' or 'pred'
        """
        attr = f"_{name}_vtk"
        sitk_attr = getattr(self, f"{name}_sitk", None)
        np_attr = getattr(self, f"{name}_np", None)

        if getattr(self, attr) is not None:
            return

        if isinstance(value, (np.ndarray, sitk.Image)):
            assert (
                sitk_attr is not None
            ), f"{name}_sitk must be set before setting the mesh"
            if isinstance(value, np.ndarray):
                assert id(value) == id(
                    np_attr
                ), f"mask must be the same object as `{name}_np`"
            else:  # sitk.Image
                assert id(value) == id(
                    sitk_attr
                ), f"mask must be the same object as `{name}_sitk`"
            setattr(self, attr, vtk_meshing(sitk_attr))

        elif isinstance(value, RUNTIME_MESH_TYPES):
            if isinstance(value, vtk.vtkPolyData):
                pass
            elif trimesh is not None and isinstance(value, trimesh.Trimesh):
                value = trimesh_to_vtk(value)
            elif meshio is not None and isinstance(value, meshio.Mesh):
                value = meshio_to_vtk(value)
            else:
                raise ValueError(
                    f"{name}_vtk mesh must be vtk.vtkPolyData, trimesh.Trimesh or meshio.Mesh"
                )
            assert vtk_is_mesh_closed(value), f"{name} mesh must be closed"
            # We no longer check for non-manifold verts/edges, as some valid meshes can be non-manifold
            # due to a bug in vtk SurfaceNets. However, this does not impact any of the calculations,
            # as they all rely on absolute distances to the mesh surface.
            setattr(self, attr, value)

        else:
            raise ValueError(
                f"{name}_vtk mask must be a numpy.ndarray, SimpleITK.Image, or vtk.vtkPolyData"
            )

    @property
    def ref_np(self) -> np.ndarray:
        return self._ref_np

    @ref_np.setter
    def ref_np(self, value: np.ndarray):
        self._set_np("ref", value)

    @property
    def pred_np(self) -> np.ndarray:
        return self._pred_np

    @pred_np.setter
    def pred_np(self, value: np.ndarray):
        self._set_np("pred", value)

    @property
    def ref_sitk(self) -> sitk.Image:
        return self._ref_sitk

    @ref_sitk.setter
    def ref_sitk(self, value: Union[np.ndarray, sitk.Image]):
        self._set_sitk("ref", value)

    @property
    def pred_sitk(self) -> sitk.Image:
        return self._pred_sitk

    @pred_sitk.setter
    def pred_sitk(self, value: Union[sitk.Image, np.ndarray]):
        self._set_sitk("pred", value)

    @property
    def ref_vtk(self) -> vtk.vtkPolyData:
        return self._ref_vtk

    @ref_vtk.setter
    def ref_vtk(self, value: Union[np.ndarray, sitk.Image, MeshTypes]):
        self._set_vtk("ref", value)

    @property
    def pred_vtk(self) -> vtk.vtkPolyData:
        return self._pred_vtk

    @pred_vtk.setter
    def pred_vtk(self, value: Union[np.ndarray, sitk.Image, MeshTypes]):
        self._set_vtk("pred", value)

    @cached_property
    def ref_is_empty(self) -> bool:
        return self.ref_vtk.GetNumberOfPoints() == 0

    @cached_property
    def pred_is_empty(self) -> bool:
        return self.pred_vtk.GetNumberOfPoints() == 0

    @cached_property
    def distances(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.ref_is_empty or self.pred_is_empty:
            return None
        elif self.n_dim == 2:
            d_ref2pred, b_ref, d_pred2ref, b_pred = vtk_measurements_2D(
                ref_contour=self.ref_vtk,
                pred_contour=self.pred_vtk,
                ref_sitk=self.ref_sitk,
                pred_sitk=self.pred_sitk,
            )
        elif self.n_dim == 3:
            d_ref2pred, b_ref, d_pred2ref, b_pred = vtk_measurements_3D(
                ref_mesh=self.ref_vtk,
                pred_mesh=self.pred_vtk,
            )
        else:
            raise ValueError("Only 2D and 3D masks are supported")

        return d_ref2pred, b_ref, d_pred2ref, b_pred

    def _dist_fields_within(self, max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
        """Distance fields of ref and pred foreground voxels to their own surface.

        Distances are exact for voxels closer than `max_dist` to the surface; other
        foreground voxels may be set to inf. Results are cached and reused for any
        smaller `max_dist`.
        """
        if self._dist_fields is None or self._dist_fields[0] < max_dist:
            fields = tuple(
                vtk_mask_distance_field(
                    getattr(self, f"{name}_sitk"),
                    getattr(self, f"{name}_vtk"),
                    max_dist=max_dist,
                )
                for name in ("ref", "pred")
            )
            self._dist_fields = (max_dist, *fields)
        return self._dist_fields[1:]

    @property
    def img_dist_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full (exact) distance fields; foreground voxels to their own surface."""
        ref_dist_field_np, pred_dist_field_np = self._dist_fields_within(np.inf)
        return self.ref_np, ref_dist_field_np, self.pred_np, pred_dist_field_np

    @staticmethod
    def perc_surface_dist(dists, b_sizes, perc) -> float:
        if len(dists) > 0:
            cum_surfel_areas = np.cumsum(b_sizes) / np.sum(b_sizes)
            idx = np.searchsorted(cum_surfel_areas, perc / 100.0)
            idx = min(idx, len(dists) - 1)
            return dists[idx]
        else:
            return np.inf

    def hd(self, percentile: float = 100.0) -> float:
        """Hausdorff distance at the p-th percentile (HDp).

        Reference:
            https://archive.org/details/grundzgedermen00hausuoft/
            https://doi.org/10.1109/34.232073

        HDp is a non-parametric absolute metric that measures the maximum of the
        directed p-th percentile surface distances between two binary segmentation masks.

        Note:
            - `percentile = 100` corresponds to the classic Hausdorff distance.
            - `percentile = 95` (HD95) is widely used in medical image analysis
            to reduce sensitivity to outliers and noise.
            - Arbitrary percentile can be chosen depending on the application
            and the desired robustness.

        Args:
            percentile (float): The percentile of the surface distance distribution
                                to compute. Must be between 0 and 100.
                                Default is 100 (classic Hausdorff distance).

        Returns:
            float: The HDp value in [0, inf) in the same physical units as the input (e.g. mm).
            Returns 0.0 if both masks are empty, and inf if only one mask is empty.
        """

        assert 0 <= percentile <= 100, "percentile must be between 0 and 100"

        if self.ref_is_empty and self.pred_is_empty:
            if self.verbose:
                warnings.warn("Both masks are empty")
            return 0.0
        elif self.ref_is_empty or self.pred_is_empty:
            if self.verbose:
                warnings.warn("One of the masks is empty")
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            perc_d_ref2pred = self.perc_surface_dist(
                np.abs(d_ref2pred), b_ref, percentile
            )
            perc_d_pred2ref = self.perc_surface_dist(
                np.abs(d_pred2ref), b_pred, percentile
            )
            return max(perc_d_ref2pred, perc_d_pred2ref)

    def masd(self) -> float:
        """Mean average surface distance (MASD).
        Synonyms: mean surface distance.

        Reference:
            https://doi.org/10.1109/TMI.2005.851757

        MASD is a non-parametric absolute metric that measures the mean of the average
        directional surface distance between two binary segmentation masks.

        Returns:
            float: The MASD value in [0, inf) in the same physical units as the input (e.g. mm).
            Returns 0.0 if both masks are empty, and inf if only one mask is empty.
        """

        if self.ref_is_empty and self.pred_is_empty:
            if self.verbose:
                warnings.warn("Both masks are empty")
            return 0.0
        elif self.ref_is_empty or self.pred_is_empty:
            if self.verbose:
                warnings.warn("One of the masks is empty")
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            mean_d_ref2pred = np.dot(np.abs(d_ref2pred), b_ref) / b_ref.sum()
            mean_d_pred2ref = np.dot(np.abs(d_pred2ref), b_pred) / b_pred.sum()
            return (mean_d_ref2pred + mean_d_pred2ref) / 2

    def assd(self) -> float:
        """Average symmetric surface distance (ASSD).

        Reference:
            https://webdoc.sub.gwdg.de/ebook/serien/ah/reports/zib/zib2004/paperweb/reports/ZR-04-09.pdf

        ASSD is a non-parametric absolute metric that measures the mean bidirectional
        surface distance between two binary segmentation masks.

        Returns:
            float: The ASSD value in [0, inf) in the same physical units as the input (e.g. mm).
            Returns 0.0 if both masks are empty, and inf if only one mask is empty.
        """

        if self.ref_is_empty and self.pred_is_empty:
            if self.verbose:
                warnings.warn("Both masks are empty")
            return 0.0
        elif self.ref_is_empty or self.pred_is_empty:
            if self.verbose:
                warnings.warn("One of the masks is empty")
            return np.inf
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            num = np.dot(np.abs(d_ref2pred), b_ref) + np.dot(np.abs(d_pred2ref), b_pred)
            denom = b_ref.sum() + b_pred.sum()
            value = num / denom
            return value

    def nsd(self, tau: float) -> float:
        """Normalized surface distance (NSD).
        Synonyms: (normalized) surface dice.

        Reference:
            https://doi.org/10.2196/26151

        NSD is a parametric relative metric that quantifies the agreement between two binary
        masks by evaluating the proportion of surface points that lie within
        a specified tolerance distance.

        Note:
            - The choice of `tau` is application-specific and should reflect the
            maximum acceptable distance error for the task at hand.
            - At coarse image resolutions, quantization effects can occur when
            voxel size is comparable to or larger than `tau`.

        Args:
            tau (float): Distance tolerance for defining the tolerance region.
                         Must be greater or equal than zero.

        Returns:
            float: The NSD score in [0, 1].
            Returns 1.0 if both masks are empty, and 0 if only one mask is empty.
        """
        assert isinstance(tau, numbers.Real), "tolerance must be a real number"
        assert tau >= 0, "tolerance must be greater than or equal to zero"

        if self.ref_is_empty and self.pred_is_empty:
            if self.verbose:
                warnings.warn("Both masks are empty")
            return 1.0
        elif self.ref_is_empty or self.pred_is_empty:
            if self.verbose:
                warnings.warn("One of the masks is empty")
            return 0.0
        else:
            d_ref2pred, b_ref, d_pred2ref, b_pred = self.distances
            overlap_ref = b_ref[np.abs(d_ref2pred) <= tau].sum()
            overlap_pred = b_pred[np.abs(d_pred2ref) <= tau].sum()
            num = overlap_ref + overlap_pred
            denom = b_ref.sum() + b_pred.sum()
            return num / denom

    def biou(self, tau: float) -> float:
        """Boundary Intersection over Union (BIoU).

        Reference:
            https://doi.org/10.1109/CVPR46437.2021.01508

        BIoU is a parametric relative metric that measures the overlap between
        the boundary regions of two binary masks, and thus improves sensitivity
        of the well-known IoU metric to the boundary deviations.

        Note:
            - This metric is computed using a hybrid mesh-grid approach:
            the grid provides the representation, while precise distances
            to the mesh surface are used for calculations. This is because
            mesh thinning and boolean operations on meshes are not robustly
            implemented (yet).
            It actually serves as a good compromise, since BIoU itself is a hybrid
            between overlap-based and distance-based metrics.

        Args:
            tau (float): Distance tolerance for defining the boundary region.
                         Must be greater than zero.

        Returns:
            float: The BIoU score in [0, 1].
            Returns 1.0 if both masks are empty, and 0.0 if only one mask is empty.

        Raises:
            ValueError: If `tau` is so small (relative to the pixel/voxel spacing) that
            both boundary regions are empty, in which case BIoU is undefined.
        """

        assert isinstance(tau, numbers.Real), "tolerance must be a real number"
        assert tau > 0, "tolerance must be greater than zero"

        if self.ref_is_empty and self.pred_is_empty:
            if self.verbose:
                warnings.warn("Both masks are empty")
            return 1.0
        elif self.ref_is_empty or self.pred_is_empty:
            if self.verbose:
                warnings.warn("One of the masks is empty")
            return 0.0
        else:
            ref_dist_field_np, pred_dist_field_np = self._dist_fields_within(tau)

            ref_hollow = (ref_dist_field_np < tau) & self.ref_np.astype(bool)
            pred_hollow = (pred_dist_field_np < tau) & self.pred_np.astype(bool)

            num = (ref_hollow & pred_hollow).sum()
            denom = (ref_hollow | pred_hollow).sum()
            if denom == 0:
                raise ValueError(
                    f"BIoU is undefined for tau={tau}: no pixel/voxel centre lies closer than "
                    "tau to the mask boundary (boundary regions are empty). "
                    f"Use a larger tau, e.g. at least the pixel/voxel spacing {self.spacing}."
                )

            return num / denom

    def dsc(self) -> float:
        """Dice Similarity Coefficient (DSC).

        Reference:
            https://doi.org/10.2307/1932409

        DSC is a non-parametric relative metric that quantifies the overlap
        between two binary masks.

        Note:
            - This function calculates DSC on a regular grid (voxel-based). If surface
            meshes are supplied instead of volumetric masks, they are first rasterized
            or voxelized before the DSC calculation - in such cases, the user must provide
            the pixel/voxel spacing. This is because boolean operations
            on meshes are not robustly implemented (yet).

        Returns:
            float: The DSC score in [0, 1].
            Returns 1.0 if both masks are empty, and 0.0 if only one mask is empty.
        """

        if self.ref_is_empty and self.pred_is_empty:
            if self.verbose:
                warnings.warn("Both masks are empty")
            return 1.0
        elif self.ref_is_empty or self.pred_is_empty:
            if self.verbose:
                warnings.warn("One of the masks is empty")
            return 0.0
        else:
            intersection = np.logical_and(self.ref_np, self.pred_np).sum()
            union = np.logical_or(self.ref_np, self.pred_np).sum()
            return 2 * intersection / (union + intersection)

    def iou(self) -> float:
        """Intersection over Union (IoU).

        Reference:
            https://doi.org/10.1111/j.1469-8137.1912.tb05611.x

        IoU is a non-parametric relative metric that quantifies the overlap
        between two binary masks.

        Note:
            - This function calculates IoU on a regular grid (voxel-based). If surface
            meshes are supplied instead of volumetric masks, they are first rasterized
            or voxelized before the IoU calculation - in such cases, the user must provide
            the pixel/voxel spacing. This is because boolean operations
            on meshes are not robustly implemented (yet).

        Returns:
            float: The IoU score in [0, 1].
            Returns 1.0 if both masks are empty, and 0.0 if only one mask is empty.
        """

        if self.ref_is_empty and self.pred_is_empty:
            if self.verbose:
                warnings.warn("Both masks are empty")
            return 1.0
        elif self.ref_is_empty or self.pred_is_empty:
            if self.verbose:
                warnings.warn("One of the masks is empty")
            return 0.0
        else:
            intersection = np.logical_and(self.ref_np, self.pred_np).sum()
            union = np.logical_or(self.ref_np, self.pred_np).sum()
            return intersection / union


## test
if __name__ == "__main__":
    from .utils import create_synthetic_examples_2d

    # Create synthetic examples
    vtk_mesh1, vtk_mesh2, sitk_mask1, sitk_mask2 = create_synthetic_examples_2d(
        r1=5.0, r2=10.0, spacing=(1.0, 1.0, 1.0)
    )

    # Set parameters
    percentile = 95  # percentile for HD
    tau = 2.0  # tolerance for NSD and BIoU

    # Initialize distance metrics class
    mesh_metrics = DistanceMetrics()

    ## ----- example (2D) -----
    mesh_metrics.set_input(sitk_mask1, sitk_mask2)
    # store flags indicating empty masks
    results = {
        "ref_is_empty": mesh_metrics.ref_is_empty,
        "pred_is_empty": mesh_metrics.pred_is_empty,
    }
    # Hausdorff distance (HD), by default, HD percentile is set to 100 (equivalent to HD)
    results["HD_100"] = mesh_metrics.hd()
    # p-th percentile HD (HD_p)
    results[f"HD_{percentile}"] = mesh_metrics.hd(percentile=percentile)
    # Mean average surface distance (MASD)
    results["MASD"] = mesh_metrics.masd()
    # Average symmetric surface distance (ASSD)
    results["ASSD"] = mesh_metrics.assd()
    # Normalized surface distance (NSD) with tau
    results[f"NSD_{tau}"] = mesh_metrics.nsd(tau=tau)
    # Boundary intersection over union (BIoU) with tau
    results[f"BIoU_{tau}"] = mesh_metrics.biou(tau=tau)

    # print metric values
    units = {"HD": "mm", "MASD": "mm", "ASSD": "mm", "NSD": "%", "BIoU": "%"}
    for k, v in results.items():
        unit = units.get(k.split("_")[0], "")
        f = 100.0 if unit == "%" else 1.0
        print(f"{k}: {v*f:.2f} {unit}")
