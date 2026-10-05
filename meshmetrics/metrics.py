from __future__ import annotations

from functools import cached_property
import numbers
import warnings
from typing import Dict, Iterable, Optional, Tuple, Union

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
    vtk_contour_segments,
    _vtk_extrude_contour,
    vtk_voxelizer,
    vtk_is_mesh_closed,
    vtk_meshes_bbox_sitk_image,
    vtk_points_outside_image,
    vtk_mesh_index_bounds,
    crop_to_foreground,
    crop_np_to_foreground,
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

            - Masks (numpy arrays and SimpleITK images) are cropped to the bounding box of their
            foreground (and of the mesh, for mixed inputs) plus one pixel/voxel, which speeds up
            the computation without changing the results. `ref_sitk`, `pred_sitk`, `ref_np` and
            `pred_np` therefore hold the cropped masks (SimpleITK images keep their physical
            position).
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
        ref, pred = crop_np_to_foreground([ref, pred])

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
        ref, pred = crop_to_foreground([ref, pred])

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
            [self.ref_vtk, self.pred_vtk],
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

        The image grid is kept (cropped to the image foreground and the mesh) and the mesh is
        voxelized onto it.
        """
        self.clear_cache()
        self.spacing = img.GetSpacing()

        # convert the mesh first, so that the image can be cropped to include it
        setattr(self, f"{mesh_name}_vtk", mesh)
        mesh_vtk = getattr(self, f"{mesh_name}_vtk")
        if vtk_points_outside_image(mesh_vtk, img):
            warnings.warn(
                f"`{mesh_name}` mesh extends beyond the `{img_name}` image grid; "
                "the part outside is clipped in grid-based metrics (DSC, IoU, BIoU)"
            )
        (img,) = crop_to_foreground([img], extra_bounds=[vtk_mesh_index_bounds(mesh_vtk, img)])

        setattr(self, f"{img_name}_sitk", img)
        setattr(self, f"{img_name}_np", img)
        setattr(self, f"{img_name}_vtk", img)

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
            # Non-manifold verts/edges are not checked, as they do not impact any of the
            # calculations, which all rely on absolute distances to the mesh surface.
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
            ref_contour, pred_contour = self._contours_2d
            ref_surface, pred_surface = self._surfaces_2d
            d_ref2pred, b_ref, d_pred2ref, b_pred = vtk_measurements_2D(
                ref_contour=ref_contour,
                pred_contour=pred_contour,
                ref_sitk=self.ref_sitk,
                pred_sitk=self.pred_sitk,
                ref_surface=ref_surface,
                pred_surface=pred_surface,
            )
        elif self.n_dim == 3:
            d_ref2pred, b_ref, d_pred2ref, b_pred = vtk_measurements_3D(
                ref_mesh=self.ref_vtk,
                pred_mesh=self.pred_vtk,
            )
        else:
            raise ValueError("Only 2D and 3D masks are supported")

        return d_ref2pred, b_ref, d_pred2ref, b_pred

    @cached_property
    def _contours_2d(self) -> Tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
        """2D contours of ref and pred as 2-point segments: the given contours, or the contours
        created from the masks. Used both as sources and (extruded) targets of distances."""
        return tuple(vtk_contour_segments(getattr(self, f"{name}_vtk")) for name in ("ref", "pred"))

    @cached_property
    def _surfaces_2d(self) -> Tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
        """Extruded 2D contours, the targets of 2D distances (created once)."""
        return tuple(_vtk_extrude_contour(c) for c in self._contours_2d)

    def _dist_fields_within(self, max_dist: float) -> Tuple[np.ndarray, np.ndarray]:
        """Distance fields of ref and pred foreground voxels to their own surface.

        Distances are exact for voxels closer than `max_dist` to the surface; other
        foreground voxels may be set to inf. Results are cached and reused for any
        smaller `max_dist`.
        """
        if self._dist_fields is None or self._dist_fields[0] < max_dist:
            if self.n_dim == 2:
                extra = [
                    dict(contour_2d=c, surface_2d=s)
                    for c, s in zip(self._contours_2d, self._surfaces_2d)
                ]
            else:
                extra = [{}, {}]
            fields = tuple(
                vtk_mask_distance_field(
                    getattr(self, f"{name}_sitk"),
                    getattr(self, f"{name}_vtk"),
                    max_dist=max_dist,
                    **kw,
                )
                for name, kw in zip(("ref", "pred"), extra)
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
            return float(max(perc_d_ref2pred, perc_d_pred2ref))

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
            return float((mean_d_ref2pred + mean_d_pred2ref) / 2)

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
            return float(value)

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
            return float(num / denom)

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

            return float(num / denom)

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
            return float(2 * intersection / (union + intersection))

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
            return float(intersection / union)


ALL_METRICS = ("hd", "masd", "assd", "nsd", "biou", "dsc", "iou")


def compute_metrics(
    ref: Union[np.ndarray, sitk.Image, MeshTypes],
    pred: Union[np.ndarray, sitk.Image, MeshTypes],
    spacing: Union[tuple, list, np.ndarray] = None,
    taus: Iterable[float] = (),
    percentiles: Iterable[float] = (100, 95),
    metrics: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> Dict[str, Union[float, bool]]:
    """Compute several metrics for one pair of segmentations in a single call.

    Accepts the same inputs as `DistanceMetrics.set_input`.

    Args:
        ref, pred, spacing: See `DistanceMetrics.set_input`.
        taus: Tolerances (in physical units) for NSD and BIoU. These are application-specific,
            so NSD and BIoU are only computed for the given values.
        percentiles: Percentiles for HD (100 is the classic Hausdorff distance).
        metrics: Subset of ``("hd", "masd", "assd", "nsd", "biou", "dsc", "iou")`` to compute.
            Defaults to all, except NSD and BIoU when no `taus` are given.
        verbose: Warn about empty masks.

    Returns:
        dict with ``ref_is_empty``, ``pred_is_empty`` and one entry per metric, e.g.
        ``HD_100``, ``HD_95``, ``MASD``, ``ASSD``, ``NSD_2.0``, ``BIoU_2.0``, ``DSC``, ``IoU``.

    Example:
        >>> results = compute_metrics(ref_sitk, pred_sitk, taus=(1.0, 2.0))
    """
    taus, percentiles = list(taus), list(percentiles)
    if metrics is None:
        metrics = [m for m in ALL_METRICS if taus or m not in ("nsd", "biou")]
    metrics = [m.lower() for m in metrics]
    unknown = sorted(set(metrics) - set(ALL_METRICS))
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}, available: {list(ALL_METRICS)}")
    if not taus and {"nsd", "biou"} & set(metrics):
        raise ValueError("NSD and BIoU need at least one tolerance in `taus`")
    if not percentiles and "hd" in metrics:
        raise ValueError("HD needs at least one value in `percentiles`")

    dm = DistanceMetrics(verbose=verbose)
    dm.set_input(ref, pred, spacing=spacing)

    results = {"ref_is_empty": dm.ref_is_empty, "pred_is_empty": dm.pred_is_empty}
    for m in ALL_METRICS:  # fixed output order
        if m not in metrics:
            continue
        if m == "hd":
            for p in percentiles:
                results[f"HD_{p:g}"] = dm.hd(percentile=p)
        elif m in ("nsd", "biou"):
            fn = dm.nsd if m == "nsd" else dm.biou
            # largest tau first: BIoU reuses the cached distance band for smaller taus
            values = {tau: fn(tau=tau) for tau in sorted(taus, reverse=True)}
            name = "NSD" if m == "nsd" else "BIoU"
            for tau in taus:
                results[f"{name}_{float(tau)}"] = values[tau]
        else:
            results[{"iou": "IoU"}.get(m, m.upper())] = getattr(dm, m)()
    return results


def _label_bounding_boxes(label_img: sitk.Image) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Index bounds (lo, hi; inclusive) of every non-zero label, in a single pass."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.ComputePerimeterOff()
    stats.ComputeFeretDiameterOff()
    stats.ComputeOrientedBoundingBoxOff()
    stats.Execute(sitk.Cast(label_img, sitk.sitkUInt32))
    ndim = label_img.GetDimension()
    boxes = {}
    for label in stats.GetLabels():
        bbox = np.array(stats.GetBoundingBox(label))
        boxes[int(label)] = (bbox[:ndim], bbox[:ndim] + bbox[ndim:] - 1)
    return boxes


def _compute_metrics_for_label(job: tuple) -> Dict[str, Union[float, bool]]:
    """Worker: rebuild the cropped binary masks of one label and compute its metrics."""
    ref_np, pred_np, spacing, origin, direction, kwargs = job
    images = []
    for arr in (ref_np, pred_np):
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)
        images.append(img)
    return compute_metrics(*images, **kwargs)


def compute_metrics_multilabel(
    ref: Union[np.ndarray, sitk.Image],
    pred: Union[np.ndarray, sitk.Image],
    spacing: Union[tuple, list, np.ndarray] = None,
    labels: Optional[Iterable[int]] = None,
    taus: Iterable[float] = (),
    percentiles: Iterable[float] = (100, 95),
    metrics: Optional[Iterable[str]] = None,
    verbose: bool = True,
    n_jobs: int = 1,
) -> Dict[int, Dict[str, Union[float, bool]]]:
    """Compute metrics for every label of two label maps (e.g. multi-organ segmentations).

    Each label is evaluated as a binary segmentation, as with `compute_metrics`. The bounding
    boxes of all labels are found in a single pass over each label map, and each label is then
    computed on a small crop, so this is much faster than thresholding the full label maps once
    per label.

    Args:
        ref, pred: Label maps as SimpleITK images (same geometry), or numpy arrays (same shape;
            `spacing` required, in the order of the array axes, as in `set_input`).
        spacing: Pixel/voxel size, only for numpy inputs.
        labels: Labels to evaluate. Defaults to all non-zero labels present in either map. A
            label missing from one or both maps gets the values for empty masks.
        taus, percentiles, metrics, verbose: See `compute_metrics`.
        n_jobs: Number of parallel processes over labels (1: sequential).

    Returns:
        dict mapping each label to the result dict of `compute_metrics`.
    """
    if isinstance(ref, np.ndarray) or isinstance(pred, np.ndarray):
        assert isinstance(ref, np.ndarray) and isinstance(
            pred, np.ndarray
        ), "if `ref` is a numpy array, `pred` must also be a numpy array and vice versa"
        assert spacing is not None, "spacing must be provided for numpy inputs"
        assert ref.shape == pred.shape, "label maps must have the same shape"
        assert ref.ndim == len(spacing), "label maps and spacing must have the same dimensionality"
        ref, pred = np2sitk(ref), np2sitk(pred)
        ref.SetSpacing(tuple(float(s) for s in spacing))
        pred.SetSpacing(tuple(float(s) for s in spacing))
    else:
        assert isinstance(ref, sitk.Image) and isinstance(
            pred, sitk.Image
        ), "label maps must be numpy arrays or SimpleITK images"
        assert spacing is None, "spacing must not be provided for SimpleITK images"
        assert ref.GetSize() == pred.GetSize(), "label map size must be the same"
        assert np.allclose(ref.GetOrigin(), pred.GetOrigin()), "label map origin must be the same"
        assert np.allclose(ref.GetSpacing(), pred.GetSpacing()), "label map spacing must be the same"
        assert np.allclose(
            ref.GetDirection(), pred.GetDirection()
        ), "label map direction must be the same"

    ref_boxes, pred_boxes = _label_bounding_boxes(ref), _label_bounding_boxes(pred)
    if labels is None:
        labels = sorted(set(ref_boxes) | set(pred_boxes))
    labels = [int(label) for label in labels]

    kwargs = dict(taus=tuple(taus), percentiles=tuple(percentiles), metrics=metrics, verbose=verbose)
    size = np.array(ref.GetSize())
    jobs = []
    for label in labels:
        boxes = [b for b in (ref_boxes.get(label), pred_boxes.get(label)) if b is not None]
        if boxes:  # union of both boxes, plus one pixel/voxel
            lo = np.maximum(np.min([b[0] for b in boxes], axis=0) - 1, 0)
            hi = np.minimum(np.max([b[1] for b in boxes], axis=0) + 1, size - 1)
        else:  # label in neither map: a single (empty) pixel/voxel
            lo = hi = np.zeros(len(size), int)
        roi_size, roi_index = (hi - lo + 1).tolist(), lo.tolist()
        crops = [sitk.RegionOfInterest(img, roi_size, roi_index) for img in (ref, pred)]
        masks = [sitk.GetArrayFromImage(c == label).astype(np.uint8) for c in crops]
        jobs.append((*masks, crops[0].GetSpacing(), crops[0].GetOrigin(), crops[0].GetDirection(), kwargs))

    if n_jobs == 1 or len(jobs) <= 1:
        results = [_compute_metrics_for_label(job) for job in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_compute_metrics_for_label, jobs))
    return dict(zip(labels, results))


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
