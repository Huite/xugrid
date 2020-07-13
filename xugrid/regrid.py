from typing import Tuple

import numba as nb
import numpy as np
import scipy.sparse
import xarray as xr

import xugrid

PARALLEL = True
IntArray = np.ndarray
FloatArray = np.ndarray


def find_topology(ds: xr.Dataset) -> str:
    for var in ds.data_vars:
        attrs = ds[var].attrs
        if "cf_role" in attrs:
            if attrs["cf_role"].lower() == "mesh_topology":
                return var
    raise ValueError("mesh_topology not in dataset")


def get_faces_vertices(ds: xr.Dataset) -> Tuple[str, str]:
    topology_var = find_topology(ds)
    mesh_attrs = ds[topology_var].attrs

    faces_var = mesh_attrs["face_node_connectivity"]
    faces = ds[faces_var].values.astype(np.int64)

    x_var, y_var = map(str.strip, mesh_attrs["node_coordinates"].split())
    x = ds[x_var].values.astype(np.float64)
    y = ds[y_var].values.astype(np.float64)
    vertices = np.stack([x, y], axis=1)
    return faces, vertices


@nb.njit(parallel=PARALLEL)
def compute_overlap(
    src_faces: IntArray,
    src_vertices: FloatArray,
    src_ind: IntArray,
    dst_faces: IntArray,
    dst_vertices: FloatArray,
    dst_ind: IntArray,
) -> FloatArray:
    n_overlap = src_ind.size
    out = np.zeros(n_overlap, dtype=np.float64)
    for i in nb.prange(n_overlap):  # pylint: disable=not-an-iterable
        src_i = src_ind[i]
        dst_i = dst_ind[i]
        a = src_vertices[src_faces[src_i]]
        b = dst_vertices[dst_faces[dst_i]]
        if xugrid.geometry.separating_axis.polygons_intersect(a, b):
            out[i] = xugrid.geometry.sutherland_hodgman.area_of_intersection(a, b)
    return out


def compute_weights(
    src_faces: IntArray,
    src_vertices: FloatArray,
    dst_faces: IntArray,
    dst_vertices: FloatArray,
) -> scipy.sparse.csr_matrix:
    dst_celltree = xugrid.CellTree2D(dst_faces, dst_vertices)
    dst_bboxes = xugrid.geometry.cell_tree.build_bboxes(src_faces, src_vertices)
    dst_ind, src_ind = dst_celltree.locate_bboxes(dst_bboxes)
    overlap = compute_overlap(
        src_faces, src_vertices, src_ind, dst_faces, dst_vertices, dst_ind
    )
    intersects = overlap != 0.0
    data = overlap[intersects]
    i = dst_ind[intersects]
    j = src_ind[intersects]
    weights = scipy.sparse.coo_matrix((data, (i, j)))
    return weights.to_csr()


@nb.njit
def numba_area_mean(indptr, indices, overlap, src_data):
    n_dst = indptr.size - 1
    dst_data = np.full(n_dst, np.nan)
    for i in range(n_dst):
        v_acc = 0.0  # value accumulator
        w_acc = 0.0  # weight accumulator
        # nzi for non-zero index
        for nzi in range(indptr[i], indptr[i + 1]):
            w = overlap[nzi]  # grab weight
            v = src_data[indices[nzi]]  # grab src value
            v_acc += v * w
            w_acc += w
        dst_data[i] = v_acc / w_acc
    return dst_data


def scipy_area_mean(weights, src_data):
    weight_sum = weights.sum(axis=1).transpose()
    return weights.dot(src_data) / weight_sum


class Regridder:
    """A basic area-weighted regridder for 2D unstructured meshes"""

    def __init__(self, source: xr.Dataset, destination: xr.Dataset) -> None:
        """
        Parameters
        ----------
        source : xr.Dataset
            Contains the mesh topology of the source data.
        destination : xr.Dataset
            Contains the mesh topology of the destination.
        """
        src_faces, src_vertices = get_faces_vertices(source)
        dst_faces, dst_vertices = get_faces_vertices(destination)
        self.weights = compute_weights(src_faces, src_vertices, dst_faces, dst_vertices)

    def regrid(self, source_data: xr.DataArray, method="mean") -> FloatArray:
        src_data = source_data.values
        dst_data = numba_area_mean(
            self.weights.indptr, self.weights.indices, self.weights.data, src_data,
        )
        return dst_data
