from typing import NamedTuple, Tuple

import numba as nb
import numpy as np

from xugrid.geometry.common import (
    FLOAT_MAX,
    FLOAT_MIN,
    FLOAT_TYPE,
    FLOAT_TYPE_T,
    INT_MAX,
    INT_TYPE,
    INT_TYPE_T,
    PARALLEL,
    FloatArray,
    IntArray,
    stack_empty,
)

# -----------------------------
# Type declarations & constants
# -----------------------------

BucketArray = np.ndarray
NodeArray = np.ndarray
# Dimension of celltree. 2D is still rather hard-baked in, so changing this
# alone to 3 will NOT suffice to generalize it to a 3D CellTree.
NDIM = 2


class Point(NamedTuple):
    x: FLOAT_TYPE
    y: FLOAT_TYPE


class Vector(NamedTuple):
    x: FLOAT_TYPE
    y: FLOAT_TYPE


class Node(NamedTuple):
    child: INT_TYPE
    Lmax: FLOAT_TYPE
    Rmin: FLOAT_TYPE
    ptr: INT_TYPE
    size: INT_TYPE
    dim: bool


class Bucket(NamedTuple):
    Max: FLOAT_TYPE
    Min: FLOAT_TYPE
    Rmin: FLOAT_TYPE
    Lmax: FLOAT_TYPE
    index: INT_TYPE
    size: INT_TYPE


class CellTreeData(NamedTuple):
    faces: IntArray
    vertices: FloatArray
    nodes: NodeArray
    bb_indices: IntArray
    cells_per_leaf: int


node_dtype = np.dtype(
    [
        # Index of left child. Right child is child + 1.
        ("child", INT_TYPE),
        # Range of the bounding boxes inside of the node.
        ("Lmax", FLOAT_TYPE),
        ("Rmin", FLOAT_TYPE),
        # Index into the bounding box index array, bb_indices.
        ("ptr", INT_TYPE),
        # Number of bounding boxes in this node.
        ("size", INT_TYPE),
        # False = 0 = x, True = 1 = y.
        ("dim", bool),
    ]
)


bucket_dtype = np.dtype(
    [
        # Range of the bucket.
        ("Max", FLOAT_TYPE),
        ("Min", FLOAT_TYPE),
        # Range of the bounding boxes inside the bucket.
        ("Rmin", FLOAT_TYPE),
        ("Lmax", FLOAT_TYPE),
        # Index into the bounding box index array, bb_indices.
        ("index", INT_TYPE),
        # Number of bounding boxes in this bucket.
        ("size", INT_TYPE),
    ]
)

# ----------------------
# Tree building methods
# ----------------------


@nb.njit(inline="always")
def create_node(ptr: int, size: int, dim: bool) -> Node:
    return Node(-1, -1.0, -1.0, ptr, size, dim)


@nb.njit(inline="always")
def push(nodes: NodeArray, node: Node, index: int) -> int:
    """
    Push to the end of the array.
    """
    nodes[index]["child"] = node.child
    nodes[index]["Lmax"] = node.Lmax
    nodes[index]["Rmin"] = node.Rmin
    nodes[index]["ptr"] = node.ptr
    nodes[index]["size"] = node.size
    nodes[index]["dim"] = node.dim
    return index + 1


@nb.njit(inline="always")
def bounding_box(
    polygon: IntArray, vertices: FloatArray, max_n_verts: int
) -> Tuple[float, float, float, float]:
    first_vertex = vertices[polygon[0]]
    xmin = xmax = first_vertex[0]
    ymin = ymax = first_vertex[1]
    for i in range(1, max_n_verts):
        index = polygon[i]
        if index == -1:
            break
        vertex = vertices[index]
        x = vertex[0]
        y = vertex[1]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)
    return (xmin, xmax, ymin, ymax)


@nb.njit(parallel=PARALLEL)
def build_bboxes(
    faces: IntArray, vertices: FloatArray,
) -> Tuple[BucketArray, IntArray]:
    # Make room for the bounding box of every polygon.
    n_polys, max_n_verts = faces.shape
    bbox_indices = np.empty(n_polys, INT_TYPE)
    bbox_coords = np.empty((n_polys, NDIM * 2), FLOAT_TYPE)

    for i in nb.prange(n_polys):  # pylint: disable=not-an-iterable
        polygon = faces[i]
        bbox_coords[i] = bounding_box(polygon, vertices, max_n_verts)
        bbox_indices[i] = i

    return bbox_coords, bbox_indices


@nb.njit(inline="always")
def centroid_test(bucket: np.void, box: FloatArray, dim: int):
    """
    Tests whether the centroid of the bounding box in the selected dimension falls
    within this bucket.
    """
    centroid = box[2 * dim] + 0.5 * (box[2 * dim + 1] - box[2 * dim])
    return (centroid >= bucket.Min) and (centroid < bucket.Max)


@nb.njit(inline="always")
def stable_partition(
    bb_indices: IntArray,
    bb_coords: FloatArray,
    begin: int,
    end: int,
    bucket: np.void,
    dim: int,
) -> int:
    """
    Rearranges the elements in the range(begin, end), in such a way that all
    the elements for which a predicate returns True precede all those for which it
    returns False. The relative order in each group is maintained.

    In this case, the predicate is a `centroid_test`.

    Parameters
    ----------
    bb_indices: np.ndarray of ints
        Array to sort.
    bb_coords: np.ndarray of floats
        Coordinates of bounding boxes.
    begin, end: int
        Defines the range of arr in which to sort.
    bucket: np.void
        Element of BucketArray, contains data for a single bucket.
    dim: int
        Dimension number (0: x, 1: y, etc.)

    Returns
    -------
    current: int 
        Points to the first element of the second group for which predicate is True.
    """
    # Allocates a temporary buffer, ands fill from front and back: O(N)
    # A swapping algorithm can be found here, O(N log(N)):
    # https://csjobinterview.wordpress.com/2012/03/30/array-stable-partition/
    # via: https://stackoverflow.com/questions/21554635/how-is-stable-partition-an-adaptive-algorithm
    temp = np.empty(end - begin, dtype=bb_indices.dtype)
    # TODO: add statically allocated work-array? Then use views for size?

    count_true = 0
    count_false = -1
    for i in bb_indices[begin:end]:
        box = bb_coords[i]
        if centroid_test(bucket, box, dim):
            temp[count_true] = i
            count_true += 1
        else:
            temp[count_false] = i
            count_false -= 1

    for i in range(count_true):
        bb_indices[begin + i] = temp[i]

    start_second = begin + count_true
    for i in range(-1 - count_false):
        bb_indices[start_second + i] = temp[-i - 1]

    return start_second


@nb.njit(inline="always")
def sort_bbox_indices(
    bb_indices: IntArray,
    bb_coords: FloatArray,
    buckets: BucketArray,
    node: np.void,
    dim: int,
):
    current = node.ptr
    end = node.ptr + node.size

    b = buckets[0]
    buckets[0] = Bucket(b.Max, b.Min, b.Rmin, b.Lmax, node.ptr, b.size)

    i = 1
    while current != end:
        bucket = buckets[i - 1]
        current = stable_partition(bb_indices, bb_coords, current, end, bucket, dim)
        start = bucket.index

        b = buckets[i - 1]
        buckets[i - 1] = Bucket(b.Max, b.Min, b.Rmin, b.Lmax, b.index, current - start)

        if i < len(buckets):
            b = buckets[i]
            buckets[i] = Bucket(
                b.Max,
                b.Min,
                b.Rmin,
                b.Lmax,
                buckets[i - 1].index + buckets[i - 1].size,
                b.size,
            )

        start = current
        i += 1


@nb.njit(inline="always")
def get_bounds(
    index: int, size: int, bb_coords: FloatArray, bb_indices: IntArray, dim: int,
):
    Rmin = FLOAT_MAX
    Lmax = FLOAT_MIN
    for i in range(index, index + size):
        data_index = bb_indices[i]
        value = bb_coords[data_index, 2 * dim]
        if value < Rmin:
            Rmin = value
        value = bb_coords[data_index, 2 * dim + 1]
        if value > Lmax:
            Lmax = value
    return Rmin, Lmax


@nb.njit(inline="always")
def split_plane(
    buckets: BucketArray,
    root: np.void,
    range_Lmax: float,
    range_Rmin: float,
    bucket_length: float,
):
    plane_min_cost = FLOAT_MAX
    plane = INT_MAX
    bbs_in_left = 0
    bbs_in_right = 0

    # if we split here, lmax is from bucket 0, and rmin is from bucket 1 after
    # computing those, we can compute the cost to split here, and if this is the
    # minimum, we split here.
    for i in range(1, len(buckets)):
        current_bucket = buckets[i - 1]
        next_bucket = buckets[i]
        bbs_in_left += current_bucket.size
        bbs_in_right = root.size - bbs_in_left
        left_volume = (current_bucket.Lmax - range_Rmin) / bucket_length
        right_volume = (range_Lmax - next_bucket.Rmin) / bucket_length
        plane_cost = left_volume * bbs_in_left + right_volume * bbs_in_right
        if plane_cost < plane_min_cost:
            plane_min_cost = plane_cost
            plane = i
    return plane


@nb.njit
def pessimistic_n_nodes(n_polys: int):
    """
    In the worst case, *all* branches end in a leaf with a single cell. Rather
    unlikely in the case of non-trivial grids, but we need a guess to
    pre-allocate -- overestimation is at maximum two times in case of
    cells_per_leaf == 2.
    """
    n_nodes = n_polys
    nodes = int(np.ceil(n_polys / 2))
    while nodes > 1:
        n_nodes += nodes
        nodes = int(np.ceil(nodes / 2))
    # Return, add root.
    return n_nodes + 1


@nb.njit
def build(
    nodes: NodeArray,
    node_index: int,
    bb_indices: IntArray,
    bb_coords: FloatArray,
    n_buckets: int,
    cells_per_leaf: int,
    root_index: int,
    dim: int,
):
    """
    Recursively build the CellTree.
    """
    dim_flag = dim
    if dim < 0:
        dim += 2

    root = Node(
        nodes[root_index]["child"],
        nodes[root_index]["Lmax"],
        nodes[root_index]["Rmin"],
        nodes[root_index]["ptr"],
        nodes[root_index]["size"],
        nodes[root_index]["dim"],
    )

    # Is it a leaf? if so, we're done, otherwise split.
    if root.size <= cells_per_leaf:
        return node_index

    # Find bounding range of node's entire dataset in dimension 0 (x-axis).
    range_Rmin, range_Lmax = get_bounds(
        root.ptr, root.size, bb_coords, bb_indices, dim,
    )
    bucket_length = (range_Lmax - range_Rmin) / n_buckets

    # Create buckets
    buckets = []
    # Specify ranges on the buckets
    for i in range(n_buckets):
        buckets.append(
            Bucket(
                (i + 1) * bucket_length + range_Rmin,  # Max
                i * bucket_length + range_Rmin,  # Min
                -1.0,  # Rmin
                -1.0,  # Lmax
                -1,  # index
                0,  # size
            )
        )
    # NOTA BENE: do not change the default size (0) given to the bucket here
    # it is used to detect empty buckets later on.

    # Now that the buckets are setup, sort them
    sort_bbox_indices(bb_indices, bb_coords, buckets, root, dim)

    # Determine Lmax and Rmin for each bucket
    for i in range(n_buckets):
        Rmin, Lmax = get_bounds(
            buckets[i].index, buckets[i].size, bb_coords, bb_indices, dim
        )
        b = buckets[i]
        buckets[i] = Bucket(b.Max, b.Min, Rmin, Lmax, b.index, b.size)

    # Special case: 2 bounding boxes share the same centroid, but boxes_per_leaf
    # is 1 This will break most of the usual bucketing code. Unless the grid has
    # overlapping triangles (which it shouldn't!). This is the only case to deal
    # with
    if (cells_per_leaf == 1) and (root.size == 2):
        nodes[root_index]["Lmax"] = range_Lmax
        nodes[root_index]["Rmin"] = range_Rmin
        left_child = create_node(root.ptr, 1, not dim)
        right_child = create_node(root.ptr + 1, 1, not dim)
        nodes[root_index]["child"] = node_index
        node_index = push(nodes, left_child, node_index)
        node_index = push(nodes, right_child, node_index)

    while buckets[0].size == 0:
        b = buckets[1]
        buckets[1] = Bucket(b.Max, buckets[0].Min, b.Rmin, b.Lmax, b.index, b.size)
        buckets.pop(0)

    i = 1
    while i < len(buckets):
        next_bucket = buckets[i]
        # if a empty bucket is encountered, merge it with the previous one and
        # continue as normal. As long as the ranges of the merged buckets are
        # still proper, calcualting cost for empty buckets can be avoided, and
        # the split will still happen in the right place
        if next_bucket.size == 0:
            b = buckets[i - 1]
            buckets[i - 1] = Bucket(
                next_bucket.Max, b.Min, b.Rmin, b.Lmax, b.index, b.size
            )
            buckets.pop(i)
        else:
            i += 1

    # Check if all the cells are in one bucket. If so, restart and switch
    # dimension.
    for bucket in buckets:
        if bucket.size == root.size:
            if dim_flag >= 0:
                dim_flag = (not dim) - 2
                nodes[root_index]["dim"] = not root.dim
                node_index = build(
                    nodes,
                    node_index,
                    bb_indices,
                    bb_coords,
                    n_buckets,
                    cells_per_leaf,
                    root_index,
                    dim_flag,
                )
            else:  # Already split once, convert to leaf.
                nodes[root_index]["Lmax"] = -1
                nodes[root_index]["Rmin"] = -1
            return node_index

    # plane is the separation line to split on:
    # 0 [bucket0] 1 [bucket1] 2 [bucket2] 3 [bucket3]
    plane = split_plane(buckets, root, range_Lmax, range_Rmin, bucket_length)

    right_index = buckets[plane].index
    right_size = root.ptr + root.size - right_index
    left_index = root.ptr
    left_size = root.size - right_size
    nodes[root_index]["Lmax"] = buckets[plane - 1].Lmax
    nodes[root_index]["Rmin"] = buckets[plane].Rmin
    left_child = create_node(left_index, left_size, not dim)
    right_child = create_node(right_index, right_size, not dim)
    nodes[root_index]["child"] = node_index
    child_ind = node_index
    node_index = push(nodes, left_child, node_index)
    node_index = push(nodes, right_child, node_index)
    node_index = build(
        nodes,
        node_index,
        bb_indices,
        bb_coords,
        n_buckets,
        cells_per_leaf,
        child_ind,
        left_child.dim,
    )
    node_index = build(
        nodes,
        node_index,
        bb_indices,
        bb_coords,
        n_buckets,
        cells_per_leaf,
        child_ind + 1,
        right_child.dim,
    )
    return node_index


@nb.njit
def initialize(
    vertices: FloatArray, faces: IntArray, n_buckets: int = 4, cells_per_leaf: int = 2
):
    # Prepare bounding boxes for tree building.
    bb_coords, bb_indices = build_bboxes(faces, vertices)

    # Pre-allocate the space for the tree.
    n_polys, _ = faces.shape
    n_nodes = pessimistic_n_nodes(n_polys)
    nodes = np.empty(n_nodes, dtype=node_dtype)

    # Insert first node
    node = create_node(0, bb_indices.size, False)
    node_index = push(nodes, node, 0)

    # Recursively build the tree.
    node_index = build(
        nodes,
        node_index,
        bb_indices,
        bb_coords,
        n_buckets,
        cells_per_leaf,
        0,  # root_ind
        0,  # dim
    )

    # Remove the unused part in nodes.
    return nodes[:node_index], bb_indices


# ----------------------
# Point search functions
# ----------------------


@nb.njit(inline="always")
def point_in_polygon(
    bbox_index: int, point: Point, faces: IntArray, vertices: FloatArray,
) -> bool:
    face = faces[bbox_index]
    polygon_length = face.size

    c = False
    for i in range(polygon_length):
        v1 = vertices[face[i - 1]]
        v2 = vertices[face[i]]
        # Do not split this in two conditionals: if the first conditional fails,
        # the second will not be executed in Python's (and C's) execution model.
        # This matters because the second can result in division by zero.
        if (v1[1] > point[1]) != (v2[1] > point[1]) and point[0] < (
            (v2[0] - v1[0]) * (point[1] - v1[1]) / (v2[1] - v1[1]) + v1[0]
        ):
            c = not c

    return c


@nb.njit(inline="always")
def allocate_empty():
    # Maximum tree depth is hereby defined as 32.
    # Equals 2.1E9 cells if 1 cell per node; should be sufficient.
    arr_ptr = stack_empty(32, np.int64)  # pylint: disable=no-value-for-parameter
    arr = nb.carray(arr_ptr, 32, dtype=np.int64)
    return arr


@nb.njit(inline="always")
def _pop(array, size):
    return array[size - 1], size - 1


@nb.njit(inline="always")
def _push(array, value, size):
    array[size] = value
    return size + 1


@nb.njit(inline="always")
def locate_point(point: Point, tree: CellTreeData):
    stack = allocate_empty()
    stack[0] = 0
    return_value = -1
    size = 1

    while size > 0:
        node_index, size = _pop(stack, size)
        current = tree.nodes[node_index]

        # Check if it's a leaf
        if current["child"] == -1:
            for i in range(current["ptr"], current["ptr"] + current["size"]):
                bbox_index = tree.bb_indices[i]
                if point_in_polygon(bbox_index, point, tree.faces, tree.vertices):
                    return bbox_index
            continue

        dim = 1 if current["dim"] else 0
        left = point[dim] <= current["Lmax"]
        right = point[dim] >= current["Rmin"]

        if left and right:
            if (current["Lmax"] - point[dim]) < (point[dim] - current["Rmin"]):
                size = _push(stack, current["child"], size)
                size = _push(stack, current["child"] + 1, size)
            else:
                size = _push(stack, current["child"] + 1, size)
                size = _push(stack, current["child"], size)
        elif left:
            size = _push(stack, current["child"], size)
        elif right:
            size = _push(stack, current["child"] + 1, size)

    return return_value


@nb.njit(parallel=PARALLEL)
def locate_points(
    points: FloatArray, tree: CellTreeData,
):
    n_points = points.shape[0]
    result = np.empty(n_points, dtype=INT_TYPE)
    for i in nb.prange(n_points):  # pylint: disable=not-an-iterable
        point = Point(points[i, 0], points[i, 1])
        result[i] = locate_point(point, tree)
    return result


# -----------------------------
# Bounding box search functions
# -----------------------------


@nb.njit(inline="always")
def count_bbox(bbox: FloatArray, tree: CellTreeData):
    stack = allocate_empty()
    stack[0] = 0
    count = 0
    size = 1

    while size > 0:
        node_index, size = _pop(stack, size)
        current = tree.nodes[node_index]

        # Check if it's a leaf
        if current["child"] == -1:
            count += current["size"]

        dim = 1 if current["dim"] else 0
        left = bbox[dim * 2 + 1] <= current["Lmax"]
        right = bbox[dim * 2] >= current["Rmin"]

        if left and right:
            size = _push(stack, current["child"], size)
            size = _push(stack, current["child"] + 1, size)
        elif left:
            size = _push(stack, current["child"], size)
        elif right:
            size = _push(stack, current["child"] + 1, size)

    return count


@nb.njit(inline="always")
def locate_bbox(bbox: FloatArray, tree: CellTreeData, indices: IntArray):
    stack = allocate_empty()
    stack[0] = 0
    count = 0
    size = 1
    count = 0

    while size > 0:
        node_index, size = _pop(stack, size)
        current = tree.nodes[node_index]

        # Check if it's a leaf
        if current["child"] == -1:
            for i in range(current["ptr"], current["ptr"] + current["size"]):
                bbox_index = tree.bb_indices[i]
                indices[count] = bbox_index
                count += 1

        dim = 1 if current["dim"] else 0
        left = bbox[dim * 2 + 1] <= current["Lmax"]
        right = bbox[dim * 2] >= current["Rmin"]

        if left and right:
            size = _push(stack, current["child"], size)
            size = _push(stack, current["child"] + 1, size)
        elif left:
            size = _push(stack, current["child"], size)
        elif right:
            size = _push(stack, current["child"] + 1, size)

    return


@nb.njit(parallel=PARALLEL)
def locate_bboxes(
    bbox_coords: FloatArray, tree: CellTreeData,
):
    n_bbox = bbox_coords.shape[0]
    counts = np.empty(n_bbox + 1, dtype=INT_TYPE)
    counts[0] = 0
    for i in nb.prange(n_bbox):  # pylint: disable=not-an-iterable
        # counts[i + 1] = locate_bbox_helper_count_only(bbox_coords[i], 0, tree, 0)
        counts[i + 1] = count_bbox(bbox_coords[i], tree)

    # Run a cumulative sum
    total = 0
    for i in range(1, n_bbox + 1):
        total += counts[i]
        counts[i] = total

    ii = np.empty(total, dtype=INT_TYPE)
    jj = np.empty(total, dtype=INT_TYPE)
    for i in nb.prange(n_bbox):  # pylint: disable=not-an-iterable
        start = counts[i]
        end = counts[i + 1]
        ii[start:end] = i
        indices = jj[start:end]
        # locate_bbox_helper(bbox_coords[i], 0, tree, 0, indices)
        locate_bbox(bbox_coords[i], tree, indices)

    return ii, jj


# ----------------------
# Numba class definition
# ----------------------

celltree2d_spec = [
    ("vertices", FLOAT_TYPE_T[:, :]),
    ("faces", INT_TYPE_T[:, :]),
    ("n_buckets", INT_TYPE_T),
    ("cells_per_leaf", INT_TYPE_T),
    ("nodes", nb.from_dtype(node_dtype)[:]),
    ("bb_indices", INT_TYPE_T[:]),
]


@nb.experimental.jitclass(celltree2d_spec)
class CellTree2D:
    def __init__(
        self,
        vertices: FloatArray,
        faces: IntArray,
        n_buckets: int = 4,
        cells_per_leaf: int = 2,
    ):
        if n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if cells_per_leaf < 1:
            raise ValueError("cells_per_leaf must be >= 1")
        self.vertices = vertices
        self.faces = faces
        self.n_buckets = n_buckets
        self.cells_per_leaf = cells_per_leaf
        nodes, bb_indices = initialize(vertices, faces, n_buckets, cells_per_leaf)
        self.nodes = nodes
        self.bb_indices = bb_indices

    def locate_points(self, points: FloatArray) -> IntArray:
        celltree_data = CellTreeData(
            self.faces, self.vertices, self.nodes, self.bb_indices, self.cells_per_leaf,
        )
        return locate_points(points, celltree_data)

    def locate_bboxes(self, bbox_coords):
        celltree_data = CellTreeData(
            self.faces, self.vertices, self.nodes, self.bb_indices, self.cells_per_leaf,
        )
        return locate_bboxes(bbox_coords, celltree_data)
