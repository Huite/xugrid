from typing import NamedTuple, Sequence, Tuple

import numba as nb
import numpy as np
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

IntArray = np.ndarray
FloatArray = np.ndarray
INT_TYPE = np.int64
INT_TYPE_T = nb.from_dtype(INT_TYPE)
FLOAT_TYPE = np.float64
FLOAT_TYPE_T = nb.from_dtype(FLOAT_TYPE)
FLOAT_MIN = np.finfo(FLOAT_TYPE).min
FLOAT_MAX = np.finfo(FLOAT_TYPE).max
INT_MAX = np.iinfo(INT_TYPE).max
PARALLEL = True


class Point(NamedTuple):
    x: float
    y: float


class Vector(NamedTuple):
    x: float
    y: float


@intrinsic
def stack_empty(typingctx, size, dtype):
    def impl(context, builder, signature, args):
        ty = context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty, size=args[0])
        return ptr

    sig = types.CPointer(dtype.dtype)(types.int64, dtype)
    return sig, impl


@nb.njit(inline="always")
def cross_product(u: Vector, v: Vector) -> float:
    return u.x * v.y - u.y * v.x


@nb.njit(inline="always")
def dot_product(u: Vector, v: Vector) -> float:
    return u.x * v.x + u.y * v.y


@nb.njit(inline="always")
def polygon_area(polygon):
    length = len(polygon)
    area = 0.0
    a = polygon[0]
    b = polygon[1]
    u = Point(b.x - a.x, b.y - a.y)
    for i in range(2, length):
        c = polygon[i]
        v = Point(a.x - c.x, a.y - c.y)
        area += abs(cross_product(u, v))
        b = c
        u = v
    return 0.5 * area


@nb.njit(inline="always")
def point_norm(p: Point, v0: Vector, v1: Vector) -> Vector:
    # Use in case the polygon in not guaranteed counter-clockwise.
    n = Vector(-(v1.y - v0.y), (v1.x - v0.x))
    v = Vector(v0.x - p.x, v0.y - p.y)
    dot = dot_product(n, v)
    if dot == 0:
        raise ValueError
    elif dot < 0:
        n = Vector(-n.x, -n.y)
    return n


@nb.njit(inline="always")
def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    # Refer to: https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    # Copyright (c) 1970-2003, Wm. Randolph Franklin
    # MIT license.
    #
    # Quote:
    # > I run a semi-infinite ray horizontally (increasing x, fixed y) out from
    # > the test point, and count how many edges it crosses. At each crossing,
    # > the ray switches between inside and outside. This is called the Jordan
    # > curve theorem.
    # >
    # > The case of the ray going thru a vertex is handled correctly via a
    # > careful selection of inequalities. Don't mess with this code unless
    # > you're familiar with the idea of Simulation of Simplicity. This pretends
    # > to shift the ray infinitesimally down so that it either clearly
    # > intersects, or clearly doesn't touch. Since this is merely a conceptual,
    # > infinitesimal, shift, it never creates an intersection that didn't exist
    # > before, and never destroys an intersection that clearly existed before.
    # >
    # > The ray is tested against each edge thus:
    # > 1. Is the point in the half-plane to the left of the extended edge? and
    # > 2. Is the point's Y coordinate within the edge's Y-range?
    # >
    # > Handling endpoints here is tricky.
    #
    # For the Simulation of Simplicity concept, see:
    # Edelsbrunner, H., & Mücke, E. P. (1990). Simulation of simplicity: a
    # technique to cope with degenerate cases in geometric algorithms. ACM
    # Transactions on Graphics (tog), 9(1), 66-104.
    #
    # In this case, this guarantees there will be no "on-edge" answers, which
    # are degenerative. For another application of simulation of simplicity,
    # see:
    # Rappoport, A. (1991). An efficient algorithm for line and polygon
    # clipping. The Visual Computer, 7(1), 19-28.
    length = len(poly)
    c = False
    for i in range(length):
        v0 = poly[i]
        v1 = poly[(i + 1) % length]
        # Do not split this in two conditionals: if the first conditional fails,
        # the second will not be executed in Python's (and C's) execution model.
        # This matters because the second can result in division by zero.
        if (v0.y > p.y) != (v1.y > p.y) and p.x < (
            (v1.x - v0.x) * (p.y - v0.y) / (v1.y - v0.y) + v0.x
        ):
            c = not c
    return c
