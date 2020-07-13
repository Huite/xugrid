"""
Sutherland-Hodgman clipping 
---------------------------

Vertices (always lower case, single letter):
Clipping polygon with vertices r, s, ...
Subject polgyon with vertices a, b, ...  

Vectors (always upper case, single letter):
* U: r -> s
* N: norm, orthogonal to u
* V: a -> b
* W: a -> r
   
   s ----- ...
   |
   |  b ----- ...
   | / 
   |/   
   x     
  /|      
 / |       
a--+-- ...
   |
   r ----- ...

Floating point rounding should not be an issue, since we're only looking at
finding the area of overlap of two convex polygons.

In case of intersection failure, we can ignore it when going out -> in. It will
occur when the outgoing point is very close the clipping edge. In that case the
intersection point ~= vertex b, and we can safely skip the intersection. 

When going in -> out, b might be located on the edge. If intersection fails,
again the intersection point ~= vertex b. We treat b as if it is just on the
inside and append it. For consistency, we set b_inside to True, as it will be
used as a_inside in the next iteration.
"""
import os
from typing import NamedTuple, Sequence, Tuple

import numba as nb
import numpy as np

from xugrid.geometry.common import (
    FLOAT_TYPE,
    PARALLEL,
    FloatArray,
    Point,
    Vector,
    cross_product,
    dot_product,
    stack_empty,
)


@nb.njit(inline="always")
def _push(array: np.ndarray, n: int, value: Vector) -> int:
    array[n] = value
    return n + 1


@nb.njit(inline="always")
def _copy(src, dst, n) -> None:
    for i in range(n):
        dst[i] = src[i]


@nb.njit(inline="always")
def _inside(p: Point, r: Point, U: Vector):
    # U: a -> b direction vector
    # p is point r or s
    return U.x * (p.y - r.y) > U.y * (p.x - r.x)


@nb.njit(inline="always")
def _intersection(a: Point, V: Vector, r: Point, N: Vector) -> Tuple[bool, Point]:
    W = Vector(r.x - a.x, r.y - a.y)
    nw = dot_product(N, W)
    nv = dot_product(N, V)
    if nv != 0:
        t = nw / nv
        return True, Point(a.x + t * V.x, a.y + t * V.y)
    else:
        return False, Point(0.0, 0.0)


@nb.njit(inline="always")
def _polygon_area(polygon: Sequence, length: Sequence) -> float:
    area = 0.0
    a = Point(polygon[0][0], polygon[0][1])
    b = Point(polygon[1][0], polygon[1][1])
    U = Vector(b.x - a.x, b.y - a.y)
    for i in range(2, length):
        c = Point(polygon[i][0], polygon[i][1])
        V = Vector(a.x - c.x, a.y - c.y)
        area += abs(cross_product(U, V))
        b = c
        U = V
    return 0.5 * area


def make_allocate(nvertex, ndim):
    size = nvertex * ndim

    jit_disabled = os.environ.get("NUMBA_DISABLE_JIT", "0") != "0"
    if jit_disabled:

        def allocate_empty():
            return np.empty((nvertex, ndim), dtype=FLOAT_TYPE)

    else:

        @nb.njit(inline="always")
        def allocate_empty():
            arr_ptr = stack_empty(  # pylint: disable=no-value-for-parameter
                size, np.float64
            )
            arr = nb.carray(arr_ptr, (nvertex, ndim), dtype=FLOAT_TYPE)
            return arr

    return allocate_empty


def make_clip_polygons(nvertex, ndim):
    allocate_empty = make_allocate(nvertex, ndim)

    @nb.njit(inline="always")
    def _clip_polygons(polygon: Sequence, clipper: Sequence) -> float:
        n_output = len(polygon)
        n_clip = len(clipper)

        subject = allocate_empty()  # pylint: disable=undefined-variable
        output = allocate_empty()  # pylint: disable=undefined-variable

        # Copy polygon into output
        _copy(polygon, output, n_output)

        # Grab last point
        r = Point(clipper[n_clip - 1][0], clipper[n_clip - 1][1])
        for i in range(n_clip):
            s = Point(clipper[i][0], clipper[i][1])

            U = Vector(s.x - r.x, s.y - r.y)
            N = Vector(-U.y, U.x)
            if U.x == 0 and U.y == 0:
                continue

            # Copy output into subject
            length = n_output
            _copy(output, subject, length)
            # Reset
            n_output = 0
            # Grab last point
            a = Point(subject[length - 1][0], subject[length - 1][1])
            a_inside = _inside(a, r, U)
            for j in range(length):
                b = Point(subject[j][0], subject[j][1])

                V = Vector(b.x - a.x, b.y - a.y)
                if V.x == 0 and V.y == 0:
                    continue

                b_inside = _inside(b, r, U)
                if b_inside:
                    if not a_inside:  # out, or on the edge
                        succes, point = _intersection(a, V, r, N)
                        if succes:
                            n_output = _push(output, n_output, point)
                    n_output = _push(output, n_output, b)
                elif a_inside:
                    succes, point = _intersection(a, V, r, N)
                    if succes:
                        n_output = _push(output, n_output, point)
                    else:  # Floating point failure
                        b_inside = True  # flip it for consistency, will be set as a
                        n_output = _push(output, n_output, b)  # push b instead

                # Advance to next polygon edge
                a = b
                a_inside = b_inside

            # Exit early in case not enough vertices are left.
            if n_output < 3:
                return 0.0

            # Advance to next clipping edge
            r = s

        area = _polygon_area(output, n_output)
        return area

    return _clip_polygons
