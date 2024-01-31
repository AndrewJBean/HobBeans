import struct
import math
from typing import Callable, List, Iterator


def normalize(vec: List[float]) -> List[float]:
  total = sum(v * v for v in vec)
  if total > 0.0:
    return [v / math.sqrt(total) for v in vec]
  return [1.0] + [0.0 for _ in vec[1:]]


def right_unit_normal(a: List[float], b: List[float]) -> List[float]:
  # compute vector cross product, using right hand convention
  n0 = a[1] * b[2] - a[2] * b[1]
  n1 = a[2] * b[0] - a[0] * b[2]
  n2 = a[0] * b[1] - a[1] * b[0]

  # normalize or take unit vector if zero
  n = normalize([n0, n1, n2])
  return n


def write_header(f):
  # 80 bytes / 4 = 20 ints
  # all zeros
  num_vals = 20
  vals = [0] * num_vals
  data = struct.pack("i" * num_vals, *vals)
  f.write(data)


def write_num_triangles(f, num_triangles):
  f.write(struct.pack("i", num_triangles))


def write_triangle(f, x1, x2, x3):
  a = [val2 - val1 for val2, val1 in zip(x2, x1)]
  b = [val3 - val1 for val3, val1 in zip(x3, x1)]

  n = right_unit_normal(a, b)

  vals = n + x1 + x2 + x3 + [0]
  f.write(struct.pack("f" * 12 + "h", *vals))


class TriangulatedObject:
  def __init__(self):
    self.triangles = []

  def add_triangle(self, x1, x2, x3):
    self.triangles.append([x1, x2, x3])

  def write_stl(self, filename):
    with open(filename, "wb") as f:
      write_header(f)
      write_num_triangles(f, len(self.triangles))
      for t in self.triangles:
        write_triangle(f, *t)


class ParametricGrid:
  def __init__(
    self,
    x: Callable[[float, float], float],
    y: Callable[[float, float], float],
    z: Callable[[float, float], float],
  ):

    self._x = x
    self._y = y
    self._z = z

  def triangles(
    self,
    u_vals: List[float],
    v_vals: List[float],
    left_handed=False,
  ) -> Iterator[List[List[float]]]:
    """
    generate triangles on the given grid for the parametric equations
    """

    for u_idx in range(len(u_vals) - 1):
      for v_idx in range(len(v_vals) - 1):
        # two triangles:
        # (u_idx, v_idx), (u_idx+1, v_idx), (u_idx+1, v_idx+1)
        p1 = [
          [
            self._x(u_vals[u_idx], v_vals[v_idx]),
            self._y(u_vals[u_idx], v_vals[v_idx]),
            self._z(u_vals[u_idx], v_vals[v_idx]),
          ],
          [
            self._x(u_vals[u_idx + 1], v_vals[v_idx]),
            self._y(u_vals[u_idx + 1], v_vals[v_idx]),
            self._z(u_vals[u_idx + 1], v_vals[v_idx]),
          ],
          [
            self._x(u_vals[u_idx + 1], v_vals[v_idx + 1]),
            self._y(u_vals[u_idx + 1], v_vals[v_idx + 1]),
            self._z(u_vals[u_idx + 1], v_vals[v_idx + 1]),
          ],
        ]
        # (u_idx, v_idx), (u_idx+1, v_idx+1), (u_idx, v_idx+1)
        p2 = [
          [
            self._x(u_vals[u_idx], v_vals[v_idx]),
            self._y(u_vals[u_idx], v_vals[v_idx]),
            self._z(u_vals[u_idx], v_vals[v_idx]),
          ],
          [
            self._x(u_vals[u_idx + 1], v_vals[v_idx + 1]),
            self._y(u_vals[u_idx + 1], v_vals[v_idx + 1]),
            self._z(u_vals[u_idx + 1], v_vals[v_idx + 1]),
          ],
          [
            self._x(u_vals[u_idx], v_vals[v_idx + 1]),
            self._y(u_vals[u_idx], v_vals[v_idx + 1]),
            self._z(u_vals[u_idx], v_vals[v_idx + 1]),
          ],
        ]
        if left_handed:
          yield [p1[0], p1[2], p1[1]]
          yield [p2[0], p2[2], p2[1]]
        else:
          yield p1
          yield p2

  def add_triangles_to(
    self,
    obj: TriangulatedObject,
    u_vals: List[float],
    v_vals: List[float],
    left_handed=False,
  ):
    for p in self.triangles(u_vals, v_vals, left_handed):
      obj.add_triangle(*p)
