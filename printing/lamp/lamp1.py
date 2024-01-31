import math
import numpy as np
from printing.stl_utils import ParametricGrid, TriangulatedObject


filename = "swirl_lamp_shorter.stl"


num_pts = 512
num_bumps = 12
bump_return = 1.1
waves_height = 1

num_twists = 0.4
base_radius = 60
left_thickness = 3 / 2
bottom_radius = 20 - left_thickness

object_height = 150


def waves_height_fn(v: float) -> float:
  return waves_height * (1 - math.cos(v / object_height * 2 * math.pi)) / 2


def bump_return_fn(v: float) -> float:
  return bump_return * (1 - math.cos(v / object_height * 2 * math.pi)) / 2


def base_radius_fn(v: float) -> float:
  h_frac = v / object_height * 0.999
  return base_radius * np.sin(h_frac * math.pi) + bottom_radius * math.exp(-10 * h_frac)


def radius_angle(u: float, v: float) -> float:
  # u from 0 to 2pi
  theta = u + bump_return_fn(v) * math.sin(num_bumps * u) / num_bumps
  y = waves_height_fn(v) * math.sin(num_bumps * u) / num_bumps
  r = (y + 1) * base_radius_fn(v)
  return r, theta + 2 * math.pi * num_twists * v / object_height


def x_center(u: float, v: float) -> float:
  r, theta = radius_angle(u, v)
  return r * math.cos(theta)


def y_center(u: float, v: float) -> float:
  r, theta = radius_angle(u, v)
  return r * math.sin(theta)


def left_unit_normal(u: float, v: float) -> float:
  forward_x = x_center(u + 0.001, v) - x_center(u, v)
  forward_y = y_center(u + 0.001, v) - y_center(u, v)
  length = math.sqrt(forward_x * forward_x + forward_y * forward_y)
  norm_x = -forward_y / length
  norm_y = forward_x / length
  return norm_x, norm_y


def x_offset(u: float, v: float, offset: float) -> float:
  xc, yc = x_center(u, v), y_center(u, v)
  if math.sqrt(xc * xc + yc * yc) < offset:
    return 0
  norm_x, norm_y = left_unit_normal(u, v)
  return x_center(u, v) + offset * norm_x


def y_offset(u: float, v: float, offset: float) -> float:
  xc, yc = x_center(u, v), y_center(u, v)
  if math.sqrt(xc * xc + yc * yc) < offset:
    return 0
  norm_x, norm_y = left_unit_normal(u, v)
  return y_center(u, v) + offset * norm_y


def get_equations(offset):
  def x(u: float, v: float) -> float:
    if v > object_height:
      return 0
    elif v < 0:
      return x_offset(u, v, 0)
    return x_offset(u, v, offset)

  def y(u: float, v: float) -> float:
    if v > object_height:
      return 0
    elif v < 0:
      return y_offset(u, v, 0)
    return y_offset(u, v, offset)

  def z(u: float, v: float) -> float:
    if v > object_height:
      return object_height
    elif v < 0:
      return 0
    return v

  return x, y, z


my_object = TriangulatedObject()

z_res = 100
u_res = 200

my_parametric = ParametricGrid(*get_equations(-left_thickness))
my_parametric.add_triangles_to(
  my_object,
  np.linspace(0.0, 2 * math.pi, u_res).tolist(),
  [-1] + np.linspace(0.0, object_height, z_res).tolist() + [object_height + 1],
)

my_parametric = ParametricGrid(*get_equations(left_thickness))
my_parametric.add_triangles_to(
  my_object,
  np.linspace(0.0, -2 * math.pi, u_res).tolist(),
  [-1] + np.linspace(0, object_height - left_thickness + 2, z_res).tolist(),
)

my_object.write_stl(filename)
