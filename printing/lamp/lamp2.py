import math
import numpy as np
from printing.stl_utils import ParametricGrid, TriangulatedObject


filename = "swirl_lamp_taller.stl"


num_bumps = 12
bump_return = 1.1
waves_height = 1.5

num_twists = 0.7
num_flame_waves = 3
flame_wave_size = 0.0
base_radius = 50
left_thickness = 1.0
bottom_radius = 20 - left_thickness

thickness_wiggle = 0.0
num_thickness_wiggles = 80

object_height = 210

z_res = 120
u_res = num_bumps * 30


def waves_height_fn(v: float) -> float:
  return waves_height * (1 - math.cos(v / object_height * 2 * math.pi)) / 2


def bump_return_fn(v: float) -> float:
  return bump_return * (1 - math.cos(v / object_height * 2 * math.pi)) / 2


def base_radius_fn(v: float) -> float:
  h_frac = v / object_height * 0.999
  return base_radius * np.sin(h_frac * math.pi) + bottom_radius * math.exp(-8.5 * h_frac)


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


def get_equations(offset, wiggle):
  def x(u: float, v: float) -> float:
    if v > object_height:
      ret_val = 0
    elif v < 0:
      ret_val = x_offset(u, v, 0)
    else:
      ret_val = x_offset(
        u,
        v,
        offset * (1 + wiggle * math.cos(math.pi * v / object_height * num_thickness_wiggles) ** 2),
      )
    return (
      ret_val
      + flame_wave_size
      * math.cos(v / object_height * num_flame_waves * 2 * math.pi)
      * math.sin(v / object_height * math.pi / 2) ** 2
    )

  def y(u: float, v: float) -> float:
    if v > object_height:
      ret_val = 0
    elif v < 0:
      ret_val = y_offset(u, v, 0)
    else:
      ret_val = y_offset(
        u,
        v,
        offset * (1 + wiggle * math.cos(math.pi * v / object_height * num_thickness_wiggles) ** 2),
      )
    return (
      ret_val
      + flame_wave_size
      * math.sin(v / object_height * num_flame_waves * 2 * math.pi)
      * math.sin(v / object_height * math.pi / 2) ** 2
    )

  def z(u: float, v: float) -> float:
    if v > object_height:
      ret_val = object_height
    elif v < 0:
      ret_val = 0
    else:
      ret_val = v
    return ret_val

  return x, y, z


my_object = TriangulatedObject()

my_parametric = ParametricGrid(*get_equations(-left_thickness, wiggle=0))
my_parametric.add_triangles_to(
  my_object,
  np.linspace(0.0, 2 * math.pi, u_res).tolist(),
  [-1] + np.linspace(0.0, object_height, z_res).tolist() + [object_height + 1],
)

my_parametric = ParametricGrid(*get_equations(left_thickness, wiggle=thickness_wiggle))
my_parametric.add_triangles_to(
  my_object,
  np.linspace(0.0, -2 * math.pi, u_res).tolist(),
  [-1] + np.linspace(0, object_height - left_thickness + 2, z_res).tolist(),
)

my_object.write_stl(filename)
