"""
Ellipse geometry used by the vehicle-to-vehicle CBF.

Main paper:
    Eq. (21) computes the distance l_i from the center of a vehicle ellipse
    to its boundary along the line joining the two vehicle centers.

The paper describes al_i and bl_i as major/minor-axis lengths. In the
parameter files we interpret them as semi-axis lengths because Eq. (21) is
the standard radial-distance formula for an ellipse using semi-axes.
This interpretation was already marked as an implementation assumption in
parameters/cbfParameters.py.
"""

from __future__ import annotations

import numpy as np


def centerline_angle(
    center_from: np.ndarray,
    center_to: np.ndarray,
) -> float:
    """Angle of the line from one ellipse center to the other [rad]."""
    p0 = np.asarray(center_from, dtype=np.float64)
    p1 = np.asarray(center_to, dtype=np.float64)

    if p0.shape != (2,) or p1.shape != (2,):
        raise ValueError("ellipse centers must both have shape (2,)")

    delta = p1 - p0

    if np.linalg.norm(delta) == 0.0:
        return 0.0

    return float(np.arctan2(delta[1], delta[0]))


def boundary_distance(
    semi_major_axis: float,
    semi_minor_axis: float,
    ellipse_heading: float,
    ray_angle: float,
) -> float:
    """
    Compute l_i from main-paper Eq. (21).

    A numerically stable equivalent of the tan(delta) expression is used:

        l = a*b / sqrt((b*cos(delta))^2 + (a*sin(delta))^2)

    where delta is the angle between the center-connecting line and the
    ellipse major axis.
    """
    a = float(semi_major_axis)
    b = float(semi_minor_axis)

    if a <= 0.0 or b <= 0.0:
        raise ValueError("ellipse semi-axes must be positive.")

    delta = float(ray_angle - ellipse_heading)

    denominator = np.sqrt(
        (b * np.cos(delta)) ** 2
        + (a * np.sin(delta)) ** 2
    )

    return float(a * b / denominator)


def pair_boundary_distances(
    ego_center: np.ndarray,
    ego_heading: float,
    ego_semi_major: float,
    ego_semi_minor: float,
    surrounding_center: np.ndarray,
    surrounding_heading: float,
    surrounding_semi_major: float,
    surrounding_semi_minor: float,
) -> tuple[float, float]:
    """
    Return (l_ev, l_sv) along the same center-connecting line.

    For the surrounding ellipse the ray points toward the ego vehicle, hence
    its ray angle differs by pi.
    """
    line_angle = centerline_angle(ego_center, surrounding_center)

    l_ego = boundary_distance(
        semi_major_axis=ego_semi_major,
        semi_minor_axis=ego_semi_minor,
        ellipse_heading=ego_heading,
        ray_angle=line_angle,
    )

    l_surrounding = boundary_distance(
        semi_major_axis=surrounding_semi_major,
        semi_minor_axis=surrounding_semi_minor,
        ellipse_heading=surrounding_heading,
        ray_angle=line_angle + np.pi,
    )

    return l_ego, l_surrounding
