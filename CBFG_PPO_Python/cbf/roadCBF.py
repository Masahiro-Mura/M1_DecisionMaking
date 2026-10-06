"""
Road-boundary control barrier functions.

Main paper:
    Eq. (23), Eq. (24) define two road-boundary CBFs h1 and h2.

However, the literal implementation of Eq. (23)-(24) combined with the
current HighwayEnv RoadCBFContext produced:

    lane center:
        h1 = -W_l / 2
        h2 = +W_l / 2

This is inconsistent with the CBF safe-set definition

    C = {X : h(X) >= 0},

because the ego vehicle was classified as unsafe even while exactly at
the lane center.

Diagnostic testing also showed that this caused the CBF controller to
continuously generate positive steering, pushing the ego vehicle away
from the main road.

Therefore, this implementation preserves the intention of the paper
(road-boundary safety using h >= 0), while expressing the barriers in
HighwayEnv-compatible local lane coordinates.

For a signed lateral deviation d from the local lane center:

    h1 = W_l / 2 - d
    h2 = W_l / 2 + d

Thus:

    lane center:
        h1 = h2 = W_l / 2 > 0

    corresponding boundary:
        h = 0

    outside the road boundary:
        h < 0

The local lateral deviation at a predicted state is approximated using
the tangent of the lane at the previous vehicle position.

IMPLEMENTATION ASSUMPTION: MergeEnvironment supplies a connected-road
physical_barriers evaluator for the parallel merge/downstream geometry. In
that mode h1/h2 measure the physical outer edges, not the current lane edges;
both stay active across lane-ID changes. Local-only callers retain W/2 +/- d.

NOTE
----
The function names left_road_barrier / right_road_barrier are retained
for compatibility with safeController.py and the paper's h1/h2
activation structure.

They should be interpreted as h1 / h2 rather than as literal global
left/right geometric boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from parameters.cbfParameters import ROAD_LANE_WIDTH


@dataclass(frozen=True)
class RoadCBFContext:
    """
    Local lane information used to evaluate the road CBFs.

    Parameters
    ----------
    prior_x, prior_y:
        Ego position used as the local linearization point.

    prior_lateral_deviation:
        Signed HighwayEnv lateral deviation from the lane center
        at the linearization point.

    lane_slope:
        Slope of the local lane tangent:
            m = tan(lane_heading)

    lane_width:
        Width of the current lane.
    """

    prior_x: float
    prior_y: float
    prior_lateral_deviation: float
    lane_slope: float
    lane_width: float = ROAD_LANE_WIDTH
    # IMPLEMENTATION ASSUMPTION: optional connected-road envelope in the
    # simulator coordinates. It prevents lane-center switches from redefining
    # the safe set and constrains both physical outer edges independently of
    # lane-change permission. The paper's printed local equations are ambiguous.
    physical_barriers: Callable[[np.ndarray], tuple[float, float]] | None = None


def _validate_state(
    ego_state: np.ndarray,
) -> np.ndarray:
    """Validate and return ego state [x, y, v, theta]."""

    state = np.asarray(
        ego_state,
        dtype=np.float64,
    )

    if state.shape != (4,):
        raise ValueError(
            f"ego_state must have shape (4,), got {state.shape}"
        )

    return state


def local_lateral_deviation(
    ego_state: np.ndarray,
    context: RoadCBFContext,
) -> float:
    """
    Estimate signed lateral deviation from the local lane center.

    HighwayEnv's local lane coordinate is positive along the
    local lateral normal.

    Let the lane tangent heading at the previous point be psi.

        psi = atan(m)

    The corresponding lateral unit vector is

        n = [-sin(psi), cos(psi)]

    Therefore the change in lateral position is

        Delta d =
            -sin(psi) * Delta x
            + cos(psi) * Delta y

    and

        d_k =
            d_previous + Delta d

    This is a local tangent approximation of the lane geometry.
    """

    state = _validate_state(ego_state)

    x = float(state[0])
    y = float(state[1])

    dx = x - context.prior_x
    dy = y - context.prior_y

    lane_heading = float(
        np.arctan(context.lane_slope)
    )

    lateral_change = (
        -np.sin(lane_heading) * dx
        + np.cos(lane_heading) * dy
    )

    return float(
        context.prior_lateral_deviation
        + lateral_change
    )


def left_road_barrier(
    ego_state: np.ndarray,
    context: RoadCBFContext,
) -> float:
    """
    Compute road CBF h1.

    h1 = W_l / 2 - d

    where d is the signed local lateral deviation.

    Properties
    ----------
    At lane center:
        h1 = W_l / 2 > 0

    At the +lateral boundary:
        h1 = 0

    Outside that boundary:
        h1 < 0
    """

    lateral_deviation = local_lateral_deviation(
        ego_state,
        context,
    )

    if context.physical_barriers is not None:
        return float(context.physical_barriers(_validate_state(ego_state))[0])
    return float(
        context.lane_width / 2.0
        - lateral_deviation
    )


def right_road_barrier(
    ego_state: np.ndarray,
    context: RoadCBFContext,
) -> float:
    """
    Compute road CBF h2.

    h2 = W_l / 2 + d

    where d is the signed local lateral deviation.

    Properties
    ----------
    At lane center:
        h2 = W_l / 2 > 0

    At the -lateral boundary:
        h2 = 0

    Outside that boundary:
        h2 < 0
    """

    lateral_deviation = local_lateral_deviation(
        ego_state,
        context,
    )

    if context.physical_barriers is not None:
        return float(context.physical_barriers(_validate_state(ego_state))[1])
    return float(
        context.lane_width / 2.0
        + lateral_deviation
    )
