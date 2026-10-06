"""
Road geometry for the on-ramp merging scenario.

Responsibility:
    Define only the road/lane geometry and construct a HighwayEnv Road.

This implementation follows the HighwayEnv merge-road structure cited as
reference [45] by the main paper. The geometry is also consistent with the
paper's condensed case:
    - parallel merge section: x = 230 ... 310 m
    - ramp lateral region:    y = 6 ... 10 m
    - main-road region: y = 2 ... 6 m
"""

from __future__ import annotations

import numpy as np

from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork

from parameters.environmentParameters import (
    LANE_WIDTH,
    MAIN_ROAD_LANE_CENTERS,
    PARALLEL_RAMP_CENTER_Y,
    MERGE_SINE_AMPLITUDE,
    ROAD_APPROACH_LENGTH,
    ROAD_CONVERGENCE_LENGTH,
    ROAD_DOWNSTREAM_LENGTH,
    ROAD_PARALLEL_MERGE_LENGTH,
)


class MergeRoad:
    """Build the HighwayEnv road network used for the merge scenario."""

    MAIN_LANE_COUNT = len(MAIN_ROAD_LANE_CENTERS)
    RAMP_LANE_INDEX = MAIN_LANE_COUNT

    @staticmethod
    def build_network() -> RoadNetwork:
        """Create the single-lane main road and the on-ramp lane network."""
        network = RoadNetwork()

        continuous = LineType.CONTINUOUS_LINE
        striped = LineType.STRIPED
        none = LineType.NONE

        approach = ROAD_APPROACH_LENGTH
        converge = ROAD_CONVERGENCE_LENGTH
        parallel = ROAD_PARALLEL_MERGE_LENGTH
        downstream = ROAD_DOWNSTREAM_LENGTH

        x_b = approach + converge
        x_c = x_b + parallel
        x_d = x_c + downstream

        # ------------------------------------------------------------
        # Main paper Fig. 5: one straight main-road lane.
        # Keep y=4 m to match the explicit IV-B coordinates.
        # ------------------------------------------------------------
        lane_centers = MAIN_ROAD_LANE_CENTERS
        normal_line_types = ((continuous, continuous),)
        merge_line_types = ((continuous, striped),)

        for lane_id, y in enumerate(lane_centers):
            network.add_lane(
                "a",
                "b",
                StraightLane(
                    [0.0, y],
                    [x_b, y],
                    width=LANE_WIDTH,
                    line_types=normal_line_types[lane_id],
                ),
            )
            network.add_lane(
                "b",
                "c",
                StraightLane(
                    [x_b, y],
                    [x_c, y],
                    width=LANE_WIDTH,
                    line_types=merge_line_types[lane_id],
                ),
            )
            network.add_lane(
                "c",
                "d",
                StraightLane(
                    [x_c, y],
                    [x_d, y],
                    width=LANE_WIDTH,
                    line_types=normal_line_types[lane_id],
                ),
            )

        # ------------------------------------------------------------
        # On-ramp.
        #
        # HighwayEnv's merge geometry consists of:
        #   j->k : straight approach
        #   k->b : sinusoidal convergence
        #   b->c : parallel merge lane
        #
        # The parallel ramp is the second lane on edge ("b", "c"),
        # therefore its lane index is 1.
        # ------------------------------------------------------------
        amplitude = MERGE_SINE_AMPLITUDE

        ramp_approach_y = 2.0 * amplitude + PARALLEL_RAMP_CENTER_Y

        ramp_approach = StraightLane(
            [0.0, ramp_approach_y],
            [approach, ramp_approach_y],
            width=LANE_WIDTH,
            line_types=(continuous, continuous),
            forbidden=True,
        )

        ramp_convergence = SineLane(
            ramp_approach.position(approach, -amplitude),
            ramp_approach.position(x_b, -amplitude),
            amplitude,
            2.0 * np.pi / (2.0 * converge),
            np.pi / 2.0,
            width=LANE_WIDTH,
            line_types=(continuous, continuous),
            forbidden=True,
        )

        ramp_parallel = StraightLane(
            ramp_convergence.position(converge, 0.0),
            ramp_convergence.position(converge, 0.0) + np.array([parallel, 0.0]),
            width=LANE_WIDTH,
            line_types=(none, continuous),
            forbidden=True,
        )

        network.add_lane("j", "k", ramp_approach)
        network.add_lane("k", "b", ramp_convergence)
        network.add_lane("b", "c", ramp_parallel)

        return network

    @classmethod
    def build_road(
        cls,
        np_random: np.random.Generator | None = None,
        record_history: bool = False,
    ) -> Road:
        """Construct a HighwayEnv ``Road`` using the merge network."""
        if np_random is None:
            np_random = np.random.default_rng()

        return Road(
            network=cls.build_network(),
            np_random=np_random,
            record_history=record_history,
        )

    @staticmethod
    def main_lane_index_for_x(x_position: float, lane_id: int) -> tuple[str, str, int]:
        """
        Return the main-road LaneIndex containing global x.

        The main road is piecewise only because HighwayEnv uses graph nodes;
        geometrically it remains straight.
        """
        if lane_id not in range(MergeRoad.MAIN_LANE_COUNT):
            raise ValueError("lane_id must be 0 for the single-lane main road.")

        x_b = ROAD_APPROACH_LENGTH + ROAD_CONVERGENCE_LENGTH
        x_c = x_b + ROAD_PARALLEL_MERGE_LENGTH

        if x_position < x_b:
            return ("a", "b", lane_id)
        if x_position < x_c:
            return ("b", "c", lane_id)
        return ("c", "d", lane_id)

    @staticmethod
    def lane_longitudinal_from_global_x(
        x_position: float,
        lane_index: tuple[str, str, int],
    ) -> float:
        """Convert global x to longitudinal coordinate on a straight main lane."""
        edge = lane_index[:2]

        if edge == ("a", "b"):
            segment_start_x = 0.0
        elif edge == ("b", "c"):
            segment_start_x = ROAD_APPROACH_LENGTH + ROAD_CONVERGENCE_LENGTH
        elif edge == ("c", "d"):
            segment_start_x = (
                ROAD_APPROACH_LENGTH
                + ROAD_CONVERGENCE_LENGTH
                + ROAD_PARALLEL_MERGE_LENGTH
            )
        else:
            raise ValueError(f"Unsupported main-road edge: {edge}")

        return x_position - segment_start_x
