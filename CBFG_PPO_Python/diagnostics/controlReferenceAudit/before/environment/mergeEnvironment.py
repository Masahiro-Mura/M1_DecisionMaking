"""
Gymnasium environment integrating scenario, IDM, MPC, and CBF.

This file intentionally contains orchestration only. PPO equations, MPC
equations, CBF equations, and IDM equations remain in their dedicated modules.

Per-step flow:
    normalized RL action
        -> physical [v_ref, theta_ref]
        -> MPC [a, delta]
        -> CBF minimum correction
        -> safe low-level control
        -> HighwayEnv vehicle dynamics
        -> reward / termination / next observation
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.graphics import VehicleGraphics

from cbf.roadCBF import RoadCBFContext
from cbf.safeController import SafeController
from cbf.vehicleCBF import SurroundingCBFState
from environment.actionAdapter import action_to_reference
from environment.observation import (
    build_observation,
    lane_change_feasibility,
    lateral_deviation,
)
from environment.reward import compute_reward
from environment.termination import evaluate_termination, check_merge_completed, TerminationResult
from mpc.mpcController import MPCController
from parameters.cbfParameters import INTERVENTION_THRESHOLD
from parameters.environmentParameters import (
    ACTION_DIM,
    ACTION_NORMALIZED_MAX,
    ACTION_NORMALIZED_MIN,
    DEFAULT_EGO_INITIAL_SPEED,
    OBSERVATION_DIM,
    OBSERVATION_NORMALIZED_MAX,
    OBSERVATION_NORMALIZED_MIN,
    SIMULATION_DT,
)
from scenario.mergeScenario import MergeScenario
from surrounding_vehicle_model.surroundingVehicle import SurroundingVehicle


class MergeEnvironment(gym.Env):
    """
    CBFG-PPO on-ramp merging environment.

    The environment's public action is the paper's normalized
    [v_ref, theta_ref], not acceleration/steering.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        ego_initial_speed: float = DEFAULT_EGO_INITIAL_SPEED,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(
                "render_mode must be None, 'human', or 'rgb_array'"
            )

        self.render_mode = render_mode
        self.viewer = None

        # HighwayEnv's native EnvViewer expects the standard rendering keys.
        # These values affect visualization only, not vehicle dynamics or RL.
        self.config = {
            "offscreen_rendering": render_mode != "human",
            "real_time_rendering": False,
            "screen_width": 1000,
            "screen_height": 400,
            "scaling": 5.5,
            "centering_position": [0.30, 0.50],
            "render_agent": False,
            "simulation_frequency": int(round(1.0 / SIMULATION_DT)),
            "policy_frequency": int(round(1.0 / SIMULATION_DT)),
            "show_trajectories": False,
        }

        # EnvViewer accesses this attribute when drawing observation overlays.
        # Our custom observation is rendered only through road/vehicle graphics.
        self.observation_type = None
        self.action_type = None

        self.ego_initial_speed = float(ego_initial_speed)
        # Main Paper III-D: these are minimum speeds maintained during driving.
        # Initial speed is set to the same value for reproducibility.
        self.ego_minimum_speed = float(ego_initial_speed)

        self.action_space = spaces.Box(
            low=ACTION_NORMALIZED_MIN,
            high=ACTION_NORMALIZED_MAX,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=OBSERVATION_NORMALIZED_MIN,
            high=OBSERVATION_NORMALIZED_MAX,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self.mpc = MPCController()
        self.safe_controller = SafeController()

        self.road = None
        self.vehicle = None
        self.surrounding_vehicles = []

        self.step_count: int = 0
        self._stable_main_lane_steps = 0
        self._previous_reference_heading: float | None = None
        self._previous_ego_position: np.ndarray | None = None
        self._previous_lateral_deviation: float = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        scenario = MergeScenario(
            seed=seed,
            ego_initial_speed=self.ego_initial_speed,
            surrounding_vehicle_cls=SurroundingVehicle,
        )
        state = scenario.create()

        self.road = state.road
        self.vehicle = state.ego_vehicle
        self.surrounding_vehicles = state.surrounding_vehicles

        # Match HighwayEnv's standard convention: controlled ego is green.
        self.vehicle.color = VehicleGraphics.EGO_COLOR

        self.step_count = 0
        self._stable_main_lane_steps = 0
        self._previous_reference_heading = None
        self._previous_ego_position = np.asarray(
            self.vehicle.position,
            dtype=np.float64,
        ).copy()
        self._previous_lateral_deviation = lateral_deviation(
            self.road,
            self.vehicle,
        )

        self.mpc.reset()

        observation = build_observation(
            road=self.road,
            ego_vehicle=self.vehicle,
            surrounding_vehicles=self.surrounding_vehicles,
        )

        info = {
            "success": False,
            "termination_reason": None,
            "step_count": 0,
        }

        return observation, info

    def step(self, action: np.ndarray):
        self._assert_initialized()

        # ------------------------------------------------------------
        # 1) RL normalized action -> physical MPC reference.
        # ------------------------------------------------------------
        reference = action_to_reference(action, minimum_speed=self.ego_minimum_speed)

        ego_state = self._ego_state()

        # ------------------------------------------------------------
        # 2) Surrounding vehicles compute their IDM actions at X_k.
        # ------------------------------------------------------------
        for vehicle in self.surrounding_vehicles:
            vehicle.act()

        # ------------------------------------------------------------
        # 3) Motion-control layer: MPC.
        # ------------------------------------------------------------
        mpc_result = self.mpc.solve(
            current_state=ego_state,
            reference_velocity=reference.velocity,
            reference_heading=reference.heading,
            minimum_speed=self.ego_minimum_speed,
        )

        # ------------------------------------------------------------
        # 4) CBF-guided layer.
        # ------------------------------------------------------------
        left_feasible, right_feasible = lane_change_feasibility(
            self.road,
            self.vehicle,
        )

        safe_result = self.safe_controller.correct(
            ego_state=ego_state,
            nominal_control=mpc_result.control,
            surrounding_vehicles=self._cbf_surrounding_states(),
            road_context=self._road_cbf_context(),
            left_change_feasible=bool(left_feasible),
            right_change_feasible=bool(right_feasible),
            minimum_speed=self.ego_minimum_speed,
        )

        # ------------------------------------------------------------
        # 5) Apply safe control and advance HighwayEnv dynamics.
        # ------------------------------------------------------------
        control_applied = bool(mpc_result.success and safe_result.success)
        if control_applied:
            self.vehicle.act(
                {
                    "acceleration": float(safe_result.corrected_control[0]),
                    "steering": float(safe_result.corrected_control[1]),
                }
            )
            self.road.step(SIMULATION_DT)
        self.step_count += 1
        self._stable_main_lane_steps = (self._stable_main_lane_steps + 1
            if control_applied and check_merge_completed(self.road, self.vehicle) else 0)

        # ------------------------------------------------------------
        # 6) Termination.
        # ------------------------------------------------------------
        termination = evaluate_termination(
            road=self.road,
            ego_vehicle=self.vehicle,
            step_count=self.step_count,
            ego_initial_speed=self.ego_initial_speed,
            stable_main_lane_steps=self._stable_main_lane_steps,
        ) if control_applied else TerminationResult(
            # IMPLEMENTATION ASSUMPTION: infeasible/failed solver output is
            # rejected without advancing the world; an absorbing failure is
            # returned instead of claiming a safe control was deployed.
            terminated=True, truncated=False, success=False, collision=False,
            road_departure=False, time_limit=False, control_failure=True,
            reason='cbf_infeasible' if mpc_result.success else 'mpc_failed')

        # ------------------------------------------------------------
        # 7) Reward, exactly decomposed into Eq. (2)-(7).
        # ------------------------------------------------------------
        current_lateral_deviation = lateral_deviation(
            self.road,
            self.vehicle,
        )
        distance_front, distance_rear = self._front_rear_distances()

        reward_breakdown = compute_reward(
            lateral_deviation=current_lateral_deviation,
            distance_front=distance_front,
            distance_rear=distance_rear,
            current_reference_heading=reference.heading,
            previous_reference_heading=self._previous_reference_heading,
            current_speed=float(self.vehicle.speed),
            success=termination.success,
            failure=termination.failure,
        )

        # ------------------------------------------------------------
        # 8) Next observation.
        # ------------------------------------------------------------
        observation = build_observation(
            road=self.road,
            ego_vehicle=self.vehicle,
            surrounding_vehicles=self.surrounding_vehicles,
        )

        correction_norm = float(
            np.linalg.norm(safe_result.correction, ord=2)
        )
        cbf_intervention = bool(
            correction_norm > INTERVENTION_THRESHOLD
        )

        info = {
            "success": termination.success,
            "failure": termination.failure,
            "collision": termination.collision,
            "road_departure": termination.road_departure,
            "time_limit": termination.time_limit,
            "termination_reason": termination.reason,
            "step_count": self.step_count,
            "ego_minimum_speed": self.ego_minimum_speed,
            "reference_velocity": reference.velocity,
            "reference_heading": reference.heading,
            "mpc_control": mpc_result.control.copy(),
            "mpc_success": mpc_result.success,
            "safe_control": safe_result.corrected_control.copy(),
            # safe_control remains the legacy candidate field. A rejected
            # candidate is never an applied control or a valid BC teacher.
            "applied_control": safe_result.corrected_control.copy() if control_applied else None,
            "cbf_correction": safe_result.correction.copy(),
            "cbf_correction_norm": correction_norm,
            "cbf_intervention": cbf_intervention,
            "cbf_solver_success": safe_result.success,
            "control_applied": control_applied,
            "control_failure": termination.control_failure,
            "cbf_minimum_constraint": safe_result.minimum_constraint_value,
            # [37]'s initial-safe-set prerequisite, separately from feasibility.
            # Solver success does not prove invariance when this is false.
            "cbf_minimum_current_barrier": safe_result.minimum_current_barrier,
            "cbf_initial_safe_set_satisfied": safe_result.initial_safe_set_satisfied,
            "cbf_solver_message": safe_result.message,
            "ego_state_before": ego_state.copy(),
            "stable_main_lane_steps": self._stable_main_lane_steps,
            "reward_lane_center": reward_breakdown.lane_center,
            "reward_merge": reward_breakdown.merge,
            "reward_steering": reward_breakdown.steering,
            "reward_speed": reward_breakdown.speed,
            "reward_terminal": reward_breakdown.terminal,
        }

        # Save quantities required by the next Eq. (5) / road-CBF step.
        self._previous_reference_heading = reference.heading
        self._previous_ego_position = np.asarray(
            self.vehicle.position,
            dtype=np.float64,
        ).copy()
        self._previous_lateral_deviation = current_lateral_deviation

        return (
            observation,
            reward_breakdown.total,
            termination.terminated,
            termination.truncated,
            info,
        )

    def _ego_state(self) -> np.ndarray:
        """Return [x, y, v, theta] for MPC/CBF."""
        return np.array(
            [
                float(self.vehicle.position[0]),
                float(self.vehicle.position[1]),
                float(self.vehicle.speed),
                float(self.vehicle.heading),
            ],
            dtype=np.float64,
        )

    def _cbf_surrounding_states(self) -> list[SurroundingCBFState]:
        """Create current surrounding states for the CBF safe controller."""
        states: list[SurroundingCBFState] = []

        for vehicle in self.surrounding_vehicles:
            states.append(
                SurroundingCBFState(
                    state=np.array(
                        [
                            float(vehicle.position[0]),
                            float(vehicle.position[1]),
                            float(vehicle.speed),
                            float(vehicle.heading),
                        ],
                        dtype=np.float64,
                    ),
                    acceleration=float(
                        getattr(vehicle, "acceleration", 0.0)
                    ),
                )
            )

        return states

    def _road_cbf_context(self) -> RoadCBFContext:
        """
        Build the local Eq. (23)-(24) road context.

        IMPLEMENTATION ASSUMPTION: relinearize position, deviation and tangent
        together in the current lane frame; do not mix prior/current lane IDs.
        For the physical road constraints, join touching drivable lane intervals
        in that tangent frame, including the forbidden-but-drivable ramp. Both
        outer edges remain constrained even when lcf/rcf change at a lane switch.
        """
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        longitudinal, deviation = lane.local_coordinates(self.vehicle.position)
        lane_heading = float(lane.heading_at(longitudinal))

        prior = np.asarray(self.vehicle.position, dtype=np.float64).copy()
        normal = np.array([-np.sin(lane_heading), np.cos(lane_heading)])
        lanes = tuple(self.road.network.lanes_list())

        def connected_bounds(position):
            intervals = []
            for candidate in lanes:
                distance, _ = candidate.local_coordinates(position)
                if -1e-8 <= distance <= candidate.length + 1e-8:
                    center = candidate.position(distance, 0.)
                    offset = float((center - prior) @ normal)
                    half_width = float(candidate.width_at(distance)) / 2.
                    intervals.append((offset - half_width, offset + half_width))
            if not intervals:
                return 1., -1.
            connected = []
            for lower, upper in sorted(intervals):
                if connected and lower <= connected[-1][1] + 1e-8:
                    connected[-1] = (connected[-1][0], max(upper, connected[-1][1]))
                else:
                    connected.append((lower, upper))
            # The connected component nearest the fixed current ego anchor
            # stays selected within this solve; gaps are never declared road.
            lower, upper = min(connected, key=lambda bounds: max(bounds[0], -bounds[1], 0.))
            return lower, upper

        # IMPLEMENTATION ASSUMPTION: on this forward highway, the largest
        # reachable x displacement in one Euler step is v*dt. Use the corridor
        # intersection over that step so the disappearing ramp does not cause
        # an artificial CBF jump for an ego already driving on the main road.
        future = prior + np.array([max(0., self.vehicle.speed) * SIMULATION_DT, 0.])
        lower_now, upper_now = connected_bounds(prior)
        lower_future, upper_future = connected_bounds(future)
        corridor_lower = max(lower_now, lower_future)
        corridor_upper = min(upper_now, upper_future)

        def physical_barriers(state):
            lower_actual, upper_actual = connected_bounds(state[:2])
            lower = max(corridor_lower, lower_actual)
            upper = min(corridor_upper, upper_actual)
            lateral = float((state[:2] - prior) @ normal)
            return upper - lateral, lateral - lower

        return RoadCBFContext(
            prior_x=float(prior[0]),
            prior_y=float(prior[1]),
            prior_lateral_deviation=float(deviation),
            lane_slope=float(np.tan(lane_heading)),
            lane_width=float(lane.width_at(longitudinal)),
            physical_barriers=physical_barriers,
        )

    def _front_rear_distances(
        self,
    ) -> tuple[float | None, float | None]:
        """
        Find longitudinal front/rear distances used by Eq. (4).

        Source gap:
        the paper defines dP and dF but does not specify a HighwayEnv query
        procedure. In the parallel merge section, road lanes are aligned with
        global x, so v1 uses global longitudinal x differences.
        """
        ego_x = float(self.vehicle.position[0])

        front_distances = []
        rear_distances = []

        for vehicle in self.surrounding_vehicles:
            dx = float(vehicle.position[0] - ego_x)

            if dx >= 0.0:
                front_distances.append(dx)
            else:
                rear_distances.append(abs(dx))

        distance_front = (
            min(front_distances) if front_distances else None
        )
        distance_rear = (
            min(rear_distances) if rear_distances else None
        )

        return distance_front, distance_rear

    def _assert_initialized(self) -> None:
        if self.road is None or self.vehicle is None:
            raise RuntimeError(
                "Environment is not initialized. Call reset() before step()."
            )

    def render(self):
        """
        Render with HighwayEnv's native pygame-based EnvViewer.

        - render_mode="rgb_array": return H x W x 3 uint8 frame
        - render_mode="human": show HighwayEnv window
        - render_mode=None: no rendering

        This uses HighwayEnv RoadGraphics and VehicleGraphics directly rather
        than recreating the scene with Matplotlib.
        """
        if self.render_mode is None:
            return None

        if self.road is None or self.vehicle is None:
            raise RuntimeError(
                "Environment is not initialized. Call reset() before render()."
            )

        if self.viewer is None:
            self.viewer = EnvViewer(self, config=self.config)

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()

        if self.render_mode == "rgb_array":
            return self.viewer.get_image()

        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
