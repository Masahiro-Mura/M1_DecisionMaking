"""
CBF-guided safety controller.

Main paper Eq. (25):

    Delta_u = argmin ||Delta_u||

    s.t.
        h0(X_{k+1}) - (1 - gamma_hat) h0(X_k) >= 0
        h1(X_{k+1}) - (1 - gamma_hat) h1(X_k) >= 0  if lcf == 1
        h2(X_{k+1}) - (1 - gamma_hat) h2(X_k) >= 0  if rcf == 1

        X_{k+1} = F(X_k, u_k + Delta_u)

and Algorithm 1 deploys

    u = u_MPC + u_CBF.

IMPORTANT SOURCE NOTE
---------------------
The main paper's Eq. (20) states

    Delta h >= -gamma_hat * h,

which algebraically corresponds to a factor (1 - gamma_hat), whereas
Eq. (25) explicitly prints (1 - gamma_hat). These two equations are not
algebraically consistent.

This implementation intentionally follows Eq. (25) literally because the
goal of this first version is reproduction of the main paper. The discrepancy
is not silently corrected here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from cbf.roadCBF import (
    RoadCBFContext,
    left_road_barrier,
    right_road_barrier,
)
from cbf.vehicleCBF import (
    SurroundingCBFState,
    predict_surrounding_state,
    vehicle_barrier,
)
from mpc.kinematicBicycleModel import discrete_step
from parameters.cbfParameters import (
    CBF_GAMMA,
    CORRECTION_ZERO_TOLERANCE,
    SOLVER_FTOL,
    SOLVER_MAX_ITERATIONS,
    SOLVER_METHOD,
)
from parameters.mpcParameters import (
    CONTROL_LOWER_BOUND,
    CONTROL_UPPER_BOUND,
)


@dataclass
class SafeControlResult:
    """Result of one CBF correction solve."""

    corrected_control: np.ndarray
    correction: np.ndarray
    success: bool
    intervention: bool
    objective_value: float
    message: str
    minimum_constraint_value: float


class SafeController:
    """Minimum-norm correction layer applied after MPC."""

    def correct(
        self,
        ego_state: np.ndarray,
        nominal_control: np.ndarray,
        surrounding_vehicles: list[SurroundingCBFState],
        road_context: RoadCBFContext | None = None,
        left_change_feasible: bool = False,
        right_change_feasible: bool = False,
    ) -> SafeControlResult:
        """
        Solve Eq. (25) for Delta_u.

        Parameters
        ----------
        ego_state:
            [x, y, v, theta].
        nominal_control:
            MPC output [a_acc, delta].
        surrounding_vehicles:
            Current surrounding states and, when available, their one-step
            predictions.
        road_context:
            Local road information for h1/h2. Required only when either road
            CBF is activated.
        left_change_feasible, right_change_feasible:
            lcf and rcf from the main paper.
        """
        ego_state = np.asarray(ego_state, dtype=np.float64)
        nominal_control = np.asarray(nominal_control, dtype=np.float64)

        if ego_state.shape != (4,):
            raise ValueError("ego_state must have shape (4,)")
        if nominal_control.shape != (2,):
            raise ValueError("nominal_control must have shape (2,)")

        if (
            (left_change_feasible or right_change_feasible)
            and road_context is None
        ):
            raise ValueError(
                "road_context is required when a road CBF is activated."
            )

        constraints = self._build_constraints(
            ego_state=ego_state,
            nominal_control=nominal_control,
            surrounding_vehicles=surrounding_vehicles,
            road_context=road_context,
            left_change_feasible=left_change_feasible,
            right_change_feasible=right_change_feasible,
        )

        correction_bounds = [
            (
                float(CONTROL_LOWER_BOUND[i] - nominal_control[i]),
                float(CONTROL_UPPER_BOUND[i] - nominal_control[i]),
            )
            for i in range(2)
        ]

        result = minimize(
            fun=self._objective,
            x0=np.zeros(2, dtype=np.float64),
            method=SOLVER_METHOD,
            bounds=correction_bounds,
            constraints=constraints,
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FTOL,
                "disp": False,
            },
        )

        correction = np.asarray(result.x, dtype=np.float64)
        corrected_control = np.clip(
            nominal_control + correction,
            CONTROL_LOWER_BOUND,
            CONTROL_UPPER_BOUND,
        )
        correction = corrected_control - nominal_control

        constraint_values = [
            float(constraint["fun"](correction))
            for constraint in constraints
        ]

        minimum_constraint_value = (
            min(constraint_values)
            if constraint_values
            else float("inf")
        )

        intervention = bool(
            np.linalg.norm(correction, ord=2)
            > CORRECTION_ZERO_TOLERANCE
        )

        return SafeControlResult(
            corrected_control=corrected_control,
            correction=correction,
            success=bool(result.success and minimum_constraint_value >= -1e-6),
            intervention=intervention,
            objective_value=float(self._objective(correction)),
            message=str(result.message),
            minimum_constraint_value=float(minimum_constraint_value),
        )

    @staticmethod
    def _objective(correction: np.ndarray) -> float:
        """
        Minimum-norm objective from Eq. (25).

        Squared Euclidean norm is used because it has the same minimizer as
        ||Delta_u|| and is smoother for numerical optimization.
        """
        correction = np.asarray(correction, dtype=np.float64)
        return float(correction @ correction)

    def _build_constraints(
        self,
        ego_state: np.ndarray,
        nominal_control: np.ndarray,
        surrounding_vehicles: list[SurroundingCBFState],
        road_context: RoadCBFContext | None,
        left_change_feasible: bool,
        right_change_feasible: bool,
    ) -> list[dict]:
        """Create SciPy inequality constraints representing Eq. (25)."""
        constraints: list[dict] = []

        # ------------------------------------------------------------
        # Vehicle CBF h0 for every surrounding vehicle.
        # ------------------------------------------------------------
        for surrounding in surrounding_vehicles:
            current_surrounding = np.asarray(
                surrounding.state,
                dtype=np.float64,
            )

            if surrounding.predicted_next_state is None:
                next_surrounding = predict_surrounding_state(
                    state=current_surrounding,
                    acceleration=surrounding.acceleration,
                )
            else:
                next_surrounding = np.asarray(
                    surrounding.predicted_next_state,
                    dtype=np.float64,
                )

            current_h0 = vehicle_barrier(
                ego_state=ego_state,
                surrounding_state=current_surrounding,
            )

            def vehicle_constraint(
                correction,
                current_h=current_h0,
                next_sv=next_surrounding.copy(),
            ):
                next_ego = discrete_step(
                    state=ego_state,
                    control=nominal_control + correction,
                )
                next_h = vehicle_barrier(
                    ego_state=next_ego,
                    surrounding_state=next_sv,
                )

                # Main paper Eq. (25), reproduced literally.
                return float(
                    next_h
                    - (1.0 - CBF_GAMMA) * current_h
                )

            constraints.append(
                {"type": "ineq", "fun": vehicle_constraint}
            )

        # ------------------------------------------------------------
        # Road CBFs h1 and h2.
        # Paper Eq. (25) activates each only for lcf==1 / rcf==1.
        #
        # The same local road context is used for X_k and X_{k+1}.
        # The paper does not specify re-linearization inside one CBF solve;
        # this is a local-road implementation assumption.
        # ------------------------------------------------------------
        if left_change_feasible and road_context is not None:
            current_h1 = left_road_barrier(ego_state, road_context)

            def left_constraint(correction, current_h=current_h1):
                next_ego = discrete_step(
                    state=ego_state,
                    control=nominal_control + correction,
                )
                next_h = left_road_barrier(next_ego, road_context)

                return float(
                    next_h
                    - (1.0 - CBF_GAMMA) * current_h
                )

            constraints.append(
                {"type": "ineq", "fun": left_constraint}
            )

        if right_change_feasible and road_context is not None:
            current_h2 = right_road_barrier(ego_state, road_context)

            def right_constraint(correction, current_h=current_h2):
                next_ego = discrete_step(
                    state=ego_state,
                    control=nominal_control + correction,
                )
                next_h = right_road_barrier(next_ego, road_context)

                return float(
                    next_h
                    - (1.0 - CBF_GAMMA) * current_h
                )

            constraints.append(
                {"type": "ineq", "fun": right_constraint}
            )

        return constraints
