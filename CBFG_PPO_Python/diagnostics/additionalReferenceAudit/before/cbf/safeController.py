"""
CBF-guided safety controller.

Main paper Eq. (25) structure, with the Eq. (20) decay coefficient:

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
the supplied PDF Eq. (25) explicitly prints (1 + gamma_hat). These two equations are not
algebraically consistent.

IMPLEMENTATION ASSUMPTION: use (1 - gamma_hat), consistent with Eq. (20)
and the nonnegative invariant safe set, rather than the inconsistent printed
factor in Eq. (25). The objective and constraint structure follow Eq. (25).
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

        def minimum_value(correction):
            values = [float(c['fun'](correction)) for c in constraints]
            return min(values) if values and np.all(np.isfinite(values)) else (
                float('inf') if not values else float('-inf'))

        nominal_in_bounds = np.all(nominal_control >= CONTROL_LOWER_BOUND) and np.all(nominal_control <= CONTROL_UPPER_BOUND)
        if nominal_in_bounds and minimum_value(np.zeros(2)) >= 0.:
            return SafeControlResult(nominal_control.copy(), np.zeros(2), True,
                                     False, 0., 'Nominal control is feasible.',
                                     minimum_value(np.zeros(2)))

        # Present the same scalar inequalities as one vector callback. This
        # avoids a separate SciPy numerical-derivative setup for every vehicle;
        # no inequality, safe set, or optimizer tolerance is changed.
        solver_constraints = [{"type": "ineq", "fun": lambda correction:
            np.asarray([constraint["fun"](correction) for constraint in constraints])}]

        result = minimize(
            fun=self._objective,
            jac=lambda correction: 2. * np.asarray(correction),
            x0=np.zeros(2, dtype=np.float64),
            method=SOLVER_METHOD,
            bounds=correction_bounds,
            constraints=solver_constraints,
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FTOL,
                "disp": False,
            },
        )

        correction = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(correction)):
            correction = np.clip(np.zeros(2), *np.asarray(correction_bounds).T)
        corrected_control = np.clip(
            nominal_control + correction,
            CONTROL_LOWER_BOUND,
            CONTROL_UPPER_BOUND,
        )
        correction = corrected_control - nominal_control
        message = str(result.message)

        if minimum_value(correction) < -1e-6:
            # IMPLEMENTATION ASSUMPTION: bounded feasibility recovery when
            # SLSQP cannot find a feasible start. The final constraints are
            # checked independently; no penalty slack relaxes the safe set.
            feasible = []
            for acceleration in (CONTROL_LOWER_BOUND[0], 0., CONTROL_UPPER_BOUND[0]):
                for steering in np.linspace(CONTROL_LOWER_BOUND[1], CONTROL_UPPER_BOUND[1], 17):
                    candidate = np.array([acceleration, steering]) - nominal_control
                    if minimum_value(candidate) >= -1e-6:
                        feasible.append(candidate)
            if feasible:
                start = min(feasible, key=self._objective)
                recovered = minimize(self._objective, start, method=SOLVER_METHOD,
                                     jac=lambda correction: 2. * np.asarray(correction),
                                     bounds=correction_bounds, constraints=solver_constraints,
                                     options={'maxiter': SOLVER_MAX_ITERATIONS, 'ftol': SOLVER_FTOL})
                candidate = np.clip(np.asarray(recovered.x),
                                    np.asarray(correction_bounds)[:, 0],
                                    np.asarray(correction_bounds)[:, 1])
                correction = candidate if minimum_value(candidate) >= -1e-6 and self._objective(candidate) <= self._objective(start) else start
                corrected_control = nominal_control + correction
                message += ' Feasible bounded recovery verified.'

        minimum_constraint_value = minimum_value(correction)

        intervention = bool(
            np.linalg.norm(correction, ord=2)
            > CORRECTION_ZERO_TOLERANCE
        )

        return SafeControlResult(
            corrected_control=corrected_control,
            correction=correction,
            # Feasibility, not an optimizer status flag, decides deployment.
            success=bool(np.all(np.isfinite(corrected_control)) and minimum_constraint_value >= -1e-6),
            intervention=intervention,
            objective_value=float(self._objective(correction)),
            message=message,
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

                # Eq. (20); see the printed Eq. (25) discrepancy above.
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
        physical_edges = road_context is not None and road_context.physical_barriers is not None
        # IMPLEMENTATION ASSUMPTION: at a shrinking road corridor, a negative
        # current look-ahead barrier must not permit an outside next state.
        # Demand h_next >= 0 in addition to the Eq. (20) decay constraint.
        if (left_change_feasible or physical_edges) and road_context is not None:
            current_h1 = left_road_barrier(ego_state, road_context)

            def left_constraint(correction, current_h=current_h1):
                next_ego = discrete_step(
                    state=ego_state,
                    control=nominal_control + correction,
                )
                next_h = left_road_barrier(next_ego, road_context)

                return float(
                    next_h
                    - (max(0., (1.0 - CBF_GAMMA) * current_h) if physical_edges
                       else (1.0 - CBF_GAMMA) * current_h)
                )

            constraints.append(
                {"type": "ineq", "fun": left_constraint}
            )

        if (right_change_feasible or physical_edges) and road_context is not None:
            current_h2 = right_road_barrier(ego_state, road_context)

            def right_constraint(correction, current_h=current_h2):
                next_ego = discrete_step(
                    state=ego_state,
                    control=nominal_control + correction,
                )
                next_h = right_road_barrier(next_ego, road_context)

                return float(
                    next_h
                    - (max(0., (1.0 - CBF_GAMMA) * current_h) if physical_edges
                       else (1.0 - CBF_GAMMA) * current_h)
                )

            constraints.append(
                {"type": "ineq", "fun": right_constraint}
            )

        return constraints
