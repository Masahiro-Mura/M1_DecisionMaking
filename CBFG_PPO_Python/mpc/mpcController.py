"""
Nonlinear MPC controller for the motion-control layer.

The controller solves the main-paper optimization problem Eq. (18):

    minimize_u J(z, u)

    subject to
        x(k+1) = f_hat(x(k), u(k))
        z(k)   = [v(k), theta(k)]^T
        u_min <= u(k) <= u_max

Main Paper III-D additionally specifies maintained minimum EV speeds.
MergeEnvironment supplies a scenario-specific minimum; this implementation
enforces it with linear constraints on every predicted speed. This enforcement
mechanism is a reproduction choice, not an extra equation printed in Eq18.

The paper does not state a numerical optimizer. The implementation uses SLSQP,
as configured in parameters/mpcParameters.py. This is an implementation
assumption and can be replaced later without changing the model/cost modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, LinearConstraint

from parameters.mpcParameters import (
    MPC_DT,
    CONTROL_LOWER_BOUND,
    CONTROL_UPPER_BOUND,
    PREDICTION_HORIZON,
    SOLVER_FTOL,
    SOLVER_MAX_ITERATIONS,
    SOLVER_METHOD,
)
from mpc.kinematicBicycleModel import rollout
from mpc.mpcCost import trajectory_cost, tracking_objective_and_gradient


@dataclass
class MPCResult:
    """Result of one receding-horizon MPC solve."""

    control: np.ndarray
    control_sequence: np.ndarray
    predicted_states: np.ndarray
    objective_value: float
    success: bool
    message: str


class MPCController:
    """
    Receding-horizon nonlinear MPC controller.

    The RL layer must provide physical reference values

        [v_ref, theta_ref]

    rather than normalized [-1, 1] actions. Action denormalization belongs to
    the environment/RL interface, not to the MPC module.
    """

    def __init__(
        self,
        prediction_horizon: int = PREDICTION_HORIZON,
    ) -> None:
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive.")

        self.prediction_horizon = int(prediction_horizon)
        self._previous_solution: np.ndarray | None = None

    def solve(
        self,
        current_state: np.ndarray,
        reference_velocity: float,
        reference_heading: float,
        minimum_speed: float | None = None,
    ) -> MPCResult:
        """
        Solve one MPC problem and return the first control action.

        Parameters
        ----------
        current_state:
            [x, y, v, theta].
        reference_velocity:
            Physical reference speed v_ref [m/s].
        reference_heading:
            Physical reference heading theta_ref [rad].
        """
        current_state = np.asarray(current_state, dtype=np.float64)

        if current_state.shape != (4,):
            raise ValueError(
                f"current_state must have shape (4,), got {current_state.shape}"
            )

        reference_output = np.array(
            [reference_velocity, reference_heading],
            dtype=np.float64,
        )

        initial_guess = self._initial_guess()
        bounds = self._optimization_bounds()

        constraints = []
        if minimum_speed is not None:
            if not np.isfinite(minimum_speed) or minimum_speed < 0.:
                raise ValueError("minimum_speed must be finite and nonnegative.")
            # Main Paper III-D specifies minimum EV speeds, not just initial
            # speeds. Under Eq16, v_j = v_0 + dt * sum_{i<j} acceleration_i.
            # Enforcement choice: linear state constraints at every horizon
            # step, rather than clipping speed after the predicted dynamics.
            matrix = np.zeros((self.prediction_horizon, 2*self.prediction_horizon))
            matrix[:, ::2] = MPC_DT * np.tril(np.ones((self.prediction_horizon, self.prediction_horizon)))
            constraints = [LinearConstraint(matrix, minimum_speed-current_state[2], np.inf)]

        result = minimize(
            fun=self._objective_and_gradient,
            jac=True,
            x0=initial_guess.reshape(-1),
            args=(current_state, reference_output),
            method=SOLVER_METHOD,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": SOLVER_MAX_ITERATIONS,
                "ftol": SOLVER_FTOL,
                "disp": False,
            },
        )

        optimized_controls = np.asarray(
            result.x,
            dtype=np.float64,
        ).reshape(self.prediction_horizon, 2)

        # Numerical optimization may return values infinitesimally outside the
        # declared bounds. Clip before deployment.
        optimized_controls = np.clip(
            optimized_controls,
            CONTROL_LOWER_BOUND,
            CONTROL_UPPER_BOUND,
        )

        predicted_states = rollout(
            initial_state=current_state,
            control_sequence=optimized_controls,
        )

        objective_value = trajectory_cost(
            predicted_states=predicted_states,
            control_sequence=optimized_controls,
            reference_output=reference_output,
        )

        self._previous_solution = optimized_controls.copy()

        return MPCResult(
            control=optimized_controls[0].copy(),
            control_sequence=optimized_controls,
            predicted_states=predicted_states,
            objective_value=float(objective_value),
            success=bool(result.success and np.all(np.isfinite(predicted_states))
                         and (minimum_speed is None or np.min(predicted_states[1:, 2]) >= minimum_speed-1e-6)),
            message=str(result.message),
        )

    def reset(self) -> None:
        """Clear the warm-start sequence at environment reset."""
        self._previous_solution = None

    def _objective(
        self,
        flattened_controls: np.ndarray,
        current_state: np.ndarray,
        reference_output: np.ndarray,
    ) -> float:
        """Objective passed to scipy.optimize.minimize."""
        control_sequence = np.asarray(
            flattened_controls,
            dtype=np.float64,
        ).reshape(self.prediction_horizon, 2)

        predicted_states = rollout(
            initial_state=current_state,
            control_sequence=control_sequence,
        )

        return trajectory_cost(
            predicted_states=predicted_states,
            control_sequence=control_sequence,
            reference_output=reference_output,
        )

    def _objective_and_gradient(self, flattened_controls, current_state, reference_output):
        return tracking_objective_and_gradient(
            current_state,
            np.asarray(flattened_controls).reshape(self.prediction_horizon, 2),
            reference_output,
        )

    def _initial_guess(self) -> np.ndarray:
        """
        Construct the optimizer initial guess.

        First solve:
            zero acceleration / zero steering.

        Later solves:
            shift the previous optimal sequence by one step and repeat the
            final input. This is a standard MPC warm start and does not alter
            the optimization problem itself.
        """
        if self._previous_solution is None:
            return np.zeros(
                (self.prediction_horizon, 2),
                dtype=np.float64,
            )

        shifted = np.empty_like(self._previous_solution)
        shifted[:-1] = self._previous_solution[1:]
        shifted[-1] = self._previous_solution[-1]

        return shifted

    def _optimization_bounds(self) -> list[tuple[float, float]]:
        """Create SciPy bounds for the stacked control sequence."""
        one_step_bounds = [
            (
                float(CONTROL_LOWER_BOUND[0]),
                float(CONTROL_UPPER_BOUND[0]),
            ),
            (
                float(CONTROL_LOWER_BOUND[1]),
                float(CONTROL_UPPER_BOUND[1]),
            ),
        ]

        return one_step_bounds * self.prediction_horizon
