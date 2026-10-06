"""
Model Predictive Control parameters.

Main paper:
- Eq. (14)-(16): kinematic bicycle model
- Eq. (17): MPC cost
- Eq. (18): constrained optimal control problem
"""

import numpy as np

from parameters.environmentParameters import SIMULATION_DT
from parameters.vehicleParameters import (
    MIN_ACCELERATION,
    MAX_ACCELERATION,
    MIN_STEERING_ANGLE,
    MAX_STEERING_ANGLE,
)

# ============================================================
# Sampling period Ts
# Source: Main paper + Section III-C simulation interval
# Unit: s
# ============================================================
MPC_DT: float = SIMULATION_DT

# ============================================================
# Prediction horizon n_p
# Source: Implementation assumption
# The paper defines n_p but does not provide its numerical value.
# ============================================================
PREDICTION_HORIZON: int = 10

# ============================================================
# MPC output and control dimensions
# z = [v, theta]^T
# u = [a_acc, delta]^T
# Source: Main paper, Eq. (14)-(18)
# ============================================================
OUTPUT_DIM: int = 2
CONTROL_DIM: int = 2

# ============================================================
# Cost matrices Q and R
# Source: Implementation assumption
# Eq. (17) defines Q and R but numerical values are not reported.
#
# Initial conservative choice:
# - prioritize v/theta tracking equally
# - lightly penalize control magnitude
# ============================================================
Q = np.diag([1.0, 1.0])
R = np.diag([0.1, 0.1])

# ============================================================
# Control bounds
# Source: HighwayEnv defaults via vehicleParameters.py
# Main paper Eq. (18d) requires u_min and u_max but does not
# give numerical values.
# ============================================================
CONTROL_LOWER_BOUND = np.array(
    [MIN_ACCELERATION, MIN_STEERING_ANGLE],
    dtype=np.float64,
)

CONTROL_UPPER_BOUND = np.array(
    [MAX_ACCELERATION, MAX_STEERING_ANGLE],
    dtype=np.float64,
)

# ============================================================
# Nonlinear optimizer settings
# Source: Implementation assumption
# ============================================================
SOLVER_METHOD: str = "SLSQP"
SOLVER_MAX_ITERATIONS: int = 100
SOLVER_FTOL: float = 1e-6
