"""
Control Barrier Function (CBF) parameters.

Main paper:
- Eq. (19)-(20): discrete-time CBF definition
- Eq. (21)-(24): ellipse and road barrier functions
- Eq. (25): safe-controller optimization
- Eq. (28)-(31): intervention-guided learning
"""

from parameters.environmentParameters import LANE_WIDTH
from parameters.vehicleParameters import VEHICLE_LENGTH, VEHICLE_WIDTH

# ============================================================
# Discrete-time CBF decay constant gamma_hat
# Source: Main paper, Table III
# ============================================================
CBF_GAMMA: float = 0.1

# ============================================================
# Ellipse dimensions
# Source: Main paper Eq. (21) + HighwayEnv vehicle dimensions
#
# The paper denotes al_i and bl_i as ellipse major/minor axis
# lengths, but does not provide numerical values.
# We interpret them as semi-axis lengths derived from the
# HighwayEnv vehicle bounding box.
# Source classification: HighwayEnv + Implementation assumption
# ============================================================
EGO_ELLIPSE_MAJOR_SEMI_AXIS: float = VEHICLE_LENGTH / 2.0
EGO_ELLIPSE_MINOR_SEMI_AXIS: float = VEHICLE_WIDTH / 2.0

SURROUNDING_ELLIPSE_MAJOR_SEMI_AXIS: float = VEHICLE_LENGTH / 2.0
SURROUNDING_ELLIPSE_MINOR_SEMI_AXIS: float = VEHICLE_WIDTH / 2.0

# Eq. (22) does not explicitly introduce an additional margin.
# Keep zero for first reproduction.
# Source: Main paper faithful default
EXTRA_COLLISION_SAFETY_MARGIN: float = 0.0  # [m]

# Road CBF width W_l in Eq. (23)-(24)
# Source: HighwayEnv default lane width
ROAD_LANE_WIDTH: float = LANE_WIDTH

# ============================================================
# Intervention threshold delta in Eq. (29)
# Source: Implementation assumption
# The paper defines delta but does not report a numerical value.
# ============================================================
INTERVENTION_THRESHOLD: float = 1e-3

# ============================================================
# Safe-controller optimizer
# Source: Implementation assumption
# Eq. (25) specifies the optimization problem but not solver
# configuration.
# ============================================================
SOLVER_METHOD: str = "SLSQP"
SOLVER_MAX_ITERATIONS: int = 100
SOLVER_FTOL: float = 1e-6

# Minimum numerical norm treated as zero correction.
CORRECTION_ZERO_TOLERANCE: float = 1e-8
