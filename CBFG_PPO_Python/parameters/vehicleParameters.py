"""
Vehicle-level physical parameters shared by MPC, CBF, environment, and scenario.

Source labels used in this file:
- Main paper
- Cited reference
- HighwayEnv
- Implementation assumption
"""

from math import pi

# ============================================================
# Vehicle geometry
# Source: HighwayEnv default Vehicle class
# Reference: highway_env.vehicle.kinematics.Vehicle
# Unit: m
# ============================================================
VEHICLE_LENGTH: float = 5.0
VEHICLE_WIDTH: float = 2.0

# ============================================================
# Wheelbase
# Source: Implementation assumption
# Main paper Eq. (14)-(16) defines wheelbase l but does not
# provide a numerical value.
# Unit: m
# ============================================================
WHEELBASE: float = 2.5

# Symmetric front/rear axle placement around the center of gravity.
# Source: Implementation assumption, consistent with beta = atan(0.5*tan(delta))
FRONT_AXLE_TO_COG: float = WHEELBASE / 2.0
REAR_AXLE_TO_COG: float = WHEELBASE / 2.0

# ============================================================
# Low-level control limits
# Source: HighwayEnv ContinuousAction defaults
# Units: m/s^2, rad
# ============================================================
MIN_ACCELERATION: float = -5.0
MAX_ACCELERATION: float = 5.0

MIN_STEERING_ANGLE: float = -pi / 4.0
MAX_STEERING_ANGLE: float = pi / 4.0

# ============================================================
# HighwayEnv kinematic vehicle hard speed limits
# Source: HighwayEnv Vehicle defaults
# Unit: m/s
#
# NOTE:
# These are simulator hard limits, not necessarily the vmax used
# in the paper's reward Eq. (6).
# ============================================================
HIGHWAY_ENV_MIN_SPEED: float = -40.0
HIGHWAY_ENV_MAX_SPEED: float = 40.0
