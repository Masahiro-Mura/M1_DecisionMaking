"""
Intelligent Driver Model (IDM) parameters for surrounding vehicles.

Main paper:
- Eq. (32): acceleration law
- Eq. (33): desired dynamic gap
- Table II: parameter values/ranges
"""

# ============================================================
# Desired speed v0
# Source: Main paper, Table II
# Unit: m/s
# Surrounding vehicles should sample this range per scenario.
# ============================================================
DESIRED_SPEED_MIN: float = 10.0
DESIRED_SPEED_MAX: float = 20.0

# ============================================================
# Jam distance d0
# Source: Main paper, Table II
# Unit: m
# ============================================================
JAM_DISTANCE: float = 7.0

# ============================================================
# Maximum acceleration a_ma
# Source: Main paper, Table II
# Unit: m/s^2
# ============================================================
MAX_ACCELERATION: float = 3.0

# ============================================================
# Maximum/comfortable deceleration b_md
# Source: Main paper, Table II
# Unit: m/s^2
# ============================================================
MAX_DECELERATION: float = 5.0

# ============================================================
# Desired time gap T_g
# Source: Main paper, Table II
# Unit: s
# ============================================================
DESIRED_TIME_GAP: float = 1.5

# ============================================================
# IDM acceleration exponent
# Source: Main paper, Eq. (32)
# The exponent is explicitly 4.
# ============================================================
ACCELERATION_EXPONENT: float = 4.0

# ============================================================
# Numerical safeguards
# Source: Implementation assumption
# Used later to avoid division by zero in IDM evaluation.
# ============================================================
MIN_GAP_EPSILON: float = 1e-3
