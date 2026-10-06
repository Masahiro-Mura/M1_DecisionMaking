"""
Parameters for the RL environment and simulation timing.

The state/action definitions follow Table I and Eq. (1)-(7)
of the main paper.
"""

from math import pi

# ============================================================
# Simulation timing
# Source: Main paper, Section III-C
# "The simulation interval is set to 0.2 s."
# Unit: s
# ============================================================
SIMULATION_DT: float = 0.2
POLICY_FREQUENCY_HZ: float = 1.0 / SIMULATION_DT

# ============================================================
# Observation space
# Source: Main paper, Table I
# Total state dimension: R^27
# ============================================================
OBSERVATION_DIM: int = 27
NUM_SURROUNDING_VEHICLES: int = 4

EGO_VELOCITY_DIM: int = 2
EGO_POSITION_DIM: int = 2
LATERAL_DEVIATION_DIM: int = 1
LEFT_CHANGE_FEASIBILITY_DIM: int = 1
RIGHT_CHANGE_FEASIBILITY_DIM: int = 1
SURROUNDING_VELOCITY_DIM: int = 8
SURROUNDING_RELATIVE_POSITION_DIM: int = 8
SURROUNDING_ACCELERATION_DIM: int = 4

assert (
    EGO_VELOCITY_DIM
    + EGO_POSITION_DIM
    + LATERAL_DEVIATION_DIM
    + LEFT_CHANGE_FEASIBILITY_DIM
    + RIGHT_CHANGE_FEASIBILITY_DIM
    + SURROUNDING_VELOCITY_DIM
    + SURROUNDING_RELATIVE_POSITION_DIM
    + SURROUNDING_ACCELERATION_DIM
    == OBSERVATION_DIM
)

# Main paper states that all dynamic observation elements are normalized.
OBSERVATION_NORMALIZED_MIN: float = -1.0
OBSERVATION_NORMALIZED_MAX: float = 1.0

# ============================================================
# RL action space
# Source: Main paper, Eq. (1)
# A = [v_ref, theta_ref], both normalized to [-1, 1]
# ============================================================
ACTION_DIM: int = 2
ACTION_NORMALIZED_MIN: float = -1.0
ACTION_NORMALIZED_MAX: float = 1.0

# ============================================================
# Physical denormalization ranges for RL outputs
# Source: Implementation assumption
#
# The paper specifies normalized action bounds but does not state
# the exact mapping back to physical v_ref and theta_ref.
# The speed upper bound is aligned with the upper end of the IDM
# desired-speed range in Table II.
# ============================================================
REFERENCE_SPEED_MIN: float = 0.0       # [m/s]
REFERENCE_SPEED_MAX: float = 20.0      # [m/s]
REFERENCE_HEADING_MIN: float = -pi / 4.0  # [rad]
REFERENCE_HEADING_MAX: float = pi / 4.0   # [rad]

# ============================================================
# Road geometry
# Source: HighwayEnv default AbstractLane width
# Main paper Eq. (23)-(24) uses W_l but does not provide a value.
# Unit: m
# ============================================================
LANE_WIDTH: float = 4.0

# ============================================================
# Training/evaluation speed cases and episode duration
# Source: Main paper, Section III-A and III-D
# ============================================================
TRAINING_SPEED_CASES = (5.0, 10.0)  # [m/s]

EPISODE_DURATION_BY_MIN_SPEED = {
    5.0: 40.0,   # [s]
    10.0: 20.0,  # [s]
}

MAX_STEPS_BY_MIN_SPEED = {
    speed: int(duration / SIMULATION_DT)
    for speed, duration in EPISODE_DURATION_BY_MIN_SPEED.items()
}

NUM_RANDOM_SEED_TRIALS: int = 10

# ============================================================
# Reward parameters
# Source: Main paper, Eq. (2)-(7)
# ============================================================
LANE_CENTER_REWARD_COEFFICIENT: float = 4.0
MERGE_DISTANCE_SCALE: float = 40.0
MERGE_EXPONENT: float = 0.5

TERMINAL_SUCCESS_REWARD: float = 1.0
TERMINAL_FAILURE_REWARD: float = -1.0

# Eq. (6) uses vmax, but its numerical value is not given.
# Source: Implementation assumption
REWARD_MAX_SPEED: float = 20.0  # [m/s]

# ============================================================
# Termination rules
# Source: Main paper, Section III-D
# ============================================================
TERMINATE_ON_COLLISION: bool = True
TERMINATE_ON_ROAD_DEPARTURE: bool = True
TERMINATE_ON_SUCCESSFUL_MERGE: bool = True

# ============================================================
# Merge-road geometry
# Source: HighwayEnv merge environment cited by the main paper [45]
#
# The values below reproduce the standard HighwayEnv merge-road
# segmentation:
#   150 m approach
#    80 m convergence
#    80 m parallel merge section
#   150 m downstream section
#
# The paper's condensed case states that the merge section is
# x = 230...310 m and that the ramp/main-road lateral regions are
# y = 6...10 m and y = 2...6 m, respectively. This is consistent
# with the HighwayEnv merge geometry with 4 m lane width.
# ============================================================
ROAD_APPROACH_LENGTH: float = 150.0
ROAD_CONVERGENCE_LENGTH: float = 80.0
ROAD_PARALLEL_MERGE_LENGTH: float = 80.0
ROAD_DOWNSTREAM_LENGTH: float = 150.0

ROAD_MAIN_LENGTH: float = (
    ROAD_APPROACH_LENGTH
    + ROAD_CONVERGENCE_LENGTH
    + ROAD_PARALLEL_MERGE_LENGTH
    + ROAD_DOWNSTREAM_LENGTH
)

MERGE_SECTION_START_X: float = ROAD_APPROACH_LENGTH + ROAD_CONVERGENCE_LENGTH
MERGE_SECTION_END_X: float = MERGE_SECTION_START_X + ROAD_PARALLEL_MERGE_LENGTH

# Standard HighwayEnv merge geometry.
# Source: HighwayEnv merge implementation [45]
MERGE_SINE_AMPLITUDE: float = 3.25

# Main-road lane centre lines are y = 0 m and y = 4 m.
# The parallel ramp centre line is y = 8 m.
# With 4 m lane width, this yields:
#   upper main-road lane region: y = 2...6 m
#   parallel ramp region:        y = 6...10 m
MAIN_ROAD_LANE_CENTERS = (0.0, LANE_WIDTH)
PARALLEL_RAMP_CENTER_Y: float = 2.0 * LANE_WIDTH

# ============================================================
# Scenario initialization
# ============================================================

# Source: Main paper, condensed-case discussion in Section IV-B
EGO_INITIAL_RAMP_X: float = MERGE_SECTION_START_X  # 230 m
EGO_INITIAL_ACCELERATION: float = 0.0  # [m/s^2]

# Main-paper experiments use ego speed cases of 5 m/s and 10 m/s.
DEFAULT_EGO_INITIAL_SPEED: float = 5.0  # [m/s]

# Source: Main paper condensed case
# Surrounding vehicles: 10 m/s with +/- 2 m/s variation.
SURROUNDING_INITIAL_SPEED_MIN: float = 8.0   # [m/s]
SURROUNDING_INITIAL_SPEED_MAX: float = 12.0  # [m/s]
SURROUNDING_INITIAL_ACCELERATION: float = 0.0  # [m/s^2]

# Source: Implementation assumption
# The paper says the number of surrounding vehicles is randomized
# but does not report the sampling range.
SURROUNDING_VEHICLE_COUNT_MIN: int = 6
SURROUNDING_VEHICLE_COUNT_MAX: int = 10

# Source: Implementation assumption
# Positions are sampled over the useful highway region while avoiding
# spawn points too close to the road endpoints.
SURROUNDING_SPAWN_X_MIN: float = 40.0
SURROUNDING_SPAWN_X_MAX: float = ROAD_MAIN_LENGTH - 20.0

# Source: Implementation assumption
# Maximum retries for collision-free randomized initialization.
MAX_SPAWN_ATTEMPTS_PER_VEHICLE: int = 200

# ============================================================
# Observation normalization
# Source: Main paper states that all dynamic observation elements
# are normalized, but it does not report the physical min/max
# values used for that normalization.
#
# The following ranges are implementation assumptions for v1.
# They are kept here (not hard-coded in observation.py) so they
# can be tuned later without changing environment logic.
# ============================================================
OBS_SPEED_ABS_MAX: float = REFERENCE_SPEED_MAX

OBS_POSITION_X_MIN: float = 0.0
OBS_POSITION_X_MAX: float = ROAD_MAIN_LENGTH
OBS_POSITION_Y_MIN: float = -LANE_WIDTH
OBS_POSITION_Y_MAX: float = 4.0 * LANE_WIDTH

OBS_RELATIVE_X_ABS_MAX: float = ROAD_MAIN_LENGTH
OBS_RELATIVE_Y_ABS_MAX: float = 4.0 * LANE_WIDTH

# Missing surrounding-vehicle slots are zero-filled after normalization.
OBS_MISSING_VEHICLE_FILL: float = 0.0

# ============================================================
# Reward helper for missing front/rear vehicles
# Source: Implementation assumption
#
# Eq. (4) requires both dP and dF, but the paper does not specify
# what to do when one side has no detected vehicle.
# ============================================================
MERGE_MISSING_VEHICLE_DISTANCE: float = 2.0 * MERGE_DISTANCE_SCALE

# ============================================================
# Success definition
# Source: Implementation assumption based on the paper's statement
# that success means accomplishing the merge task and driving to
# the designated target location without collision.
#
# In the standard merge road, lane IDs 0 and 1 are main-road lanes,
# while lane ID 2 is the parallel ramp lane.
#
# Previous v1 used MERGE_SECTION_START_X (=230 m), which could mark
# a partially completed lane change as success. The revised criterion
# requires the ego vehicle to pass the end of the merge section.
# ============================================================
MAIN_ROAD_LANE_IDS = (0, 1)
SUCCESS_MIN_X: float = MERGE_SECTION_END_X

# Source: Implementation assumption.
# The paper does not report a numerical lateral-deviation threshold for
# declaring completion. Requiring the ego center to be within 1 m of the
# selected main-lane center prevents premature success during lane crossing.
SUCCESS_MAX_LATERAL_DEVIATION: float = 1.0

# IMPLEMENTATION ASSUMPTION: IV-A requires no further failure for a specified
# time, and IV-B describes a designated target, without numerical definitions.
# Require 2 s continuously centered on a main lane after merging, and reach a
# downstream target 10 m before the finite simulated road ends. Configurable
# reproduction choices, not values reported by the paper.
SUCCESS_STABILITY_DURATION: float = 2.0  # [s]
SUCCESS_TARGET_X: float = ROAD_MAIN_LENGTH - 10.0  # [m]

# ============================================================
# Evaluation / video settings
# ============================================================

# Source: Main paper, Section IV-B
# Each learned driving policy is tested over 50 episodes.
EVALUATION_EPISODES: int = 50

# Source: Main paper simulation interval + implementation detail
# One rendered frame per environment decision step.
VIDEO_FPS: int = int(round(1.0 / SIMULATION_DT))
