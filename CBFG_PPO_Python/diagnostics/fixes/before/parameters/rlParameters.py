"""
PPO / CBFG-PPO learning parameters.

Main paper:
- Eq. (8)-(13): PPO actor/critic losses
- Eq. (27)-(31): CBF-guided behavior-cloning objective
- Fig. 6: network architecture
- Table III: experiment hyperparameters
"""

# ============================================================
# Main-paper PPO hyperparameters
# Source: Main paper, Table III
# ============================================================
DISCOUNT_FACTOR_GAMMA: float = 0.99
LEARNING_RATE: float = 0.0006
NUM_OPTIMIZATION_EPOCHS: int = 40 
# NUM_OPTIMIZATION_EPOCHS: int = 5 # Modefied for faster reproduction, original paper uses 40 epochs

REPLAY_BUFFER_SIZE: int = 128 * 24  # 3072
# REPLAY_BUFFER_SIZE: int = 1024 # Modefied for faster reproduction, original paper uses 3072
TRAINING_BATCH_SIZE: int = 128
TOTAL_TIMESTEPS: int = 150_000 
# TOTAL_TIMESTEPS: int = 2_000  # Modefied for faster reproduction, original paper uses 150_000  

# ============================================================
# Network architecture
# Source: Main paper, Fig. 6
# Both policy and value networks use two Dense(256) layers.
# ReLU is used unless otherwise stated.
# ============================================================
OBSERVATION_DIM: int = 27
ACTION_DIM: int = 2

ACTOR_HIDDEN_DIMS = (256, 256)
CRITIC_HIDDEN_DIMS = (256, 256)
HIDDEN_ACTIVATION: str = "relu"

# Policy is stochastic and outputs a Gaussian action distribution.
# The paper's Fig. 6 indicates distribution parameters but does
# not provide an initial log standard deviation.
# Source: Implementation assumption
INITIAL_LOG_STD: float = -0.5

# ============================================================
# PPO clipping epsilon
# Source: Cited PPO paper [47]
# Schulman et al. give epsilon = 0.2 as the representative
# clipped-objective setting.
# The main paper itself does not report epsilon numerically.
# ============================================================
PPO_CLIP_EPSILON: float = 0.2

# ============================================================
# GAE lambda
# Source: Cited PPO paper [47]
# Main paper says "Calculate Advantage" but does not specify
# the estimator or lambda. PPO commonly uses GAE(lambda=0.95).
# ============================================================
GAE_LAMBDA: float = 0.95

# ============================================================
# Loss coefficients
# Source: Main paper, Eq. (13)
# varsigma = 1, epsilon = 0.5
# ============================================================
POLICY_LOSS_COEFFICIENT: float = 1.0
VALUE_LOSS_COEFFICIENT: float = 0.5

# ============================================================
# Optimizer
# Source: Main paper, Section III-C
# Adam is explicitly stated.
# ============================================================
OPTIMIZER_NAME: str = "Adam"

# ============================================================
# Entropy bonus
# Source: Implementation assumption
# The main paper does not report an entropy coefficient.
# Start with zero to avoid adding an unreported objective term.
# ============================================================
ENTROPY_COEFFICIENT: float = 0.0

# ============================================================
# Gradient clipping
# Source: Implementation assumption
# Not specified in the main paper.
# ============================================================
MAX_GRAD_NORM: float = 0.5

# ============================================================
# Main-script logging/checkpoint settings
# ============================================================

# Source: Main paper, Section IV-A
# Success-rate curves are smoothed with a sliding window of width 10.
SUCCESS_SMOOTHING_WINDOW: int = 10

# Source: Implementation assumption
# The main paper does not report checkpoint frequency.
CHECKPOINT_INTERVAL_STEPS: int = 30_000

