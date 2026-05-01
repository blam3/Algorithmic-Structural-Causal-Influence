import numpy as np

RNG_SEED = 905

N = 2000
T = 20
TAU = 0.5

G = np.array([1] * (N // 2) + [0] * (N // 2))

# starting distributions
MU_START    = np.array([0.65, 0.50])
SIGMA_STATE = np.array([0.12, 0.08])

MU_A_ADV      = 1.0
MU_A_MARG     = 0.4
SIGMA_A_INIT  = 0.12

RHO_A         = 0.95
ALPHA_A       = 0.3
SIGMA_A_NOISE = 0.05

# decision model
INTERCEPT = -1.5
BETA_A    =  0.6
BETA_W    =  0.25

# VAR feedback
VAR_MATRIX = np.array([[0.90, 0.05],
                        [0.10, 0.85]])

BETA_INTERACTION      = 0.05
GAMMA_APPROVED        = np.array([+0.50, +5.0])
GAMMA_DENIED          = np.array([-0.08, -2.0])
POVERTY_TRAP_PENALTY  = np.array([-0.80, -12.0])
VAR_NOISE_SIGMA       = 0.10

# harm weights
LAMBDA_FN = 1.0
LAMBDA_FP = 2.0

# eSCI
DECAY_LAMBDA = 0.3

N_MONTE_CARLO = 1000
