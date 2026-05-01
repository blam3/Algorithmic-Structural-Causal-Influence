import numpy as np
from scipy.special import expit

from config import (
    N, T, TAU, G,
    MU_START, SIGMA_STATE,
    MU_A_ADV, MU_A_MARG, SIGMA_A_INIT,
    RHO_A, ALPHA_A, SIGMA_A_NOISE,
    INTERCEPT, BETA_A, BETA_W,
    VAR_MATRIX, BETA_INTERACTION,
    GAMMA_APPROVED, GAMMA_DENIED, POVERTY_TRAP_PENALTY, VAR_NOISE_SIGMA,
)
from metrics import compute_pointwise_harm


def initialize_population(n, g, mu_start, sigma, mu_a_adv, mu_a_marg, sigma_a,
                           rng_local):
    S = np.zeros((n, 2))
    A = np.zeros(n)

    adv_mask  = (g == 1)
    marg_mask = (g == 0)

    S[adv_mask]  = rng_local.normal(mu_start, sigma,
                                    size=(int(adv_mask.sum()), 2))
    S[marg_mask] = rng_local.normal(mu_start, sigma,
                                    size=(int(marg_mask.sum()), 2))

    A[adv_mask]  = rng_local.normal(mu_a_adv,  sigma_a,
                                    size=int(adv_mask.sum()))
    A[marg_mask] = rng_local.normal(mu_a_marg, sigma_a,
                                    size=int(marg_mask.sum()))

    A       = np.clip(A, 0, None)
    S[:, 0] = np.clip(S[:, 0], 0, None)

    Y = (S[:, 0] >= TAU).astype(int)
    return S, A, Y


def make_decisions(S, A, rng_local):
    logit = INTERCEPT + BETA_A * A + BETA_W * S[:, 0]
    prob  = expit(logit)
    Y_hat = rng_local.binomial(1, prob)
    return Y_hat, prob


def check_degeneracy(approval_rates, t, threshold=0.05):
    for g_val, label in zip([0, 1], ['Marginalized', 'Advantaged']):
        rate = approval_rates[g_val]
        if rate < threshold or rate > (1.0 - threshold):
            print(f"  WARNING [t={t}]: {label} approval rate = {rate:.3f} "
                  f"— potential ceiling/floor effect")


def var_feedback_transition(S, Y_hat, tau, rng_local):
    Y_hat_col = Y_hat.reshape(-1, 1)

    S_ar          = S @ VAR_MATRIX.T
    S_interaction = BETA_INTERACTION * Y_hat_col * S

    gamma = np.where(Y_hat_col == 1, GAMMA_APPROVED, GAMMA_DENIED)

    poverty_trap_mask        = (Y_hat == 1) & (S[:, 0] < tau)
    gamma[poverty_trap_mask] = POVERTY_TRAP_PENALTY

    noise = rng_local.normal(0.0, VAR_NOISE_SIGMA, size=S.shape)

    S_next       = S_ar + S_interaction + gamma + noise
    S_next[:, 0] = np.clip(S_next[:, 0], 0, None)
    return S_next


def update_structural_attribute(A, Y_hat, W, tau, rng_local):
    successful_loan = (Y_hat == 1) & (W >= tau)
    noise = rng_local.normal(0.0, SIGMA_A_NOISE, size=A.shape)
    A_new = RHO_A * A + ALPHA_A * successful_loan.astype(float) + noise
    return np.clip(A_new, 0, None)


def counterfactual_intervention(A_factual, G, adv_mean_A):
    A_cf      = A_factual.copy()
    marg_mask = (G == 0)
    gap       = adv_mean_A - A_cf[marg_mask].mean()
    A_cf[marg_mask] += gap
    return A_cf


def run_simulation(rng_local):
    S_f, A_f, _ = initialize_population(
        N, G, MU_START, SIGMA_STATE,
        MU_A_ADV, MU_A_MARG, SIGMA_A_INIT,
        rng_local,
    )

    S_cf = S_f.copy()
    A_cf = A_f.copy()

    harm_traj_f   = []
    harm_traj_cf  = []
    W_history_f   = np.zeros((T, N))
    W_history_cf  = np.zeros((T, N))
    C_history_f   = np.zeros((T, N))
    C_history_cf  = np.zeros((T, N))
    approval_traj = {
        'factual': np.zeros((T, 2)),
        'cf':      np.zeros((T, 2)),
    }

    for t in range(T):
        Y_true_f  = (S_f[:, 0]  >= TAU).astype(int)
        Y_true_cf = (S_cf[:, 0] >= TAU).astype(int)

        Y_hat_f, _ = make_decisions(S_f, A_f, rng_local)

        adv_mean_A      = float(A_f[G == 1].mean())
        A_cf_intervened = counterfactual_intervention(A_cf, G, adv_mean_A)
        Y_hat_cf, _     = make_decisions(S_cf, A_cf_intervened, rng_local)

        harm_f  = compute_pointwise_harm(Y_hat_f,  Y_true_f,  S_f[:, 0],  TAU)
        harm_cf = compute_pointwise_harm(Y_hat_cf, Y_true_cf, S_cf[:, 0], TAU)

        harm_traj_f.append(harm_f)
        harm_traj_cf.append(harm_cf)

        S_f  = var_feedback_transition(S_f,  Y_hat_f,  TAU, rng_local)
        S_cf = var_feedback_transition(S_cf, Y_hat_cf, TAU, rng_local)

        A_f  = update_structural_attribute(A_f,             Y_hat_f,  S_f[:, 0],  TAU, rng_local)
        A_cf = update_structural_attribute(A_cf_intervened, Y_hat_cf, S_cf[:, 0], TAU, rng_local)

        W_history_f[t]  = S_f[:, 0]
        W_history_cf[t] = S_cf[:, 0]
        C_history_f[t]  = S_f[:, 1]
        C_history_cf[t] = S_cf[:, 1]

        for g_val in [0, 1]:
            approval_traj['factual'][t, g_val] = float(Y_hat_f[G  == g_val].mean())
            approval_traj['cf'][t, g_val]      = float(Y_hat_cf[G == g_val].mean())

        rates = {g_val: float(Y_hat_f[G == g_val].mean()) for g_val in [0, 1]}
        check_degeneracy(rates, t)

    return {
        'harm_traj_f':    harm_traj_f,
        'harm_traj_cf':   harm_traj_cf,
        'W_history_f':    W_history_f,
        'W_history_cf':   W_history_cf,
        'C_history_f':    C_history_f,
        'C_history_cf':   C_history_cf,
        'approval_traj':  approval_traj,
        'Y_hat_final_f':  Y_hat_f,
        'Y_hat_final_cf': Y_hat_cf,
        'Y_true_final':   Y_true_f,
        'S_final_f':      S_f,
        'S_final_cf':     S_cf,
    }
