import numpy as np
from scipy.stats import wasserstein_distance

from config import LAMBDA_FN, LAMBDA_FP, DECAY_LAMBDA


def compute_pointwise_harm(Y_hat, Y_true, W, tau,
                           lambda_fn=LAMBDA_FN, lambda_fp=LAMBDA_FP):
    harm = np.zeros(len(Y_hat))

    fn_mask = (Y_hat == 0) & (Y_true == 1)
    harm[fn_mask] += lambda_fn

    fp_mask         = (Y_hat == 1) & (W < tau)
    depth_below_tau = np.maximum(tau - W, 0.0)
    harm[fp_mask]  += lambda_fp * depth_below_tau[fp_mask]

    return harm


def compute_harm_distribution(harm_values, group_mask):
    return harm_values[group_mask]


def compute_dsci(harm_f_group, harm_cf_group):
    return float(wasserstein_distance(harm_f_group, harm_cf_group))


def compute_esci_path(harm_traj_f, harm_traj_cf, group_mask,
                      weighting='early_emphasis', decay_lambda=DECAY_LAMBDA):
    T_sim  = len(harm_traj_f)
    w_by_t = np.zeros(T_sim)

    for t in range(T_sim):
        h_f  = compute_harm_distribution(harm_traj_f[t],  group_mask)
        h_cf = compute_harm_distribution(harm_traj_cf[t], group_mask)
        w_by_t[t] = wasserstein_distance(h_f, h_cf)

    t_idx = np.arange(1, T_sim + 1, dtype=float)

    if weighting == 'uniform':
        weights = np.ones(T_sim)
    elif weighting == 'early_emphasis':
        weights = np.exp(-decay_lambda * (t_idx - 1))
    elif weighting == 'endpoint_emphasis':
        weights = t_idx / t_idx.sum()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    weights         = weights / weights.sum()
    esci_path_value = float(np.sum(weights * w_by_t))
    return esci_path_value, w_by_t


def demographic_parity_gap(Y_hat, G):
    return float(abs(Y_hat[G == 1].mean() - Y_hat[G == 0].mean()))


def equalized_odds_gap(Y_hat, Y_true, G):
    for g_val in [0, 1]:
        mask = G == g_val
        if (Y_true[mask] == 1).sum() == 0 or (Y_true[mask] == 0).sum() == 0:
            return np.nan
    tpr = {g: Y_hat[(G == g) & (Y_true == 1)].mean() for g in [0, 1]}
    fpr = {g: Y_hat[(G == g) & (Y_true == 0)].mean() for g in [0, 1]}
    return float(max(abs(tpr[1] - tpr[0]), abs(fpr[1] - fpr[0])))


def liu_outcome_curve(W_history, G):
    T_sim = W_history.shape[0]
    curve = np.zeros((T_sim, 2))
    for t in range(T_sim):
        curve[t, 0] = W_history[t, G == 0].mean()
        curve[t, 1] = W_history[t, G == 1].mean()
    return curve


def liu_delta_mu(outcome_curve):
    return outcome_curve[-1] - outcome_curve[0]


def cvar(harm_samples, alpha=0.95):
    if len(harm_samples) == 0:
        return 0.0
    threshold = np.percentile(harm_samples, alpha * 100)
    tail      = harm_samples[harm_samples >= threshold]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def tail_analysis(harm_f, harm_cf, group_name="Marginalized",
                  percentiles=(75, 90, 95, 99)):
    results = {}
    print(f"\n{'='*55}")
    print(f"  Tail Analysis — {group_name} Group")
    print(f"{'='*55}")
    for p in percentiles:
        q_f     = float(np.percentile(harm_f,  p))
        q_cf    = float(np.percentile(harm_cf, p))
        mass_f  = float(np.mean(harm_f  > q_f))
        mass_cf = float(np.mean(harm_cf > q_f))
        excess  = mass_f - mass_cf
        results[p] = {
            'factual_threshold': q_f,
            'cf_threshold':      q_cf,
            'excess_tail_mass':  excess,
        }
        print(f"  P{p:>2d}: Factual={q_f:.4f}  CF={q_cf:.4f}  "
              f"Excess tail mass={excess:+.4f}")

    cvar_f  = cvar(harm_f)
    cvar_cf = cvar(harm_cf)
    results['cvar95'] = {
        'factual':        cvar_f,
        'counterfactual': cvar_cf,
        'gap':            cvar_f - cvar_cf,
    }
    print(f"\n  CVaR_95 — Factual: {cvar_f:.4f} | CF: {cvar_cf:.4f} | "
          f"Gap: {cvar_f - cvar_cf:+.4f}")
    return results
