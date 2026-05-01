import numpy as np

from config import RNG_SEED, N, T, G, N_MONTE_CARLO
from simulation import run_simulation
from metrics import (
    compute_harm_distribution,
    compute_dsci,
    compute_esci_path,
    demographic_parity_gap,
    equalized_odds_gap,
    liu_outcome_curve,
    liu_delta_mu,
    cvar,
)


def run_monte_carlo(n_runs=N_MONTE_CARLO):
    results = {
        'dsci':               np.zeros(n_runs),
        'esci_path':          np.zeros(n_runs),
        'dp_gap_final':       np.zeros(n_runs),
        'eo_gap_final':       np.zeros(n_runs),
        'liu_delta_marg':     np.zeros(n_runs),
        'liu_delta_adv':      np.zeros(n_runs),
        'cvar95_gap':         np.zeros(n_runs),
        'wasserstein_traj':   np.zeros((n_runs, T)),
        'mean_W_traj_f':      np.zeros((n_runs, T)),
        'mean_W_traj_cf':     np.zeros((n_runs, T)),
        'mean_W_traj_adv_f':  np.zeros((n_runs, T)),
        'mean_W_traj_marg_f': np.zeros((n_runs, T)),
    }

    marg_mask = (G == 0)
    adv_mask  = (G == 1)

    for run in range(n_runs):
        run_rng = np.random.default_rng(RNG_SEED + run)
        sim     = run_simulation(run_rng)

        h_f_final  = compute_harm_distribution(sim['harm_traj_f'][-1],  marg_mask)
        h_cf_final = compute_harm_distribution(sim['harm_traj_cf'][-1], marg_mask)

        results['dsci'][run] = compute_dsci(h_f_final, h_cf_final)

        esci_val, w_traj = compute_esci_path(
            sim['harm_traj_f'], sim['harm_traj_cf'], marg_mask,
            weighting='early_emphasis',
        )
        results['esci_path'][run]        = esci_val
        results['wasserstein_traj'][run] = w_traj

        results['dp_gap_final'][run] = demographic_parity_gap(
            sim['Y_hat_final_f'], G)
        results['eo_gap_final'][run] = equalized_odds_gap(
            sim['Y_hat_final_f'], sim['Y_true_final'], G)

        liu_curve = liu_outcome_curve(sim['W_history_f'], G)
        delta     = liu_delta_mu(liu_curve)
        results['liu_delta_marg'][run] = float(delta[0])
        results['liu_delta_adv'][run]  = float(delta[1])

        results['cvar95_gap'][run] = cvar(h_f_final) - cvar(h_cf_final)

        results['mean_W_traj_f'][run]  = sim['W_history_f'].mean(axis=1)
        results['mean_W_traj_cf'][run] = sim['W_history_cf'].mean(axis=1)

        results['mean_W_traj_adv_f'][run]  = sim['W_history_f'][:, adv_mask].mean(axis=1)
        results['mean_W_traj_marg_f'][run] = sim['W_history_f'][:, marg_mask].mean(axis=1)

        if (run + 1) % 10 == 0:
            print(f"  Completed run {run + 1}/{n_runs}")

    return results
