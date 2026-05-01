import numpy as np

from config import RNG_SEED, N, T, G, MU_START, SIGMA_STATE, MU_A_ADV, MU_A_MARG, SIGMA_A_INIT, N_MONTE_CARLO
from simulation import run_simulation
from metrics import compute_harm_distribution, compute_esci_path, liu_outcome_curve, tail_analysis
from monte_carlo import run_monte_carlo
from plots import (
    plot_approval_trajectories,
    plot_harm_distributions,
    plot_wasserstein_trajectory,
    plot_wealth_divergence,
    plot_dp_gap_trajectory,
    plot_wealth_distribution_over_time,
    plot_credit_distribution_over_time,
    plot_esci_both_groups,
    plot_all_metrics_over_time,
    print_results_table,
)


def main():
    print("=" * 65)
    print(f"  N={N}  T={T}  MC runs={N_MONTE_CARLO}")
    print(f"  Initial wealth: BOTH groups ~ N({MU_START[0]}, {SIGMA_STATE[0]}²)")
    print(f"  Structural A: Adv ~ N({MU_A_ADV}, {SIGMA_A_INIT}²)  "
          f"Marg ~ N({MU_A_MARG}, {SIGMA_A_INIT}²)")
    print("=" * 65)

    print("\n--- Single illustrative run ---")
    single_rng = np.random.default_rng(RNG_SEED)
    sim_single = run_simulation(single_rng)

    marg_mask = (G == 0)

    plot_approval_trajectories(sim_single['approval_traj'])

    h_f_final  = compute_harm_distribution(sim_single['harm_traj_f'][-1],  marg_mask)
    h_cf_final = compute_harm_distribution(sim_single['harm_traj_cf'][-1], marg_mask)

    plot_harm_distributions(h_f_final, h_cf_final)
    tail_analysis(h_f_final, h_cf_final, group_name="Marginalized")

    _, w_traj_single = compute_esci_path(
        sim_single['harm_traj_f'], sim_single['harm_traj_cf'],
        marg_mask, weighting='early_emphasis',
    )
    plot_wasserstein_trajectory(w_traj_single)

    print("\n--- Monte Carlo simulation ---")
    mc_results = run_monte_carlo(N_MONTE_CARLO)

    plot_wealth_divergence(
        mc_results['mean_W_traj_f'],
        mc_results['mean_W_traj_cf'],
    )

    mean_w_traj = mc_results['wasserstein_traj'].mean(axis=0)
    std_w_traj  = mc_results['wasserstein_traj'].std(axis=0)
    plot_wasserstein_trajectory(mean_w_traj, std_w_traj)

    plot_dp_gap_trajectory(mc_results)

    print_results_table(mc_results)

    timesteps = (0, 4, 9, 14, 19)

    plot_wealth_distribution_over_time(sim_single['W_history_f'], G,
                                       timesteps_to_plot=timesteps)
    plot_credit_distribution_over_time(sim_single['C_history_f'], G,
                                       timesteps_to_plot=timesteps)

    esci_adv_val, esci_marg_val = plot_esci_both_groups(
        sim_single['harm_traj_f'],
        sim_single['harm_traj_cf'],
        G,
    )
    print(f"\n  eSCI (Advantaged)    = {esci_adv_val:.4f}")
    print(f"  eSCI (Marginalized)  = {esci_marg_val:.4f}")

    plot_all_metrics_over_time(mc_results, G, n_mc_detail=50)


if __name__ == "__main__":
    main()
