import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, gaussian_kde

from config import RNG_SEED, T, TAU, G, DECAY_LAMBDA
from simulation import run_simulation
from metrics import compute_harm_distribution, cvar, equalized_odds_gap


def plot_wealth_divergence(mean_W_traj_f, mean_W_traj_cf):
    t_axis = np.arange(1, T + 1)

    illus_rng = np.random.default_rng(RNG_SEED)
    sim_illus = run_simulation(illus_rng)
    W_f  = sim_illus['W_history_f']
    W_cf = sim_illus['W_history_cf']

    mean_W_adv_f   = W_f[:, G == 1].mean(axis=1)
    mean_W_marg_f  = W_f[:, G == 0].mean(axis=1)
    mean_W_adv_cf  = W_cf[:, G == 1].mean(axis=1)
    mean_W_marg_cf = W_cf[:, G == 0].mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_axis, mean_W_adv_f,   color='steelblue', lw=2.5,
            label='Advantaged — Factual')
    ax.plot(t_axis, mean_W_marg_f,  color='firebrick', lw=2.5,
            label='Marginalized — Factual')
    ax.plot(t_axis, mean_W_adv_cf,  color='steelblue', lw=1.5,
            linestyle='--', label='Advantaged — Counterfactual')
    ax.plot(t_axis, mean_W_marg_cf, color='firebrick', lw=1.5,
            linestyle='--', label='Marginalized — Counterfactual')
    ax.axhline(TAU, color='black', lw=1.2, linestyle=':',
               label=f'τ = {TAU}')
    ax.set_xlabel('Time Step $t$', fontsize=12)
    ax.set_ylabel('Mean Wealth $E[W_t | G=g]$', fontsize=12)
    ax.set_title('Wealth Trajectories', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('eq_start_wealth_divergence.png', dpi=150)
    plt.show()


def plot_approval_trajectories(approval_traj):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    labels = ['Marginalized (G=0)', 'Advantaged (G=1)']
    colors = ['firebrick', 'steelblue']
    worlds = [('factual', 'Factual World'), ('cf', 'Counterfactual World')]

    for ax, (key, title) in zip(axes, worlds):
        for g_idx, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(range(1, T + 1), approval_traj[key][:, g_idx],
                    label=label, color=color, lw=2, marker='o', markersize=3)
        ax.axhline(0.05, color='gray', linestyle='--', alpha=0.5, lw=1)
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, lw=1,
                   label='Ceiling/floor threshold')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time Step $t$')
        ax.set_ylabel('P(Ŷ=1 | G)')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle('Approval Rate Trajectories', fontsize=13)
    plt.tight_layout()
    plt.savefig('eq_start_approval_traj.png', dpi=150)
    plt.show()


def plot_harm_distributions(harm_f_final, harm_cf_final,
                            group_name="Marginalized"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x_max   = max(harm_f_final.max(), harm_cf_final.max()) + 0.1
    x_range = np.linspace(0, x_max, 400)

    ax = axes[0]
    kde_f  = gaussian_kde(harm_f_final  + 1e-9)(x_range)
    kde_cf = gaussian_kde(harm_cf_final + 1e-9)(x_range)
    ax.plot(x_range, kde_f,  label='Factual',        color='firebrick', lw=2)
    ax.plot(x_range, kde_cf, label='Counterfactual', color='steelblue', lw=2,
            linestyle='--')
    ax.set_xlabel('Individual Harm $u_i$')
    ax.set_ylabel('Density')
    ax.set_title(f'Harm Distribution — {group_name} Group')
    ax.legend()

    ax2        = axes[1]
    tail_thr   = float(np.percentile(harm_f_final, 75))
    tail_range = x_range[x_range >= tail_thr]
    if len(tail_range) > 5:
        ax2.plot(tail_range,
                 gaussian_kde(harm_f_final  + 1e-9)(tail_range),
                 label='Factual',        color='firebrick', lw=2)
        ax2.plot(tail_range,
                 gaussian_kde(harm_cf_final + 1e-9)(tail_range),
                 label='Counterfactual', color='steelblue', lw=2,
                 linestyle='--')
    ax2.set_xlabel('Individual Harm $u_i$')
    ax2.set_title(f'Tail (Top 25%) — {group_name} Group')
    ax2.legend()

    plt.suptitle(f'Factual vs. Counterfactual Harm\n({group_name} group)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('eq_start_harm_dist.png', dpi=150)
    plt.show()


def plot_wasserstein_trajectory(mean_w_traj, std_w_traj=None):
    t_idx   = np.arange(1, T + 1, dtype=float)
    weights = np.exp(-DECAY_LAMBDA * (t_idx - 1))
    weights /= weights.sum()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_idx, mean_w_traj, color='firebrick', lw=2, marker='o',
            markersize=4,
            label=r'$W_p(H_t^{\mathrm{fact}},\ H_t^{\mathrm{cf}})$')
    if std_w_traj is not None:
        ax.fill_between(t_idx,
                        mean_w_traj - std_w_traj,
                        mean_w_traj + std_w_traj,
                        alpha=0.2, color='firebrick', label='1 SD')

    ax2 = ax.twinx()
    ax2.bar(t_idx, weights, alpha=0.18, color='steelblue',
            label='Early-emphasis weight $w_t$')
    ax2.set_ylabel('Temporal weight $w_t$', color='steelblue', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='steelblue')

    ax.set_xlabel('Time Step $t$')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('eSCI Path: Wasserstein Distance Over Time', fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    plt.tight_layout()
    plt.savefig('eq_start_wasserstein_traj.png', dpi=150)
    plt.show()


def plot_dp_gap_trajectory(approval_traj_mc):
    illus_rng = np.random.default_rng(RNG_SEED + 999)
    sim_dp    = run_simulation(illus_rng)
    dp_over_t = np.abs(
        sim_dp['approval_traj']['factual'][:, 1] -
        sim_dp['approval_traj']['factual'][:, 0]
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, T + 1), dp_over_t, color='#4C72B0', lw=2, marker='s',
            markersize=4, label='DP Gap over time')
    ax.axhline(0.05, color='gray', linestyle='--', lw=1,
               label='Threshold = 0.05')
    ax.set_xlabel('Time Step $t$')
    ax.set_ylabel('|P(Ŷ=1|G=1) − P(Ŷ=1|G=0)|')
    ax.set_title('Demographic Parity Gap Over Time\n', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('eq_start_dp_over_time.png', dpi=150)
    plt.show()
    return dp_over_t


def plot_wealth_distribution_over_time(W_history_f, G,
                                       timesteps_to_plot=(0, 4, 9, 14, 19)):
    adv_mask  = (G == 1)
    marg_mask = (G == 0)

    n_steps = len(timesteps_to_plot)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4.5),
                             sharey=False)

    for col, t_idx in enumerate(timesteps_to_plot):
        ax     = axes[col]
        w_adv  = W_history_f[t_idx, adv_mask]
        w_marg = W_history_f[t_idx, marg_mask]

        x_min   = min(w_adv.min(), w_marg.min()) - 0.05
        x_max   = max(w_adv.max(), w_marg.max()) + 0.05
        x_range = np.linspace(x_min, x_max, 400)

        kde_adv  = gaussian_kde(w_adv)(x_range)
        kde_marg = gaussian_kde(w_marg)(x_range)

        ax.plot(x_range, kde_adv,  color='steelblue', lw=2, label='Advantaged')
        ax.fill_between(x_range, kde_adv,  alpha=0.25, color='steelblue')
        ax.plot(x_range, kde_marg, color='firebrick', lw=2, label='Marginalized')
        ax.fill_between(x_range, kde_marg, alpha=0.25, color='firebrick')

        ax.axvline(TAU, color='black', lw=1.2, linestyle=':', label=f'τ={TAU}')
        ax.axvline(w_adv.mean(),  color='steelblue', lw=1.2, linestyle='--', alpha=0.7)
        ax.axvline(w_marg.mean(), color='firebrick',  lw=1.2, linestyle='--', alpha=0.7)

        ax.set_title(f't = {t_idx + 1}', fontsize=11)
        ax.set_xlabel('Wealth $W$', fontsize=10)
        if col == 0:
            ax.set_ylabel('Probability Density', fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle(
        'Wealth Distribution Over Time\n'
        '(Dashed verticals = group means; dotted = τ)',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig('eq_start_wealth_dist_over_time.png', dpi=150)
    plt.show()


def plot_credit_distribution_over_time(C_history_f, G,
                                       timesteps_to_plot=(0, 4, 9, 14, 19)):
    adv_mask  = (G == 1)
    marg_mask = (G == 0)

    n_steps = len(timesteps_to_plot)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4.5),
                             sharey=False)

    for col, t_idx in enumerate(timesteps_to_plot):
        ax     = axes[col]
        c_adv  = C_history_f[t_idx, adv_mask]
        c_marg = C_history_f[t_idx, marg_mask]

        if c_adv.std() < 1e-6 or c_marg.std() < 1e-6:
            ax.set_title(f't = {t_idx + 1}\n(degenerate)', fontsize=10)
            continue

        x_min   = min(c_adv.min(), c_marg.min()) - 1.0
        x_max   = max(c_adv.max(), c_marg.max()) + 1.0
        x_range = np.linspace(x_min, x_max, 400)

        kde_adv  = gaussian_kde(c_adv)(x_range)
        kde_marg = gaussian_kde(c_marg)(x_range)

        ax.plot(x_range, kde_adv,  color='steelblue', lw=2, label='Advantaged')
        ax.fill_between(x_range, kde_adv,  alpha=0.25, color='steelblue')
        ax.plot(x_range, kde_marg, color='firebrick', lw=2, label='Marginalized')
        ax.fill_between(x_range, kde_marg, alpha=0.25, color='firebrick')

        ax.axvline(c_adv.mean(),  color='steelblue', lw=1.2, linestyle='--', alpha=0.7)
        ax.axvline(c_marg.mean(), color='firebrick',  lw=1.2, linestyle='--', alpha=0.7)

        ax.set_title(f't = {t_idx + 1}', fontsize=11)
        ax.set_xlabel('Credit Score $C$', fontsize=10)
        if col == 0:
            ax.set_ylabel('Probability Density', fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle(
        'Credit Score Distribution Over Time\n(Dashed verticals = group means)',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig('eq_start_credit_dist_over_time.png', dpi=150)
    plt.show()


def plot_esci_both_groups(harm_traj_f, harm_traj_cf, G,
                          weighting='early_emphasis',
                          decay_lambda=DECAY_LAMBDA):
    adv_mask  = (G == 1)
    marg_mask = (G == 0)

    t_count = len(harm_traj_f)
    t_idx   = np.arange(1, t_count + 1, dtype=float)

    w_adv  = np.zeros(t_count)
    w_marg = np.zeros(t_count)

    for t in range(t_count):
        h_f_adv   = compute_harm_distribution(harm_traj_f[t],  adv_mask)
        h_cf_adv  = compute_harm_distribution(harm_traj_cf[t], adv_mask)
        h_f_marg  = compute_harm_distribution(harm_traj_f[t],  marg_mask)
        h_cf_marg = compute_harm_distribution(harm_traj_cf[t], marg_mask)

        w_adv[t]  = wasserstein_distance(h_f_adv,  h_cf_adv)
        w_marg[t] = wasserstein_distance(h_f_marg, h_cf_marg)

    if weighting == 'early_emphasis':
        weights = np.exp(-decay_lambda * (t_idx - 1))
    elif weighting == 'uniform':
        weights = np.ones(t_count)
    elif weighting == 'endpoint_emphasis':
        weights = t_idx / t_idx.sum()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    weights = weights / weights.sum()

    esci_adv  = float(np.sum(weights * w_adv))
    esci_marg = float(np.sum(weights * w_marg))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()
    ax2.bar(t_idx, weights, color='steelblue', alpha=0.15,
            label='Early-emphasis weight $w_t$', zorder=1)
    ax2.set_ylabel('Temporal weight $w_t$', color='steelblue', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_ylim(0, weights.max() * 6)

    ax1.plot(t_idx, w_marg, color='firebrick', lw=2.5, marker='o',
             markersize=5, zorder=3,
             label=f'Marginalized  (eSCI={esci_marg:.4f})')
    ax1.plot(t_idx, w_adv,  color='steelblue', lw=2.5, marker='s',
             markersize=5, zorder=3, linestyle='--',
             label=f'Advantaged    (eSCI={esci_adv:.4f})')
    ax1.fill_between(t_idx, w_marg, w_adv,
                     where=(w_marg >= w_adv),
                     alpha=0.12, color='firebrick',
                     label='Excess harm on Marginalized')

    ax1.set_xlabel('Time Step $t$', fontsize=12)
    ax1.set_ylabel(
        r'$W_p(H_t^{\mathrm{fact}},\; H_t^{\mathrm{cf}})$', fontsize=12)
    ax1.set_title('eSCI Path: Wasserstein Distance Over Time', fontsize=12)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig('eq_start_esci_both_groups.png', dpi=150)
    plt.show()

    return esci_adv, esci_marg


def plot_all_metrics_over_time(mc_results, G, n_mc_detail=50,
                               rng_seed=RNG_SEED):
    adv_mask  = (G == 1)
    marg_mask = (G == 0)

    dp_mat         = np.zeros((n_mc_detail, T))
    wass_marg_mat  = np.zeros((n_mc_detail, T))
    wealth_gap_mat = np.zeros((n_mc_detail, T))
    cvar_gap_mat   = np.zeros((n_mc_detail, T))
    eo_mat         = np.full((n_mc_detail, T), np.nan)

    for run in range(n_mc_detail):
        run_rng    = np.random.default_rng(rng_seed + run)
        sim        = run_simulation(run_rng)
        approval_f = sim['approval_traj']['factual']

        for t in range(T):
            dp_mat[run, t] = float(abs(approval_f[t, 1] - approval_f[t, 0]))

            h_f_m  = compute_harm_distribution(sim['harm_traj_f'][t],  marg_mask)
            h_cf_m = compute_harm_distribution(sim['harm_traj_cf'][t], marg_mask)
            wass_marg_mat[run, t] = wasserstein_distance(h_f_m, h_cf_m)

            wealth_gap_mat[run, t] = (
                sim['W_history_f'][t, adv_mask].mean() -
                sim['W_history_f'][t, marg_mask].mean()
            )
            cvar_gap_mat[run, t] = cvar(h_f_m) - cvar(h_cf_m)

        eo_mat[run, T - 1] = equalized_odds_gap(
            sim['Y_hat_final_f'], sim['Y_true_final'], G
        )

    def ms(arr):
        return arr.mean(axis=0), arr.std(axis=0)

    dp_mean, dp_std = ms(dp_mat)
    wm_mean, wm_std = ms(wass_marg_mat)
    wg_mean, wg_std = ms(wealth_gap_mat)
    cv_mean, cv_std = ms(cvar_gap_mat)
    eo_final_mean   = float(np.nanmean(eo_mat[:, T - 1]))

    dsci_mean = float(mc_results['dsci'].mean())
    t_axis    = np.arange(1, T + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    def _band(ax, t, mean, std, color, label, ls='-', lw=2, marker=None):
        ax.plot(t, mean, color=color, lw=lw, linestyle=ls,
                marker=marker, markersize=4, label=label)
        ax.fill_between(t, mean - std, mean + std, alpha=0.13, color=color)

    _band(ax1, t_axis, wm_mean, wm_std,
          'firebrick', r'Wasserstein $W_p$ — Marg. (eSCI instant.)',
          lw=2.5, marker='o')
    _band(ax1, t_axis, cv_mean, cv_std,
          'darkorange', 'CVaR₉₅ gap — Marg. (fact − CF)', ls='-.', lw=2)

    ax1.axhline(dsci_mean, color='firebrick', lw=1.3, linestyle=':',
                label=f'dSCI (MC mean at t=T) = {dsci_mean:.4f}')
    ax1.scatter([T], [eo_final_mean], color='purple', zorder=5,
                s=80, marker='D',
                label=f'EO Gap at t=T = {eo_final_mean:.4f}')

    ax1.set_ylabel('Wasserstein Distance / CVaR Gap', fontsize=11, color='black')
    ax1.set_ylim(bottom=0)

    _band(ax2, t_axis, dp_mean, dp_std,
          '#2ca02c', 'Demographic Parity Gap', lw=2, marker='s')
    _band(ax2, t_axis, wg_mean, wg_std,
          '#0c6752', 'Liu Wealth Gap: — E[W|Adv] − E[W|Marg]', ls='--', lw=2)

    ax2.set_ylabel(
        'Group-Outcome Metrics\nApproval Rate Gap / Mean Wealth Gap',
        fontsize=10, color='gray', labelpad=8,
    )
    ax2.tick_params(axis='y', labelcolor='gray')

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8.5,
               loc='upper left', framealpha=0.9)

    ax1.set_xlabel('Time Step $t$', fontsize=12)
    ax1.set_title(
        'All Fairness & Harm Metrics Over Time\n'
        f'(Mean ± 1 SD over {n_mc_detail} Monte Carlo runs)',
        fontsize=12,
    )
    ax1.axvline(T, color='gray', lw=0.8, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('all_metrics_over_time.png', dpi=150)
    plt.show()


def print_results_table(mc_res):
    def fmt(arr):
        a = np.asarray(arr, dtype=float)
        return f"{a.mean():.4f} ± {a.std():.4f}"

    print("\n" + "=" * 65)
    print("  All Metrics")
    print("  (mean ± SD over Monte Carlo runs)")
    print("=" * 65)
    print(f"  dSCI  (Wasserstein at t=T)              : {fmt(mc_res['dsci'])}")
    print(f"  eSCI_path (early-emphasis)              : {fmt(mc_res['esci_path'])}")
    print(f"  CVaR_95 gap  (factual − counterfactual) : {fmt(mc_res['cvar95_gap'])}")
    print("-" * 65)
    print("  STATIC METRICS  (cross-sectional, evaluated at t=T)")
    print(f"  Demographic Parity Gap                  : {fmt(mc_res['dp_gap_final'])}")
    print(f"  Equalized Odds Gap                      : {fmt(mc_res['eo_gap_final'])}")
    print("-" * 65)
    print("  LIU ET AL. (2018) DYNAMIC COMPARATOR")
    print(f"  Δμ_marginalized (wealth change t=0→T)   : {fmt(mc_res['liu_delta_marg'])}")
    print(f"  Δμ_advantaged   (wealth change t=0→T)   : {fmt(mc_res['liu_delta_adv'])}")
    print("=" * 65)
