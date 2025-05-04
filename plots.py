import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from heston_asian_option import HestonAsianOption



def plot_heston_paths_with_ci(model, n_paths=50, conf=0.95):
    # Save original settings
    orig_M = model.M
    orig_scheme = model.scheme

    # Generate Euler paths
    model.M = n_paths
    model.scheme = 'euler'
    S_euler, _ = model.generate_paths()

    # Generate Milstein paths
    model.scheme = 'milstein'
    S_milstein, _ = model.generate_paths()

    # Restore original settings
    model.M = orig_M
    model.scheme = orig_scheme

    # Time grid
    t = np.linspace(0, model.T, model.N + 1)
    z = norm.ppf((1 + conf) / 2)

    # Prepare figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, S, title in zip(
        axes,
        [S_euler, S_milstein],
        ['Euler scheme', 'Milstein scheme']
    ):
        # Plot individual paths
        for i in range(n_paths):
            ax.plot(t, S[i, :], lw=0.8, alpha=0.6)

        # Compute statistics
        mean_path = S.mean(axis=0)
        se = S.std(axis=0) / np.sqrt(n_paths)
        ci = z * se

        # Plot mean and CI
        ax.plot(t, mean_path, color='black', lw=2, label='Mean')
        ax.fill_between(t, mean_path - ci, mean_path + ci, color='gray', alpha=0.3, label=f'{int(conf*100)}% CI')

        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Asset Price' if title.startswith('Euler') else '')
        ax.legend()

    plt.tight_layout()
    plt.show()

def compare_path_sets(paths1, paths2, T,
                      title1='Model 1', title2='Model 2',
                      num_paths=50, conf=0.95):
    M, N_plus_1 = paths1.shape
    time_grid = np.linspace(0, T, N_plus_1)
    z = norm.ppf((1 + conf) / 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, (paths, title, ax) in enumerate(zip(
            [paths1, paths2], [title1, title2], axes)):
        # Plot sample paths
        for i in range(min(num_paths, M)):
            ax.plot(time_grid, paths[i], lw=0.7, alpha=0.6)

        # Compute mean and CI
        mean_path = np.mean(paths, axis=0)
        se = np.std(paths, axis=0) / np.sqrt(M)
        ci = z * se

        # Plot mean and CI
        ax.plot(time_grid, mean_path, color='black', lw=2, label='Mean path')
        ax.fill_between(time_grid, mean_path - ci, mean_path + ci,
                        color='gray', alpha=0.3,
                        label=f'{int(conf*100)}% CI')

        ax.set_title(title)
        ax.set_xlabel('Time')
        if idx == 0:
            ax.set_ylabel('Price')
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_variance_reduction_efficacy(results):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot convergence of option prices
    ax1.set_title('Option Price Convergence with Increasing Number of Paths')
    ax1.plot(results['paths'], results['price_without_cv'], marker='o', label='Without Control Variate')
    ax1.plot(results['paths'], results['price_with_optimal_cv'], marker='s', label='With Optimal Control Variate')
    ax1.set_xlabel('Number of Paths')
    ax1.set_ylabel('Option Price')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True)
    
    # Plot standard errors
    ax2.set_title('Standard Error Reduction with Increasing Number of Paths')
    ax2.plot(results['paths'], results['std_error_without_cv'], marker='o', label='Without Control Variate')
    ax2.plot(results['paths'], results['std_error_with_optimal_cv'], marker='s', label='With Optimal Control Variate')
    ax2.set_xlabel('Number of Paths')
    ax2.set_ylabel('Standard Error')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # Plot variance reduction factor
    ax3.set_title('Variance Reduction Factor')
    ax3.plot(results['paths'], results['variance_reduction_factor_optimal'], marker='o')
    ax3.set_xlabel('Number of Paths')
    ax3.set_ylabel('Variance Reduction Factor')
    ax3.set_xscale('log')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_averaging_frequency():
    """Plot impact of averaging frequency (Task 4c)"""
    # Parameters
    S0 = 100
    r = 0.05
    T = 1.0
    K = 100
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    N = 252  # default daily
    M = 10000
    
    pricer = HestonAsianOption(S0, r, T, K, v0, kappa, theta, xi, rho, N, M)
    
    # Different averaging frequencies
    frequencies = [12, 52, 252]  # monthly, weekly, daily
    frequency_labels = ['Monthly (12)', 'Weekly (52)', 'Daily (252)']
    
    sigma = np.sqrt(v0)
    results = pricer.experiment_averaging_frequency(sigma, frequencies)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot option prices
    ax1.errorbar(frequency_labels, results['price_without_cv'], yerr=results['std_error_without_cv'], 
                fmt='o-', label='Without CV')
    ax1.errorbar(frequency_labels, results['price_with_cv'], yerr=results['std_error_with_cv'], 
                fmt='s-', label='With CV')
    ax1.set_xlabel('Averaging Frequency')
    ax1.set_ylabel('Option Price')
    ax1.set_title('Impact of Averaging Frequency on Option Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot variance reduction factors
    ax2.plot(frequency_labels, results['variance_reduction_factor'], 'o-')
    ax2.set_xlabel('Averaging Frequency')
    ax2.set_ylabel('Variance Reduction Factor')
    ax2.set_title('Variance Reduction Factor for Different Averaging Frequencies')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('averaging_frequency.png', dpi=300)
    plt.show()

def plot_scheme_comparison():
    """Compare Euler vs Milstein schemes"""
    # Parameters
    S0 = 100
    r = 0.05
    T = 1.0
    K = 100
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi_values = [0.1, 0.3, 0.5, 0.7, 1.0]  # Different volatility of volatility values
    rho = -0.7
    N = 252
    M = 10000
    
    # Store results
    euler_prices = []
    euler_errors = []
    euler_times = []
    milstein_prices = []
    milstein_errors = []
    milstein_times = []
    
    # Run simulations for different xi values
    for xi in xi_values:
        pricer = HestonAsianOption(S0, r, T, K, v0, kappa, theta, xi, rho, N, M)
        results = pricer.compare_schemes()
        
        euler_prices.append(results['euler']['price'])
        euler_errors.append(results['euler']['std_error'])
        euler_times.append(results['euler']['time'])
        
        milstein_prices.append(results['milstein']['price'])
        milstein_errors.append(results['milstein']['std_error'])
        milstein_times.append(results['milstein']['time'])
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot prices
    ax1.errorbar(xi_values, euler_prices, yerr=euler_errors, fmt='o-', label='Euler')
    ax1.errorbar(xi_values, milstein_prices, yerr=milstein_errors, fmt='s-', label='Milstein')
    ax1.set_xlabel('Volatility of Volatility (ξ)')
    ax1.set_ylabel('Option Price')
    ax1.set_title('Comparison of Discretization Schemes: Option Prices')
    ax1.grid(True)
    ax1.legend()
    
    # Plot standard errors
    ax2.plot(xi_values, euler_errors, 'o-', label='Euler')
    ax2.plot(xi_values, milstein_errors, 's-', label='Milstein')
    ax2.set_xlabel('Volatility of Volatility (ξ)')
    ax2.set_ylabel('Standard Error')
    ax2.set_title('Comparison of Discretization Schemes: Standard Errors')
    ax2.grid(True)
    ax2.legend()
    
    # Plot computation times
    ax3.plot(xi_values, euler_times, 'o-', label='Euler')
    ax3.plot(xi_values, milstein_times, 's-', label='Milstein')
    ax3.set_xlabel('Volatility of Volatility (ξ)')
    ax3.set_ylabel('Computation Time (s)')
    ax3.set_title('Comparison of Discretization Schemes: Computation Times')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('scheme_comparison.png', dpi=300)
    plt.show()

def plot_optimal_control_coefficient(rho_values=None,
                                     xi_values=None,
                                     K_values=None,
                                     v0_values=None):
    # Default grids
    if rho_values is None:
        rho_values = np.linspace(-0.9, 0.9, 7)
    if xi_values is None:
        xi_values = np.linspace(0.1, 1.0, 7)
    if K_values is None:
        K_values = np.linspace(80, 120, 7)
    if v0_values is None:
        v0_values = np.linspace(0.01, 0.1, 7)

    # Common fixed parameters
    S0, r, T, N, M = 100, 0.05, 1.0, 252, 20000
    # Hold one model instance for speed
    from heston_asian_option import HestonAsianOption
    pricer = HestonAsianOption(S0, r, T,
                                K=100, V0=0.04,
                                kappa=2.0, theta=0.04,
                                xi=0.3, rho=0.0,
                                N=N, M=M)

    # Helper to run CV and collect
    def run_experiment(param_name, values):
        coeffs, vr = [], []
        for val in values:
            setattr(pricer, param_name, val)
            sigma = np.sqrt(pricer.V0)
            res = pricer.control_variate_monte_carlo(pricer.generate_paths()[0], pricer.generate_gbm_paths(sigma), sigma)
            coeffs.append(res['optimal_control_coefficient'])
            vr.append(res['variance_reduction_factor_optimal'])
        return np.array(coeffs), np.array(vr)

    # Prepare subplots count
    experiments = [
        ('xi', xi_values),
        ('rho', rho_values),
        ('K', K_values),
        ('V0', v0_values)
    ]
    # Filter only those with non-empty lists
    experiments = [(name, vals) for name, vals in experiments if len(vals) > 0]
    n_plots = len(experiments)

    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 4*n_plots))
    if n_plots == 1:
        axes = np.array([axes])

    for ax_row, (param, values) in zip(axes, experiments):
        coeffs, vr = run_experiment(param, values)
        ax_row[0].plot(values, coeffs, 'o-')
        ax_row[0].set_title(f'Optimal c vs {param}')
        ax_row[0].set_xlabel(param)
        ax_row[0].set_ylabel('c*')
        ax_row[0].grid(True)

        ax_row[1].plot(values, vr, 'o-')
        ax_row[1].set_title(f'VR factor vs {param}')
        ax_row[1].set_xlabel(param)
        ax_row[1].set_ylabel('Variance Reduction Factor')
        ax_row[1].grid(True)

    plt.tight_layout()
    plt.show()

