import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from time import time

class HestonAsianOption:
    def __init__(self, S0, r, T, K, V0, kappa, theta, xi, rho, N, M, scheme='euler'):
        self.S0 = S0
        self.r = r
        self.T = T
        self.K = K
        self.V0 = V0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.N = N
        self.M = M
        self.dt = T / N
        self.scheme = scheme
        
    def generate_paths(self):
        """Generate stock price and variance paths using either Euler or Milstein discretization"""       
        
        S = np.zeros((self.M, self.N + 1))
        v = np.zeros((self.M, self.N + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.V0

        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        
        # Correlated Brownian increments
        z1 = np.random.normal(0, 1, (self.M, self.N))
        self.z1 = z1  # Store for GBM paths
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, (self.M, self.N))
        
        # Simulate paths
        for t in range(self.N):
            # Non negative variance (truncation scheme)
            v[:, t] = np.maximum(v[:, t], 0)

            # Update variance 
            if self.scheme == 'euler':
                v[:, t+1] = v[:, t] + self.kappa * (self.theta - v[:, t]) * dt + self.xi * np.sqrt(v[:, t]) * sqrt_dt * z2[:, t]
            elif self.scheme == 'milstein':
                v[:, t+1] = v[:, t] + self.kappa * (self.theta - v[:, t]) * dt + self.xi * np.sqrt(v[:, t]) * sqrt_dt * z2[:, t] + \
                            0.25 * self.xi**2 * dt * (z2[:, t]**2 - 1)
            # Update stock price
            S[:, t+1] = S[:, t] * np.exp((self.r - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t]) * sqrt_dt * z1[:, t])
            
        
        return S, v

    def generate_gbm_paths(self, sigma):
        """Generate GBM paths using the same random numbers as Heston paths"""
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        
        # Initialize array
        S_gbm = np.zeros((self.M, self.N + 1))
        
        # Set initial value
        S_gbm[:, 0] = self.S0
        
        # Use the same random numbers as in Heston simulation
        for t in range(self.N):
            S_gbm[:, t+1] = S_gbm[:, t] * np.exp((self.r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * self.z1[:, t])

        return S_gbm
    
    def price_arithmetic_asian_option(self):
        """Price arithmetic average Asian call option"""
        start_time = time()
        
        S, _ = self.generate_paths()
        
        # Calculate arithmetic average for each path
        arith_avg = np.mean(S[:, 1:], axis=1)  # Exclude initial price
        
        # Calculate payoffs
        payoffs = np.maximum(arith_avg - self.K, 0)
        
        # Calculate option price
        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(self.M)
        
        end_time = time()
        computation_time = end_time - start_time
        
        return option_price, std_error, computation_time

    def analytical_geometric_asian_price(self, sigma):
        """Task 2: Analytical pricing formula for discrete geometric-average Asian call under BS model (Kemna & Vorst)"""
        N = self.N
        # use the simpler Kemna & Vorst adjustment:
        # sigma_tilde = sigma * sqrt((2N+1)/(6(N+1)))
        sigma_tilde = sigma * np.sqrt((2*N + 1) / (6*(N + 1)))
        # r_tilde = (r - 0.5*sigma^2) + 0.5*sigma_tilde^2
        r_tilde = (self.r - 0.5*sigma**2) + 0.5*sigma_tilde**2
        # d1 and d2 as per formula
        d1 = (np.log(self.S0/self.K) + (r_tilde + 0.5*sigma_tilde**2)*self.T) / (sigma_tilde * np.sqrt(self.T))
        d2 = (np.log(self.S0/self.K) + (r_tilde - 0.5*sigma_tilde**2)*self.T) / (sigma_tilde * np.sqrt(self.T))
        # closed-form geometric-Asian price
        price = (self.S0 * np.exp((r_tilde - self.r)*self.T) * norm.cdf(d1)
                 - self.K * np.exp(-self.r*self.T) * norm.cdf(d2))
        return price
    # def analytical_geometric_asian_price(self, σ):
    #     N = self.N; r = self.r; S0 = self.S0; K = self.K; T = self.T
    #     # correct discrete Kemna–Vorst
    #     σ_tilde = σ * np.sqrt((2*N+1)*(N+1)/(6*N**2))
    #     r_tilde = (r - 0.5*σ**2)*(N+1)/(2*N) + 0.5*σ_tilde**2

    #     d1 = (np.log(S0/K) + (r_tilde+0.5*σ_tilde**2)*T)/(σ_tilde*np.sqrt(T))
    #     d2 = d1 - σ_tilde*np.sqrt(T)
    #     return ( S0 * np.exp((r_tilde-r)*T) * norm.cdf(d1)
    #            - K  * np.exp(-r*T)            * norm.cdf(d2) )

    def verify_with_gbm(self, σ=None):
        if σ is None: σ = np.sqrt(self.V0)
        # 1) MC arithmetic under GBM
        orig = self.xi; self.xi=0.0
        C_MC, se, _ = self.price_arithmetic_asian_option()
        self.xi=orig

        # 2) corrected geometric lower bound
        C_geo = self.analytical_geometric_asian_price(σ)

        # 3) corrected TW: include covariances
        dt = self.T/self.N
        times = np.linspace(dt, self.T, self.N)
        # build full Cov matrix
        Ti,Tj = np.meshgrid(times, times)
        Cov = (self.S0**2) * np.exp(self.r*(Ti+Tj)) * (np.exp(σ**2 * np.minimum(Ti,Tj)) - 1)
        varA = Cov.sum() / (self.N**2)
        mA   = self.S0 * np.mean(np.exp(self.r*times))
        μA   = np.log(mA**2/np.sqrt(mA**2+varA))
        σA   = np.sqrt(np.log(1 + varA/mA**2))
        d1   = (μA - np.log(self.K))/σA + σA
        d2   = d1 - σA
        C_TW = np.exp(-self.r*self.T)*(np.exp(μA+0.5*σA**2)*norm.cdf(d1) - self.K*norm.cdf(d2))

        return {
          'mc_arithmetic_price': C_MC,
          'mc_arithmetic_std_error': se,
          'analytic_geometric_price': C_geo,
          'turnbull_wakeman_approx': C_TW
         }
    
    # def verify_with_gbm(self, sigma=None, analytic_arith=None):
    #     """
    #     Verify the Monte Carlo arithmetic-Asian implementation by collapsing Heston to GBM (ξ=0) and comparing
    #     - MC arithmetic-Asian price under GBM
    #     - Analytic geometric-Asian price (closed-form lower bound)
    #     - Turnbull Wakeman lognormal approximation (benchmark for arithmetic-Asian)

    #     Parameters:
    #     sigma : float, optional
    #         The constant volatility of the GBM. If None, uses sqrt(V0) so that the variance process v_t ≈ V0.
    #     analytic_arith : float or None
    #         If provided, uses this value as an external benchmark for the arithmetic-Asian price
    #         instead of computing the Turnbull Wakeman approximation.

    #     Returns:
    #     dict with keys:
    #         mc_arithmetic_price, mc_arithmetic_std_error
    #         analytic_geometric_price
    #         turnbull_wakeman_approx
    #     """
    #     # 1. Choose GBM volatility: either user-supplied or sqrt(initial variance)
    #     if sigma is None:
    #         sigma = np.sqrt(self.V0)

    #     # Temporarily set xi=0 to collapse Heston to constant-vol GBM
    #     orig_xi = self.xi
    #     self.xi = 0.0

    #     # Monte Carlo price of arithmetic-Asian under GBM
    #     mc_price, mc_std_error, _ = self.price_arithmetic_asian_option()

    #     # Restore original xi
    #     self.xi = orig_xi

    #     # 2. Analytic geometric-Asian price (closed-form lower bound)
    #     geo_price = self.analytical_geometric_asian_price(sigma)

    #     # 3. Turnbull–Wakeman log-normal approximation for arithmetic-Asian (benchmark)
    #     if analytic_arith is None:
    #         # monitoring times t_i = i*dt, i=1..N
    #         dt = self.T / self.N
    #         times = np.linspace(dt, self.T, self.N)
    #         # expected arithmetic average and variance under GBM
    #         exp_rt = np.exp((self.r - 0.5*sigma**2) * times)
    #         meanA = (self.S0 / self.N) * np.sum(exp_rt)
    #         second_moment = (self.S0**2 / self.N**2) * np.sum(
    #             np.exp(2*(self.r) * times) * (np.exp(sigma**2 * times) - 1)
    #         ) + meanA**2
    #         varA = second_moment - meanA**2
    #         # lognormal parameters for A
    #         muA = np.log(meanA**2 / np.sqrt(varA + meanA**2))
    #         sigmaA = np.sqrt(np.log(1 + varA / meanA**2))
    #         # Black–Scholes on average A
    #         from scipy.stats import norm
    #         d1 = (muA - np.log(self.K) + sigmaA**2) / sigmaA
    #         d2 = d1 - sigmaA
    #         C_TW = np.exp(-self.r * self.T) * (
    #             np.exp(muA + 0.5 * sigmaA**2) * norm.cdf(d1)
    #             - self.K * norm.cdf(d2)
    #         )
    #         tw_price = C_TW
    #     else:
    #         tw_price = analytic_arith

    #     return {
    #         'mc_arithmetic_price': mc_price,
    #         'mc_arithmetic_std_error': mc_std_error,
    #         'analytic_geometric_price': geo_price,
    #         'turnbull_wakeman_approx': tw_price
    #     }


    def compare_schemes(self):
        """Compare Euler and Milstein schemes"""
        # Save original scheme
        original_scheme = self.scheme
        
        # Price using Euler scheme
        self.scheme = 'euler'
        euler_start_time = time()
        euler_price, euler_std_error, _ = self.price_arithmetic_asian_option()
        euler_time = time() - euler_start_time
        
        # Price using Milstein scheme
        self.scheme = 'milstein'
        milstein_start_time = time()
        milstein_price, milstein_std_error, _ = self.price_arithmetic_asian_option()
        milstein_time = time() - milstein_start_time
        
        # Restore original scheme
        self.scheme = original_scheme
        
        return {
            'euler': {
                'price': euler_price,
                'std_error': euler_std_error,
                'time': euler_time
            },
            'milstein': {
                'price': milstein_price,
                'std_error': milstein_std_error,
                'time': milstein_time
            }
        }

    def control_variate_monte_carlo(self, S_heston, S_gbm, sigma, c=1.0):
        """Task 3: Implement control variate Monte Carlo simulation"""
        start_time = time()

        # Calculate arithmetic average for Heston paths
        arith_avg_heston = np.mean(S_heston[:, 1:], axis=1)
        payoffs_heston = np.maximum(arith_avg_heston - self.K, 0)
        
        # Calculate geometric average for GBM paths
        geo_avg_gbm = np.exp(np.mean(np.log(S_gbm[:, 1:]), axis=1))
        payoffs_gbm = np.maximum(geo_avg_gbm - self.K, 0)
        
        # Calculate analytical price for geometric Asian option
        geo_analytical_price = self.analytical_geometric_asian_price(sigma)
        
        # Apply control variate
        cv_payoffs = payoffs_heston + c * (geo_analytical_price - np.exp(-self.r * self.T) * payoffs_gbm)
        
        # Calculate option prices
        price_without_cv = np.exp(-self.r * self.T) * np.mean(payoffs_heston)
        price_with_cv = np.mean(cv_payoffs)
        
        # Calculate standard errors
        std_error_without_cv = np.exp(-self.r * self.T) * np.std(payoffs_heston) / np.sqrt(self.M)
        std_error_with_cv = np.std(cv_payoffs) / np.sqrt(self.M)
        
        # Calculate optimal control variate coefficient
        cov_matrix = np.cov(payoffs_heston, payoffs_gbm)
        optimal_c = cov_matrix[0, 1] / np.var(payoffs_gbm)
        
        # Apply optimal control variate
        optimal_cv_payoffs = payoffs_heston + optimal_c * (geo_analytical_price - np.exp(-self.r * self.T) * payoffs_gbm)
        price_with_optimal_cv = np.mean(optimal_cv_payoffs)
        std_error_with_optimal_cv = np.std(optimal_cv_payoffs) / np.sqrt(self.M)

        var_plain = np.var(np.exp(-self.r * self.T) * payoffs_heston)
        var_cv = np.var(cv_payoffs)
        var_optimal_cv = np.var(optimal_cv_payoffs)
        
        # Calculate variance reduction factors
        vr_factor_c1 = (std_error_without_cv / std_error_with_cv) ** 2
        vr_factor_optimal = (std_error_without_cv / std_error_with_optimal_cv) ** 2
        
        end_time = time()
        computation_time = end_time - start_time
        
        results = {
            'price_without_cv': price_without_cv,
            'std_error_without_cv': std_error_without_cv,
            'price_with_cv': price_with_cv,
            'std_error_with_cv': std_error_with_cv,
            'price_with_optimal_cv': price_with_optimal_cv,
            'std_error_with_optimal_cv': std_error_with_optimal_cv,
            'geo_analytical_price': geo_analytical_price,
            'control_coefficient_c': c,
            'optimal_control_coefficient': optimal_c,
            'variance_plain': var_plain,
            'variance_cv': var_cv,
            'variance_optimal_cv': var_optimal_cv,
            'variance_reduction_factor_c1': vr_factor_c1,
            'variance_reduction_factor_optimal': vr_factor_optimal,
            'computation_time': computation_time
        }
        
        return results
    
    def experiment_varying_paths(self, sigma, path_counts):
            """Task 4a: Evaluate performance with varying numbers of simulation paths"""
            results = {
                'paths': path_counts,
                'price_without_cv': [],
                'std_error_without_cv': [],
                'price_with_cv': [],
                'std_error_with_cv': [],
                'price_with_optimal_cv': [],
                'std_error_with_optimal_cv': [],
                'variance_reduction_factor_c1': [],
                'variance_reduction_factor_optimal': [],
                'computation_time': []
            }
            
            original_M = self.M
            
            for M in path_counts:
                self.M = M
                cv_results = self.control_variate_monte_carlo(self.generate_paths()[0], self.generate_gbm_paths(sigma), sigma)
                
                results['price_without_cv'].append(cv_results['price_without_cv'])
                results['std_error_without_cv'].append(cv_results['std_error_without_cv'])
                results['price_with_cv'].append(cv_results['price_with_cv'])
                results['std_error_with_cv'].append(cv_results['std_error_with_cv'])
                results['price_with_optimal_cv'].append(cv_results['price_with_optimal_cv'])
                results['std_error_with_optimal_cv'].append(cv_results['std_error_with_optimal_cv'])
                results['variance_reduction_factor_c1'].append(cv_results['variance_reduction_factor_c1'])
                results['variance_reduction_factor_optimal'].append(cv_results['variance_reduction_factor_optimal'])
                results['computation_time'].append(cv_results['computation_time'])
            
            # Restore original M
            self.M = original_M
            
            return results

    def experiment_averaging_frequency(self, sigma, frequencies):
        """Task 4c: Explore impact of averaging frequency"""
        results = {
            'frequency': frequencies,
            'price_without_cv': [],
            'std_error_without_cv': [],
            'price_with_cv': [],
            'std_error_with_cv': [],
            'variance_reduction_factor': []
        }
        
        original_N = self.N
        
        for N in frequencies:
            self.N = N
            self.dt = self.T / N
            
            cv_results = self.control_variate_monte_carlo(self.generate_paths()[0], self.generate_gbm_paths(sigma), sigma)
            
            results['price_without_cv'].append(cv_results['price_without_cv'])
            results['std_error_without_cv'].append(cv_results['std_error_without_cv'])
            results['price_with_cv'].append(cv_results['price_with_optimal_cv'])
            results['std_error_with_cv'].append(cv_results['std_error_with_optimal_cv'])
            results['variance_reduction_factor'].append(cv_results['variance_reduction_factor_optimal'])
        
        # Restore original N
        self.N = original_N
        self.dt = self.T / self.N
        
        return results

  