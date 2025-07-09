import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def integrand(x, sigma):
    return x**3 * np.exp(-x**2 / (2 * sigma**2))

def monte_carlo_integration_infinite(f, sigma, L, n_samples=100000):
    # Sample points uniformly in [0, L]
    x_random = np.random.uniform(0, L, n_samples)
    
    # Evaluate the function at these points
    f_values = f(x_random, sigma)
    
    # Average value of the function
    avg_f = np.mean(f_values)
    
    # Estimate the integral
    integral_estimate = L * avg_f
    
    return integral_estimate

def analyze_monte_carlo_distribution_subplot(f, sigma, L, n_samples_list, exact_value, n_iterations=1000):
    """
    Perform repeated Monte Carlo integration for multiple sample sizes and plot distributions in subplots.

    Parameters:
    - f: function to integrate, signature f(x, sigma)
    - sigma: parameter for the integrand
    - L: upper bound for integration (finite approximation of infinity)
    - n_samples_list: list of sample sizes to test
    - n_iterations: number of repeated integrations per sample size
    - exact_value: known exact value of the integral for reference
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, n_samples in enumerate(n_samples_list):
        results = []
        for _ in range(n_iterations):
            estimate = monte_carlo_integration_infinite(f, sigma, L, n_samples)
            results.append(estimate)
        results = np.array(results)

        mean_estimate = np.mean(results)
        std_estimate = np.std(results)

        print(f"n_samples={n_samples}: Mean={mean_estimate:.6f}, Std={std_estimate:.6f}")

        ax = axes[i]
        count, bins, ignored = ax.hist(results, bins=30, density=True,
                                      color='skyblue', edgecolor='black',
                                      alpha=0.7, label='Monte Carlo estimates')

        dist = norm(mean_estimate, std_estimate)
        x = np.linspace(min(results), max(results), 1000)
        ax.plot(x, dist.pdf(x), 'r-', lw=2, label='Normal distribution fit')

        ax.axvline(x=exact_value, color='red', linestyle='--', label='Exact value')
        ax.axvline(x=mean_estimate, color='green', linestyle='-', label='Mean estimate')

        ax.set_title(f'n_samples = {n_samples}\nMean = {mean_estimate:.4f}, Std = {std_estimate:.4f}')
        ax.set_xlabel('Integral estimate')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, ls='--', alpha=0.6)

    plt.suptitle(f'Distribution of Monte Carlo Estimates for Different Sample Sizes\n(iterations={n_iterations})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Parameters
n_iterations = int(1e5)

sigma = 3
exact_value = 2 * sigma**4
print(f"Exact integral: {exact_value:.6f}")

# Different sample sizes to test
L = 6 * sigma  # upper bound to approximate infinity
sample_sizes = np.logspace(0, 6, num=50, dtype=int)  # from 1 to 1,000,000 samples

errors = []
estimates = []

for n in sample_sizes:
    estimate = monte_carlo_integration_infinite(integrand, sigma, L, n)
    error = abs(estimate - exact_value)
    errors.append(error)
    estimates.append(estimate)
    print(f"\nSamples: {n}, Estimate: {estimate:.6f}, Error: {error:.6f}")
    print(f"Estimated integral: {estimate:.6f}")
    print(f"Absolute error: {abs(estimate - exact_value):.6f}")

plt.figure(figsize=(12, 5))

# Plot 1: Computed Value vs Number of Samples
plt.subplot(1, 2, 1)
plt.plot(sample_sizes, estimates, marker='o', linestyle='-', color='orange', label='Estimated integral')
plt.axhline(y=exact_value, color='green', linestyle='--', label='Exact integral')
plt.xscale('log')
plt.xlabel('Number of samples (log scale)')
plt.ylabel('Integral value')
plt.title('Computed Integral vs Number of Samples')
plt.legend()
plt.grid(True, which="both", ls="--")


# Perform linear fit on log-log data for errors
log_samples = np.log10(sample_sizes)
log_errors = np.log10(errors)
coefficients = np.polyfit(log_samples, log_errors, 1)  # degree 1 polynomial fit
poly = np.poly1d(coefficients)

# Generate fitted values for plotting trend line
log_errors_fit = poly(log_samples)
errors_fit = 10**log_errors_fit

print(f"\nFitting equation (log-log scale): log10(error) = {coefficients[0]:.4f} * log10(n_\samples) + {coefficients[1]:.4f}")
print(f"Equivalent power-law: error = {10**coefficients[1]:.4e} * n^{coefficients[0]:.4f}")

# Plot 2: Error vs Number of Samples (log-log)
plt.subplot(1, 2, 2)
plt.loglog(sample_sizes, errors, marker='o', linestyle='-', color='b')
plt.loglog(sample_sizes, errors_fit, linestyle='--', color='red', label=f'Trend line: slope={coefficients[0]:.2f}')
plt.xlabel('Number of samples (log scale)')
plt.ylabel('Absolute error (log scale)')
plt.title('Monte Carlo Integration Error vs Number of Samples')
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.show()


# Select 4 sample sizes from your sample_sizes array or define manually
sample_sizes = [1, 2, 5, 10]

# Run the subplot analysis
analyze_monte_carlo_distribution_subplot(integrand, sigma, L, sample_sizes, exact_value, n_iterations)
