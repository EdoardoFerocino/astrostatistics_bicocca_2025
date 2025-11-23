# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 16:05:39 2025

@author: edoardo ferocino
"""

import numpy as np
import dynesty
import matplotlib.pyplot as plt
from scipy.stats import norm
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import astropy.visualization.hist
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
def burst(x, time):

    b, t0, A, alpha = x
    y = np.where(time < t0,
                 b,
                 b + A * np.exp(-alpha * (time - t0),dtype = np.float64))
    return y

def gauss(x, time):

    b, t0, A, sigma = x
    y = b + A * np.exp(-((time - t0)**2)/(2*sigma**2))
    return y


def logLikelihood(x,model,time,flux,uncertainties):
    return np.sum(norm(loc = model(x,time), scale = uncertainties).logpdf(flux))


def ptform_burst(u):
    b, t0, A, alpha = u
    b = 50 * b           # Uniform in [0, 50]
    t0 = 100 * t0         # Uniform in [0, 100]
    A = 50 * A           # Uniform in [0, 50]
    ln_alpha = -5 + 10 * alpha   # Uniform in [-5, 5]
    alpha = np.exp(ln_alpha)
    return np.array([b, t0, A, alpha])

def ptform_gauss(u):
    b, t0, A, sigma = u
    b = 50 * b           # Uniform in [0, 50]
    t0 = 100 * t0         # Uniform in [0, 100]
    A = 50 * A           # Uniform in [0, 50]
    ln_sigma = -2 + 4 * sigma   # Uniform in [-5, 5]
    sigma = np.exp(ln_sigma)
    return np.array([b, t0, A, sigma])

def plot(labels,sresults):
    rfig, raxes = dyplot.runplot(sresults)
    tfig, taxes = dyplot.traceplot(sresults,labels=labels)
    cfig, caxes = dyplot.cornerplot(sresults, labels=labels)

    Z = np.exp(sresults.logz[-1])
    # Resample weighted samples.
    weights = sresults.importance_weights()
    samples = sresults.samples
    samples_equal = dyfunc.resample_equal(samples, weights)

    # plot the histogram
    fig, axes = plt.subplots(1, ndim, figsize=(4*ndim, 4))
    for i in range(ndim):
        ax = axes[i]
        astropy.visualization.hist(samples_equal[:, i], bins="freedman",density=True, ax=ax)
        #ax.hist(emcee_trace[:, i], bins=50, color='skyblue', edgecolor='k', alpha=0.7,density=True)
        ax.set_xlabel(labels[i])
        ax.set_title(f"Histogram of {labels[i]}")
    axes[0].set_ylabel("Frequency")


    # Compute statistics
    medians = np.median(samples_equal, axis=0)
    q5 = np.percentile(samples_equal, 5, axis=0)
    q95 = np.percentile(samples_equal, 95, axis=0)
    ul = q95-medians
    ll = medians-q5
    # Print results
    for i, label in enumerate(labels):
        print(f"{label}: median = {medians[i]:.4f}, 5th = {q5[i]:.4f}, 95th = {q95[i]:.4f}. 90% credible region: {medians[i]:.4f}(+{ul[i]:4f}, -{ll[i]:4f})")
    
    return Z

# Load the data
data = np.load('transient.npy')
time = data[:, 0]
flux = data[:, 1]
uncertainties = data[:, 2]

# Create the errorbar plot with the model and initial guess
plt.figure(figsize=(8, 5))
plt.errorbar(time, flux, yerr=uncertainties, fmt='o', 
             ecolor='gray', capsize=3, label='Flux with uncertainties')
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title('Flux vs Time with Uncertainties')

starting_guesses = np.array([10,50,5,0.12])
plt.plot(time,burst(starting_guesses,time),label='Model with starting guess')
plt.legend()
plt.grid(True)

labels = ["b","t0","A","alpha"]
ndim = 4  # number of parameters in the model

# "Static" nested sampling.
sampler = dynesty.NestedSampler(logLikelihood, ptform_burst, ndim,logl_args=(burst,time, flux, uncertainties))
sampler.run_nested()
sresults = sampler.results

Z_burst = plot(labels,sresults)

# Create the errorbar plot with the model and initial guess
plt.figure(figsize=(8, 5))
plt.errorbar(time, flux, yerr=uncertainties, fmt='o', 
             ecolor='gray', capsize=3, label='Flux with uncertainties')
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title('Flux vs Time with Uncertainties')

starting_guesses = np.array([10,52,4,6])
plt.plot(time,gauss(starting_guesses,time),label='Model with starting guess')
plt.legend()
plt.grid(True)

labels = ["b","t0","A","sigma"]
# "Static" nested sampling.
sampler = dynesty.NestedSampler(logLikelihood, ptform_gauss, ndim,logl_args=(gauss,time, flux, uncertainties))
sampler.run_nested()
sresults = sampler.results

Z_gauss = plot(labels,sresults)   
    
print('The Bayes factor of burst vs gauss is: ', np.log(Z_burst/Z_gauss))


weights = sresults.importance_weights()
samples = sresults.samples
samples_equal = dyfunc.resample_equal(samples, weights)


models = np.array([gauss(params, time) for params in samples_equal])
random_indices = np.random.choice(models.shape[0], size=100, replace=False)

plt.figure(figsize=(8, 5))

# Plot 100 random model curves with transparency
for idx in random_indices:
    plt.plot(time, models[idx, :], color='C0', alpha=0.1)

plt.errorbar(time, flux, yerr=uncertainties, fmt='o', 
             ecolor='gray', capsize=3, label='Flux with uncertainties')

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("100 Random Model Curves and Data")
plt.legend()

