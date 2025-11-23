# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 16:09:27 2025

@author: edoardo ferocino
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import emcee
import corner
import astropy.visualization.hist
from IPython.display import display, Math


def model(x, time):

    b, t0, A, alpha = x
    y = np.where(time < t0,
                 b,
                 b + A * np.exp(-alpha * (time - t0)))
    return y

def logLikelihood(x,time,flux,uncertainties):
    return np.sum(norm(loc = model(x,time), scale = uncertainties).logpdf(flux))

def logPrior(x):
    b, t0, A, alpha = x
    if 0 <= b <= 50 and 0 <= A <= 50 and 0 <= t0 <= 100 and np.exp(-5) <= alpha <= np.exp(5):
        return -np.log(alpha)  # log(1/alpha)
    else:
        return -np.inf

def logPosterior(x,time,flux,uncertainties):
    
    lp =  logPrior(x)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return logLikelihood(x,time,flux,uncertainties) + lp


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
plt.plot(time,model(starting_guesses,time),label='Model with starting guess')
plt.legend()
plt.grid(True)

labels = ["b","t0","A","alpha"]
ndim = 4  # number of parameters in the model
nwalkers = 24  # number of MCMC walkers
nsteps = 40000  # number of MCMC steps to take **for each walker**

np.random.seed(0)
starting_guesses = starting_guesses+1e-2*np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, logPosterior,args=[time,flux,uncertainties])
sampler.run_mcmc(starting_guesses, nsteps)
emcee_trace = sampler.get_chain()

# check the walkers if someone gets stuck
fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5*ndim), sharex=True)

for i in range(ndim):
    ax = axes[i]
    #for w in walker_indices:
    #    ax.plot(emcee_trace[:, w, i], label=f"walker {w}", alpha=0.8)
    ax.plot(emcee_trace[:, :, i], alpha=0.3)
    ax.set_ylabel(labels[i])
    ax.grid(True)
    ax.legend()
axes[-1].set_xlabel("Step number")

for i in range(ndim):
    print(f"Parameter {labels[i]}:")
    for w in range(nwalkers):
        std = np.std(emcee_trace[:, w, i])
        mean = np.mean(emcee_trace[:, w, i])
        print(f"  Walker {w+1}: std = {std:.4f}, mean = {mean:.4f}")
        

# plot running mean
walker_indices = np.linspace(0, nwalkers-1, 3, dtype=int) 
fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5*ndim), sharex=True)
for i, label in enumerate(labels):
    ax = axes[i]
    for w in walker_indices:
        param_chain = emcee_trace[:, w, i]
        running_mean = np.cumsum(param_chain) / np.arange(1, len(param_chain)+1)
        ax.plot(running_mean, label=f"walker {w}")
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend(loc="best")
axes[-1].set_xlabel("Step number")



# define the autocorr time and burn phase
tau = sampler.get_autocorr_time()
print(tau)
thin = int(np.max(tau)+1)
print('Max of autocorrelation time is',thin)
burn = 1000

# get the chain
emcee_trace = sampler.get_chain(discard=burn, thin=thin, flat=True)

# plot the flattened chain
fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(emcee_trace[:, i], "k", alpha=0.3)
    ax.set_xlim(0, len(emcee_trace))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number");

# plot the histogram
fig, axes = plt.subplots(1, ndim, figsize=(4*ndim, 4))
for i in range(ndim):
    ax = axes[i]
    astropy.visualization.hist(emcee_trace[:, i], bins="freedman",density=True, ax=ax)
    #ax.hist(emcee_trace[:, i], bins=50, color='skyblue', edgecolor='k', alpha=0.7,density=True)
    ax.set_xlabel(labels[i])
    ax.set_title(f"Histogram of {labels[i]}")
axes[0].set_ylabel("Frequency")

fig = corner.corner( emcee_trace, labels=labels,quantiles=(0.16, 0.84), levels=[0.68,0.95],titles =labels);

# Compute statistics
medians = np.median(emcee_trace, axis=0)
q5 = np.percentile(emcee_trace, 5, axis=0)
q95 = np.percentile(emcee_trace, 95, axis=0)
ul = q95-medians
ll = medians-q5
# Print results
for i, label in enumerate(labels):
    print(f"${label}: median = {medians[i]:.4f}, 5th = {q5[i]:.4f}, 95th = {q95[i]:.4f}. 90% credible region: {medians[i]:.4f}(+{ul[i]:4f}, -{ll[i]:4f})$")
    
log_posteriors = np.array([logPosterior(params, time, flux, uncertainties) for params in emcee_trace])
map_index = np.argmax(log_posteriors)
map_params = emcee_trace[map_index]
for i, label in enumerate(labels):
    print(f"MAP value for {label}: {map_params[i]:.4f}")

# show the models obtained from the posterior
models = np.array([model(params, time) for params in emcee_trace])
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

