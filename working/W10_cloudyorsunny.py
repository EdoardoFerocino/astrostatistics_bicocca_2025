# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:01:26 2025

@author: edoardo ferocino
"""

import numpy as np
from matplotlib import pyplot as plt
import astroML.stats
import astropy.visualization.hist


def next_state(currentstate):
    return np.random.choice(list(T[currentstate].keys()),p=list(T[currentstate].values()))
    
def run_chain(initstate, iterations):
    currentstate = initstate
    chain = [currentstate]
    for _ in range(iterations-1):
         currentstate = next_state(currentstate)
         chain.append(currentstate)
    return np.array(chain)

def plot_hist(Burn):
    plt.figure()
    astropy.visualization.hist(Trace[Burn:], bins="freedman", histtype="step",density=True, label = 'Freedman')
    #plt.hist(Trace[Burn:],bins = np.append(np.sort(Trace[Burn:])[::100], np.max(Trace[Burn:])), density=True, histtype='step')
    plt.xlabel('p(Clear)')  
    plt.title(f'p(Clear) distribution with {Burn} samples burned')

    print(f'Values with {Burn} samples burned')
    print('Median',np.median(Trace[Burn:]))
    print('SigmaG', astroML.stats.sigmaG(Trace[Burn:]))
    print('Min/Max', min(Trace[Burn:]),max(Trace[Burn:]))

T = {'Cloudy':{'Cloudy':0.5,'Clear':0.5},'Clear':{'Clear':0.9,'Cloudy':0.1}}
Ndays = 100000
Burn = 2000
InitialWeather = 'Cloudy'

Chain = run_chain(InitialWeather,Ndays)

Days = list(range(1,Ndays+1))
IsClearDays = Chain=='Clear'
Trace = np.cumsum(IsClearDays)/Days
plt.plot(Days, Trace, label = 'Trace')
plt.axvline(x=Burn, color='red', linestyle=':', linewidth=2, label = 'Burn')
plt.xlabel('Days')           
plt.ylabel('Trace') 
plt.legend()         
plt.title('Trace over Days of p(Clear) estimate')

plot_hist(0)
plot_hist(2000)

