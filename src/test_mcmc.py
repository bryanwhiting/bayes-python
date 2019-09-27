import sys
sys.path.append('/Users/bryanwhiting/Dropbox/interviews/downstream/DataScienceInterview-Bryan/src')

import numpy as np
import plotnine as g
import pandas as pd

from bryan.mcmc import MCMC

# TODO: Placeholder for unit tests

# Testing the code (would do unit tests w/more time)
k = 26
n_fake_datapoints = 10
roas = {}
cost = {}
for i in range(0, k):
    roas[i] = np.random.exponential(i * .1, n_fake_datapoints)
    cost[i] = np.random.normal(i * 10, 10, n_fake_datapoints)

# g.qplot(y = np.random.exponential(100, 100))
# q.qplot(y = np.random.gamma(shape=20, scale=15))

r_mcmc = MCMC(data=roas, niter=500)
r_mcmc.fit()
r_mcmc.estimates_as_json()['theta']
#r_mcmc.plot_theta(5)

c_mcmc = MCMC(data=cost, niter=500)
c_mcmc.fit()
t = c_mcmc.estimates_as_json()

for i in [1, 2, 5, 10, 20]:
    r_mcmc.plot_theta(i)

for i in [1, 10, 15, 20, 25]:
    c_mcmc.plot_theta(i, burnin=False)

for i in [1, 10, 100, 400]:
    c_mcmc.plot_day(i)

np.mean(t['tau2'])
c_mcmc.plot_tau2()

c_mcmc.plot_hours()
