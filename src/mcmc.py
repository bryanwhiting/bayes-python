import random

import numpy as np
import pandas as pd
import plotnine as g

def trace_plot(y, title):
    p = (g.qplot(y = y) +
            g.geom_line() +
            g.ggtitle(title)
        )
    return p

class MCMC:
    """A class to execute a Gibbs Sampler for MCMC"""

    def __init__(self, data, niter=500):
        # data = dictionary with key=hour and value = past values
        self.data = data
        # ni is the count of data points for a given hour
        # normally, this will be the same across every hour in Campaign
        # But if you only run a campaign at certain hours on certain days,
        # then this would differ across hours
        self.ni = [len(self.data[k]) for k, v in self.data.items()]

        # number of hours in a day. Add 1 index at before time=0 and 1 index
        # after time = 0 for the model to work
        self.k = len(self.data.keys())
        self.niter = niter

        # Hyperpriors
        # Distribution of the variance of theta: sigma^2
        self.a_sig = 3
        self.b_sig = 1
        # Distribution of the variance of theta i minus 1: tau
        self.a_tau = 3
        self.b_tau = 1

        # the sampler's estimates aren't reliable until they've stabilized
        # This is a hyperparameter that I've fixed at 10%
        self.burnin = int(self.niter * .1)

        # assumes theta starts at 1
        self.theta = np.ones((self.niter, self.k))
        self.sig2 = np.array([1.0] * self.niter)
        self.tau2 = np.array([1.0] * self.niter)

    def _theta_sample(self, i):
        """Sampler for theta"""
        # I noticed the sampler wasn't casting a wide-enough net, so I had to increase
        # the variance (multiply by 10)
        tau2prev = self.tau2[i - 1] * 10
        sig2prev = self.sig2[i - 1]

        # estimate theta_
        for j in range(0, self.k):
            if j == 0:
                # value for theta 0 (invented data point)
                mu = self.theta[i - 1, (self.k - 1)] + self.theta[i - 1, j + 1]
            elif j == self.k - 1:
                # value for theta k (invented data point)
                # + self.theta[i - 1, 0] = 0th point
                mu = self.theta[i, j - 1] + self.theta[i - 1, 0]
            else:
                # Add the prior row's estimate with the subsequent row's estimate
                mu = self.theta[i, j - 1] + self.theta[i - 1, j + 1]

            denom = (self.ni[j] * tau2prev + 2 * sig2prev)
            mustar = (np.mean(self.data[j]) * self.ni[j] * tau2prev + mu * sig2prev) / denom
            sigstar = (tau2prev * sig2prev) / denom
            theta_sample = np.random.normal(mustar, np.sqrt(sigstar), 1)
            if theta_sample < 0:
                theta_sample = 0

            self.theta[i, j] = theta_sample

    def _update_ssq(self, i):
        """Update sum of squares (ssq) for the model after fitting theta"""
        self.ssq = 0
        for j in range(0, self.k):
            self.ssq = self.ssq + np.sum((self.data[j] - self.theta[i, j]) ** 2)

    def _sig2_sample(self, i):
        """Sampler for sig2"""
        s_shape = self.a_sig + sum(self.ni) * 0.5
        s_scale = (1/self.b_sig + self.ssq/2) ** (-1)
        assert s_scale > 0, print(self.b_sig, self.ssq)
        denom = np.random.gamma(shape=s_shape,
                                scale=s_scale,
                                size=1)
        sig2_sample = 1.0/denom
        if sig2_sample <= 0:
            sig2_sample = 0.001

        self.sig2[i] = sig2_sample

    def _tau2_sample(self, i):
        """Sampler for tau2"""

        # Calculate sstau for tau2
        sstau = np.sum((self.theta[i,] - self.theta[i - 1,])**2)

        #generate tau2
        t_shape = self.a_tau + (self.k * 0.5)
        t_scale = (1/self.b_tau + .5 * sstau)**(-1)
        denom = np.random.gamma(shape=t_shape,
                                scale=t_scale,
                                size=1)
        tau2_sample = 1.0/denom
        if tau2_sample <= 0:
            tau2_sample = self.tau[i - 1]
        self.tau2[i] = tau2_sample

    def fit(self):
        """Return the MCMC for input parameters"""
        for i in range(1, self.niter):
            self._theta_sample(i)
            self._update_ssq(i)
            self._sig2_sample(i)
            self._tau2_sample(i)

        self.results = {'theta': self.theta,
                        'sig2': self.sig2,
                        'tau2': self.tau2}
        self.mu = {i: np.mean(self.results['theta'][self.burnin:, i]) for i in range(0, self.k)}

    def estimates_as_json(self):
        return self.results

    def estimates_as_json_noburn(self):
        return {k: v[self.burnin:] for k, v in self.results.items()}

    def mu_theta(self):
        return self.mu

    def plot_theta(self, i, burnin=True):
        y = self.results['theta'][:, i]
        if not burnin:
            y = y[self.burnin:]

        p = (trace_plot(y = y, title=f'theta {i}') +
                g.geom_hline(yintercept = np.mean(self.data[i]), color='blue') +
                g.geom_hline(yintercept = np.mean(self.mu[i]), color='red') +
                g.geom_line()
                )
        display(p)

    def plot_tau2(self, burnin=True):
        y = self.results['tau2']
        if burnin:
            y = y[self.burnin:]
        p = trace_plot(y = y, title='tau2')
        display(p)

    def plot_sig2(self, burnin=True):
        y = self.results['sig2']
        if burnin:
            y = y[self.burnin:]
        p = trace_plot(y = y, title='sig2')
        display(p)

    def plot_day(self, i):
        t = self.results['theta']
        p = g.qplot(y = t[i, ]) + g.geom_line() + g.labs(title=f'day {i}', x='hour', y='theta')
        display(p)

    def plot_hours(self, burnin=True):
        """Plot mean for the hour and error bars"""
        dat = pd.DataFrame(self.results['theta']).melt()
        if not burnin:
            dat = dat.loc[dat.index > self.burnin]
        dat.columns = ['hour', 'estimate']
        # https://gist.github.com/HenrikEckermann/1d334a44f61349ac71f0e235b3443a69
        p = (g.ggplot(dat, g.aes(x='hour', y='estimate')) +
            #g.stat_summary(fun_data = np.max, geom = 'point', fill='blue') +
            #g.stat_summary(fun_data = np.min, geom = 'point', fill='blue') +
            g.stat_summary(fun_data = 'mean_sdl', fun_args = {'mult':1}, geom = 'errorbar') +
            g.stat_summary(fun_y = np.mean, geom = 'point', fill = 'red')
        )
        display(p)
