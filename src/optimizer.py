import sys
# used when coding in Atom
# sys.path.append('/Users/bryanwhiting/Dropbox/interviews/downstream/DataScienceInterview-Bryan/src')
import numpy as np
import pandas as pd
import plotnine as g

from bryan.mcmc import MCMC
from simulation.campaign import Campaign

# TODO:
# in the future, I'd like to take unused budget from prior hours and use them later in the day

class OptimizeBudget:
    def __init__(self, quarterly_budget, campaign_duration, model_type, mcmc_niter=100):
        self.quarterly_budget = quarterly_budget
        self.campaign_duration = campaign_duration
        self.mcmc_niter = mcmc_niter
        self.model_type = model_type


    @staticmethod
    def convert_data_to_dict(df_campaign):
        """
        the model i've chosen builds a time-dependent markov chain to estimate
        roas and cost in each hour. In order for the markove chain to start,
        it needs an initial value. So I make up an hour 0 and re-index all other
        hours to be from 1 to 24. I also need fake data to end the markov chain
        That's why there's a 25. This makes 26 hours in total
        """

        roas = {}
        cost = {}
        # hour 0 becomes hour 1
        for hour in range(24):
            hr_dat = df_campaign.query(f'hour == {hour}')[['cost', 'roas']]
            roas[hour + 1] = np.array(hr_dat['roas'])
            cost[hour + 1] = np.array(hr_dat['cost'])

        # initialize fake data for 0 and 25
        # only taking 5 data points makes for uninformative data that
        # the model estiamtes can override. In a way, creating fake data is
        # like creating a prior.
        # assume a default roas value of 0.2 and a cost of 100 with high variance.
        for hour in [0, 25]:
            roas[hour] = np.random.exponential(scale=.5, size=5)
            cost[hour] = [max(c, 0) for c in np.random.normal(loc=100, scale=1000, size=5)]

        return cost, roas

    def model_mus(self, cost, roas, model_type='bayes'):
        """Now that the data are gathered, we can model them. The goal of modeling
        is to get the best estimate of what cost and roas will be during each
        hour on the next day.

        This code will model the data using my Bayesian MCMC algorithm. It
        also returns just the average of the last 10 days, to compare if
        my model is better than just taking a moving average
        """
        # here is where the model belongs
        mu = {}

        if model_type == 'bayes':
            data = {'cost': cost, 'roas': roas}
            for k in data.keys():
                # Model the mean roas/cost for each hour
                mcmc = MCMC(data = data[k], niter = self.mcmc_niter)
                mcmc.fit()
                # returns a dictionary with a modeled estimate of cost/roas for each hour
                mu[k] = mcmc.mu_theta()
        else:
            # Compare to just the last 10 points of data
            # taking the running average [-10:] will adjust for non-stationarity
            mu['cost'] = {k: np.mean(v[-10:]) for k, v in cost.items()}
            mu['roas'] = {k: np.mean(v[-10:]) for k, v in roas.items()}

        return mu

    @staticmethod
    def assign_hourly_budget(daily_budget, mu, default_alloc=0.02, cost_mult=1.10):
        """Daily budget is the max I want to spend on a day.

        mu: is the expected cost or roas the following day
        default_alloc: for learning, we always want to allocate some budget. You have to
        pay to get data on cost and roas, due to non-stationarity risks. This value cannot
        be more than 1/24 (roughly 4.1%)
        cost_mult: mu['cost'] has the estimated cost for the following day. If this is
            an under prediction, the profitable hours might not get enough budget
            Therefore, multiply cost by cost_mult to enable more (or less) budget
            to the top hours

        Budget allocation:
            - For data learning, make sure that all rows get some allocation
            - Identify rows with highest roas. Allocate as much budget
            that estimated cost will allow. Add 10% in case estimated cost is under.
            - Continue allocating budget in a waterfall pattern, where hours with
            highest roas get the most budget.
            - don't provide any budget if roas < 0.8. (0.8 is arbitrary threshold)
            - any remaining budget gets distributed among

        FUTURE TODO (build into simulator):
            - Whatever budget isn't used in prior hours to subsequent hours
            - Identify the optimal minimum bound of roas to consider
        """
        avg_by_hour = pd.DataFrame(mu)
        # filter to just real hours for budget allocation
        avg_by_hour = avg_by_hour[~avg_by_hour.index.isin([0, 25])].reset_index(drop=True)
        avg_by_hour['hour'] = range(24)

        # Define budget allocaiton. ----
        if default_alloc > 1/24:
            print(f'Default allocation is {default_alloc} > 1/24. Cannot allocate more than 1/24th of budget to a given hour.' )
            default_alloc = 1/24

        # allocate default budget
        avg_by_hour['budget_alloc'] = daily_budget * default_alloc

        # order by roas
        avg_by_hour.sort_values(['roas', 'cost'], ascending=False, inplace=True)
        # Allocate remaining budget as estimated cost will allow
        remaining_budget = daily_budget - avg_by_hour['budget_alloc'].sum()
        for i, row in avg_by_hour.iterrows():
            if row['roas'] > 0.8:
                cost = row['cost'] * cost_mult # increase in case of underprediction
                if remaining_budget > cost:
                    add_budget = cost
                else:
                    add_budget = remaining_budget
                # add budget to the hour
                avg_by_hour.loc[i, 'budget_alloc'] += add_budget
                remaining_budget -= add_budget
                if remaining_budget <= 0:
                    break

        # if reamining-budget isn't finished, distribute among top 3 hours
        if remaining_budget > 0:
            distribution = remaining_budget / 3
            for i in range(0, 3):
                avg_by_hour.loc[i, 'budget_alloc'] += distribution
                remaining_budget -= distribution

        # sort back into hours 0:23 and return allocated budgets in a list
        avg_by_hour.sort_values(['hour'], ascending=True, inplace=True)
        hourly_budget_alloc = avg_by_hour['budget_alloc'].tolist()
        assert len(hourly_budget_alloc) == 24
        tot_alloc = np.sum(hourly_budget_alloc)
        # warning if what's allocated is greater than the allowed daily budget
        # multiply by 1.02 to account for rounding error
        if tot_alloc > daily_budget*1.02:
            print(f'WARNING: budget allocated is tot_alloc: {tot_alloc} vs {daily_budget}: hourly values: {hourly_budget_alloc}')
        return hourly_budget_alloc

    def execute_campaign(self):

        campaign = Campaign()
        estimates = {}
        budget_alloc = {}
        total_return = {}
        starting_budget = self.quarterly_budget
        quarterly_budget = self.quarterly_budget
        campaign_duration = self.campaign_duration

        for d in range(self.campaign_duration):
            # for first 5 days, collect data. After that, allocate budget according
            # to best cost and roas
            if d < 5:
                # uniformly distribute the budget across all 24 hours
                used_budget = quarterly_budget * 0.01
                hourly_budgets = [used_budget/24] * 24
                campaign.run_campaign_for_single_day(hourly_budgets)
                cost, roas = self.convert_data_to_dict(df_campaign=campaign.get_data_as_df())
                mu = self.model_mus(cost, roas, model_type='mean')
            else:
                # dynamically allocate the budget. See function definitions for intuition
                # convert dataframe into dictionaries.
                cost, roas = self.convert_data_to_dict(df_campaign=campaign.get_data_as_df())
                # model cost and roas
                mu = self.model_mus(cost, roas, model_type=self.model_type)
                if d < 20:
                    # for days 5 to 20, start distributing to highest roas, but still
                    # allocating across other hours
                    used_budget = quarterly_budget * 0.05
                    hourly_budgets = self.assign_hourly_budget(daily_budget=used_budget,
                                                               mu=mu,
                                                               default_alloc=0.03,
                                                               cost_mult=1.20)
                    campaign.run_campaign_for_single_day(hourly_budgets)
                else:
                    # for the remainder of the days, split across campaign_duration
                    used_budget = quarterly_budget/(campaign_duration - d)
                    # for days 21+, distribute more to highest roas
                    hourly_budgets = self.assign_hourly_budget(daily_budget=used_budget,
                                                          mu=mu,
                                                          default_alloc=0.005,
                                                          cost_mult=1.10)
                    campaign.run_campaign_for_single_day(hourly_budgets)

            # quarterly_budget -= used_budget
            # remaining budget is starting budget minus how much you've spent
            quarterly_budget = starting_budget - campaign.get_data_as_df().cost.sum()
            estimates[d] = mu
            # budget alloc is saved in the dataframe
            print('Day:', d, 'Budget allocated today:', used_budget, 'Remaining budget:', quarterly_budget)

        self.campaign = campaign
        self.estimates = estimates

    def results(self):
        return self.campaign, self.estimates
