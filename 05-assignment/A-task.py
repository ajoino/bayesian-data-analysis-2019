import pickle
from pathlib import Path
import pystan
import numpy as np
np.set_printoptions(precision=2)
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')
from utils.utils import load_stan_model, fit_stan_model, \
                        plot_trace, plot_posterior, \
                        compare_distributions, roc_curve
from utils.hdi import hdi_from_book as hdi
from pprint import pprint
import json

def point_1(data):
    model = load_stan_model('05-assignment/hierarchical_model.stan', True)

    fit, summary, fit_model = fit_stan_model(model, data, iter=10000, warmup = 1000)
    the_dude_posterior = fit.extract('exp_theta')['exp_theta'][:, 3]
    mu_posterior = fit.extract('exp_mu')['exp_mu']
    random_posteriors = fit.extract('exp_theta')['exp_theta'][:, [10, 1]]
    sns.distplot(the_dude_posterior, bins=30, norm_hist=True, label='The Dude θ')
    sns.distplot(mu_posterior, bins=30, norm_hist=True, label='Group θ')
    sns.distplot(random_posteriors[:, 0], bins=30, norm_hist=True)
    sns.distplot(random_posteriors[:, 1], bins=30, norm_hist=True)
    print(f'The Dude\'s mean: {np.mean(the_dude_posterior):.2f}')
    print(f'The group mean: {np.mean(mu_posterior):.2f}')
    plt.axvline(np.mean(the_dude_posterior), color='b', lw=2, linestyle='--',label='The Dude mean')
    plt.axvline(np.mean(mu_posterior), color='r', lw=2, linestyle='--',label='Group mean')
    the_dude_hdi = hdi(the_dude_posterior, 0.95)
    mu_hdi = hdi(mu_posterior, 0.95)
    print(f'The Dude\'s HDI: {the_dude_hdi}')
    print(f'The group HDI: {mu_hdi}')
    plt.legend()
    sns.despine()
    plt.savefig('05-assignment/point_1.png')

    return fit, summary, fit_model

def point_2(fit, summary, fit_model):
    num_posterior_samples = (fit.sim['iter'] - fit.sim['warmup']) * fit.sim['chains']
    #random_new_person_index = np.random.randint(num_posterior_samples)

    num_new_people = 10000
    random_person_samples = np.zeros((num_new_people, ))
    for i in range(num_new_people):
        new_person_mu = fit.extract('mu')['mu'][np.random.randint(num_posterior_samples)]
        new_person_tau = fit.extract('tau')['tau'][np.random.randint(num_posterior_samples)]
        new_person_sigma = fit.extract('sigma')['sigma'][np.random.randint(num_posterior_samples)]

        new_person_theta = np.random.normal(new_person_mu, new_person_tau)
        sample_log_reaction_times = np.random.normal(new_person_theta, new_person_sigma, (1, ))
        #sample_reaction_times = np.exp(sample_log_reaction_times)
        random_person_samples[i] = np.exp(sample_log_reaction_times)

    random_person_hdi = hdi(random_person_samples, 0.95)
    print(f'Expected value of randomly drawn people: {np.mean(random_person_samples):.2f}')
    print(f'HDI of randomly drawn people samples: {random_person_hdi}')


    plt.figure()
    sns.distplot(random_person_samples, bins=20, norm_hist=True)
    plt.plot(random_person_hdi, [0, 0], 'k', linewidth=5)
    plt.savefig('05-assignment/point_2.png')

def point_3(fit, summary, fit_model, data):
    data_df = pd.DataFrame(data, columns=['reaction_times', 'users'])
    frequentist_mean = data_df.groupby('users').mean().to_numpy()
    group_frequentist_mean = data_df['reaction_times'].mean()
    bayesian_mean = summary[summary.index.str.contains('exp_theta')]['mean'].to_numpy()
    group_bayesian_mean = np.exp(summary.loc['mu', 'mean'])    

    print(f'Bayesian group mean:\t\t {group_bayesian_mean:.2f}')
    print(f'Frequentist group mean:\t\t {group_frequentist_mean:.2f}')
    #print(f'Difference of group means:\t {group_bayesian_mean - group_frequentist_mean:.2f}')
    #print(f'Individual bayesian means smaller than individual frequentist means:\n\t\t{np.sum(bayesian_mean[:, np.newaxis] - frequentist_mean < 0)}/{bayesian_mean.shape[0]}')

    print(frequentist_mean)
    print(bayesian_mean)
    plt.figure()
    plt.plot([group_bayesian_mean, group_bayesian_mean], [0, 34], '--k')
    for user, (user_bayesian_mean, user_frequentist_mean) in enumerate(zip(bayesian_mean, frequentist_mean)):
        plt.plot([user_bayesian_mean, user_bayesian_mean], [user, user+1], 'k')
        plt.plot([user_frequentist_mean, user_frequentist_mean], [user, user+1], ':k')
        if user != 0:
            plt.plot([200, 650], [user, user], 'k')

    plt.savefig('05-assignment/point_3.png')

    bayesian_distance = np.abs(group_bayesian_mean - bayesian_mean)[:, np.newaxis]
    print(bayesian_distance.shape)
    frequentist_distance = np.abs(group_bayesian_mean - frequentist_mean)
    print(frequentist_distance.shape)
    print(f'Individual bayesian means closer to the bayesian mean than individual frequentist means:\n\t\t{np.sum(bayesian_distance < frequentist_distance)}/34')

if __name__ == '__main__':
    with open('05-assignment/data.json') as json_file:
        data = json.load(json_file)
    output_1 = point_1(data)
    point_2(*output_1)
    point_3(*output_1, data)
