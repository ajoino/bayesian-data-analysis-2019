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

def plot_dist_mean_hdi(data, label='', color='k'):
    data_hdi = hdi(data)
    sns.distplot(data, bins=20, norm_hist=True, label=f'{label} posterior')
    plt.axvline(np.mean(data), color=color, lw=2, label=f'{label} mean')
    plt.axvline(data_hdi[0], color=color, lw=2, linestyle='--', label=f'{label} 95% HDI')
    plt.axvline(data_hdi[1], color=color, lw=2, linestyle='--')

def plot_data(data_df):
    sns.distplot(np.log(data_df[data_df.is_datum_child == 1]['reaction_times']), bins=20, norm_hist=True, label='Children')
    sns.distplot(np.log(data_df[data_df.is_datum_child == 0]['reaction_times']), bins=20, norm_hist=True, label='Adults')

    adult_reaction_mean, child_reaction_mean = data_df.groupby('is_datum_child').mean()['reaction_times'].to_numpy()
    adult_reaction_median, child_reaction_median = data_df.groupby('is_datum_child').median()['reaction_times'].to_numpy()
    print(adult_reaction_mean, adult_reaction_median)
    print(child_reaction_mean, child_reaction_median)

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Log reaction time')
    plt.ylabel('Probability density')

    plt.savefig('06-assignment/normal/reaction_times.png')

def point_1(fit, fit_summary):
    adult_posterior_mu = fit.extract('mu[1]')['mu[1]']
    child_posterior_correction = fit.extract('mu[2]')['mu[2]']
    child_posterior_mu = adult_posterior_mu + child_posterior_correction

    adult_mean = np.mean(adult_posterior_mu)
    child_mean = np.mean(child_posterior_mu)
    diff_mean = np.mean(child_posterior_correction)

    adult_hdi = hdi(adult_posterior_mu, 0.95)
    child_hdi = hdi(child_posterior_mu, 0.95)
    diff_hdi = hdi(child_posterior_correction, 0.95)

    child_correction_effect_size = child_posterior_correction / np.std(child_posterior_correction)

    print(f'Adult mean: {adult_mean:.2f}, HDI: {adult_hdi}')
    print(f'Child mean: {child_mean:.2f}, HDI: {child_hdi}')
    print(f'Difference mean: {diff_mean:.2f}, HDI: {diff_hdi}')
    print(f'Effect mean: {np.mean(child_correction_effect_size):.2f}, HDI: {hdi(child_correction_effect_size)}')

    plt.figure()
    sns.distplot(adult_posterior_mu, bins=20, norm_hist=True, label='Adult log posterior μ')
    sns.distplot(child_posterior_mu, bins=20, norm_hist=True, label='Child log posterior μ')
    plt.axvline(adult_mean, color='b', lw=2, label='Adult mean')
    plt.axvline(adult_hdi[0], color='b', lw=2, linestyle='--', label='Adult 95% HDI')
    plt.axvline(adult_hdi[1], color='b', lw=2, linestyle='--')
    plt.axvline(child_mean, color='r', lw=2, label='Child mean')
    plt.axvline(child_hdi[0], color='r', lw=2, linestyle='--', label='Child 95% HDI')
    plt.axvline(child_hdi[1], color='r', lw=2, linestyle='--')

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Posterior log reaction time')
    plt.ylabel('Probability density')
    plt.xlim(5.4, 6.4)
    plt.savefig('06-assignment/normal/posterior_log_reaction_times.png')

    plt.figure()
    sns.distplot(child_posterior_correction, bins=20, norm_hist=True, label='Posterior difference')
    plt.axvline(diff_mean, color='b', lw=2, label='Difference mean')
    plt.axvline(diff_hdi[0], color='b', lw=2, linestyle='--', label='Difference 95% HDI')
    plt.axvline(diff_hdi[1], color='b', lw=2, linestyle='--')

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Difference in posterior log reaction times')
    plt.ylabel('Probability density')
    plt.xlim(-0.1, 0.9)
    plt.savefig('06-assignment/normal/posterior_log_reaction_difference.png')

    plt.figure()
    plot_dist_mean_hdi(child_correction_effect_size, color='b', label='Effect size')

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Effect size')
    plt.ylabel('Probability density')
    plt.xlim(-0.5, 9)
    plt.savefig('06-assignment/normal/effect_size.png')

def point_2(fit, fit_summary):
    posterior_tau_task_6 = fit.extract('tau')['tau']
    task_6_mean = np.mean(posterior_tau_task_6)
    task_6_hdi = hdi(posterior_tau_task_6)

    posterior_tau_task_5 = pickle.load(open('05-assignment/tau_posterior_for_assignment_6.pkl', 'rb'))['tau']
    task_5_mean = np.mean(posterior_tau_task_5)
    task_5_hdi = hdi(posterior_tau_task_5)

    tau_diff = posterior_tau_task_6 - posterior_tau_task_5
    diff_mean = np.mean(tau_diff)
    diff_hdi = hdi(tau_diff)
    diff_effect = tau_diff / np.std(tau_diff)

    print(f'τ difference mean: {diff_mean:.2f}, HDI: {diff_hdi}')
    print(f'Effect on τ mean: {np.mean(diff_effect):.2f}, HDI: {hdi(diff_effect)}')

    plt.figure()
    plot_dist_mean_hdi(posterior_tau_task_6, color='b', label='Task 6')
    plot_dist_mean_hdi(posterior_tau_task_5, color='r', label='Task 5')

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Posterior τ')
    plt.ylabel('Probability density')
    plt.savefig('06-assignment/normal/posterior_tau_comparison.png')

def point_3(fit, fit_summary):
    #adult_mean = fit_summary
    adult_mean = fit_summary.loc['mu[1]', 'mean']
    child_mean = fit_summary.loc['mu[1]', 'mean'] + fit_summary.loc['mu[2]', 'mean']
    single_mean_task_5 = 5.76
    standard_deviation_task_5 = 0.28
    standard_deviation = fit_summary.loc['tau', 'mean']
    log_reaction_time = np.linspace(4.5, 7, 1001)

    adult_pdf = stats.norm.pdf(log_reaction_time, adult_mean, standard_deviation)
    child_pdf = stats.norm.pdf(log_reaction_time, child_mean, standard_deviation)
    task_5_pdf = stats.norm.pdf(log_reaction_time, single_mean_task_5, standard_deviation_task_5)

    plt.figure()
    sns.lineplot(log_reaction_time, adult_pdf, label='Adult')
    sns.lineplot(log_reaction_time, child_pdf, label='Child')
    sns.lineplot(log_reaction_time, task_5_pdf, label='Task 5')

    plt.ylabel('Probability density')
    plt.xlabel('Log reaction time')
    sns.despine(left=True)
    plt.savefig('06-assignment/normal/prior_pdfs.png')

def point_4(fit, fit_summary):
    num_posterior_samples = (fit.sim['iter'] - fit.sim['warmup']) * fit.sim['chains']

    # 10000 Random people samples (from assignment 5)
    uninformed_random_people_samples = pickle.load(open('05-assignment/random_person_samples.pkl', 'rb'))

    # 10000 Random people samples (uninformed with bernoulli)
    num_new_people = 10000
    random_person_samples_bernoulli = np.zeros((num_new_people, ))
    for i in range(num_new_people):
        new_person_mu_1 = fit.extract('mu[1]')['mu[1]'][np.random.randint(num_posterior_samples)]
        new_person_mu_2 = fit.extract('mu[2]')['mu[2]'][np.random.randint(num_posterior_samples)]
        new_person_tau = fit.extract('tau')['tau'][np.random.randint(num_posterior_samples)]
        new_person_sigma = fit.extract('sigma')['sigma'][np.random.randint(num_posterior_samples)]
        is_child = np.random.binomial(1, 9/34, (1, ))

        new_person_theta = np.random.normal(new_person_mu_1 + new_person_mu_2 * is_child, new_person_tau)
        sample_log_reaction_time = np.random.normal(new_person_theta, new_person_sigma, (1, ))
        random_person_samples_bernoulli[i] = sample_log_reaction_time

    # 10000 random people samples (uninformed with average)
    num_new_people = 10000
    random_person_samples_average = np.zeros((num_new_people, ))
    for i in range(num_new_people):
        new_person_mu_1 = fit.extract('mu[1]')['mu[1]'][np.random.randint(num_posterior_samples)]
        new_person_mu_2 = fit.extract('mu[2]')['mu[2]'][np.random.randint(num_posterior_samples)]
        new_person_tau = fit.extract('tau')['tau'][np.random.randint(num_posterior_samples)]
        new_person_sigma = fit.extract('sigma')['sigma'][np.random.randint(num_posterior_samples)]

        new_person_theta = np.random.normal(new_person_mu_1 + new_person_mu_2 * 9 / 34, new_person_tau)
        sample_log_reaction_time = np.random.normal(new_person_theta, new_person_sigma, (1, ))
        random_person_samples_average[i] = sample_log_reaction_time

    # 10000 random adults
    num_new_people = 10000
    random_adult_samples = np.zeros((num_new_people, ))
    for i in range(num_new_people):
        new_person_mu_1 = fit.extract('mu[1]')['mu[1]'][np.random.randint(num_posterior_samples)]
        new_person_tau = fit.extract('tau')['tau'][np.random.randint(num_posterior_samples)]
        new_person_sigma = fit.extract('sigma')['sigma'][np.random.randint(num_posterior_samples)]

        new_person_theta = np.random.normal(new_person_mu_1, new_person_tau)
        sample_log_reaction_time = np.random.normal(new_person_theta, new_person_sigma, (1, ))
        random_adult_samples[i] = sample_log_reaction_time

    # 10000 random children
    num_new_people = 10000
    random_child_samples = np.zeros((num_new_people, ))
    for i in range(num_new_people):
        new_person_mu_1 = fit.extract('mu[1]')['mu[1]'][np.random.randint(num_posterior_samples)]
        new_person_mu_2 = fit.extract('mu[2]')['mu[2]'][np.random.randint(num_posterior_samples)]
        new_person_tau = fit.extract('tau')['tau'][np.random.randint(num_posterior_samples)]
        new_person_sigma = fit.extract('sigma')['sigma'][np.random.randint(num_posterior_samples)]

        new_person_theta = np.random.normal(new_person_mu_1 + new_person_mu_2, new_person_tau)
        sample_log_reaction_time = np.random.normal(new_person_theta, new_person_sigma, (1, ))
        #sample_reaction_times = np.exp(sample_log_reaction_times)
        random_child_samples[i] = sample_log_reaction_time

    plt.figure()
    sns.distplot(uninformed_random_people_samples, bins=20, norm_hist=True, label='Uninformed from Ass. 5')
    sns.distplot(random_person_samples_bernoulli, bins=20, norm_hist=True, label='Uninformed from bernoulli')
    sns.distplot(random_person_samples_average, bins=20, norm_hist=True, label='Uninformed from average')

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Log reaction time')
    plt.ylabel('Probability density')
    plt.savefig('06-assignment/normal/compare_uninformed_samples.png')

    plt.figure()
    sns.distplot(uninformed_random_people_samples, bins=20, norm_hist=True, label='Uninformed random person')
    sns.distplot(random_adult_samples, bins=20, norm_hist=True, label='Adult random person')
    sns.distplot(random_child_samples, bins=20, norm_hist=True, label='Child random person')

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Log reaction time')
    plt.ylabel('Probability density')
    plt.savefig('06-assignment/normal/posterior_predictive_check.png')

    plt.figure()
    sns.distplot(np.exp(uninformed_random_people_samples), bins=20, norm_hist=True, label='Uninformed random person')
    sns.distplot(np.exp(random_adult_samples), bins=20, norm_hist=True, label='Adult random person')
    sns.distplot(np.exp(random_child_samples), bins=20, norm_hist=True, label='Child random person')

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Reaction time (ms)')
    plt.ylabel('Probability density')
    plt.savefig('06-assignment/normal/posterior_predictive_check_ms.png')

def main():
    with open('06-assignment/data.json') as json_file:
        data = json.load(json_file)
    data_df = pd.read_csv('06-assignment/data.csv')

    plot_data(data_df)

    model = load_stan_model('06-assignment/hierarchical_model.stan')

    fit, fit_summary, fit_model = fit_stan_model(model, data, iter=10000, warmup=1000)

    fit_summary.to_csv('06-assignment/normal/fit_summary.csv')

    point_1(fit, fit_summary)
    point_2(fit, fit_summary)
    point_3(fit, fit_summary)
    point_4(fit, fit_summary)
if __name__ == '__main__':
    main()
