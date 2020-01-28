import pickle
from pathlib import Path
import pystan
import numpy as np
import itertools
from scipy import signal
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

def plot_all_users(data, column):
    plt.figure()
    num_users = 34
    y_extra = 0.05
    data_range = np.array(
            [np.ceil((1 - y_extra) * np.min(data[column])), 
             np.floor((1 + y_extra) * np.max(data[column]))])
    for col in range(3):
        for row in range(12):
            if row * col > num_users:
                break
            plt.subplot(12, 3, row * 3 + col + 1)
            sns.scatterplot(x='attempt', y=column, data=data[data['users'] == row * 3 + col + 1])
            plt.xlim([-1, 21])
            plt.ylim(data_range)

def plot_ass_users(data, users, column):
    fig = plt.figure()
    num_users = len(users)
    y_extra = 0.05
    data_range = np.array(
            [np.ceil((1 - y_extra) * np.min(data[column])), 
             np.floor((1 + y_extra) * np.max(data[column]))])
    for j, user in enumerate(users):
        plt.subplot(num_users, 1, j + 1)
        sns.scatterplot(x='attempt', y=column, data=data[data['users'] == user + 1], label='Input', zorder=2)
        plt.xlim([0, 21])
        plt.ylim(data_range)

    return fig, num_users

def plot_user_avg(fig, fit, users, attempts, plot_log=True):
    allaxes = fig.get_axes()
    theta_mean = np.mean(fit['theta'], axis=0)[:, users]
    sigma_mean = np.mean(fit['sigma'])
    attempts = np.array([[1] * len(attempts), attempts])
    means = theta_mean.transpose() @ attempts
    for i, (ax, mean) in enumerate(zip(allaxes, means)):
        data = pd.DataFrame()
        data['attempts'] = attempts[1, :]
        if not plot_log:
            data['mean'] = np.exp(mean[:, np.newaxis] + sigma_mean**2 / 2)
        else:
            data['mean'] = mean[:, np.newaxis]
        sns.scatterplot(x='attempts', y='mean', data=data, ax=ax, zorder=2, label='Predicted average')
        print(f'Expected log reaction time for user {users[i]}: {mean}')
        print(f'Expected reaction time for user {users[i]}: {np.exp(mean + sigma_mean**2 / 2)}')
        if i == 1:
            ax.set_xlabel('')
            ax.set_ylabel('Reaction time (ms)')
        elif i == 2:
            ax.set_xlabel('Attempt')
            ax.set_ylabel('')
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')

    return fig

def plot_credible_lines(fig, fit, users, attempts, num_posterior_lines, plot_log=True):
    allaxes = fig.get_axes()
    num_samples, *sample_size = fit['theta'].shape
    posterior_theta = fit['theta']
    line_x = np.array([[1, 1], [-1, 23]])
    return_thetas = []
    for i, (ax, user) in enumerate(zip(allaxes, users)):
        random_posterior_index = np.random.randint(num_samples, size=(sample_size[0], num_posterior_lines))
        random_posterior_theta_0 = posterior_theta[random_posterior_index[0, :], 0, user]
        random_posterior_theta_1 = posterior_theta[random_posterior_index[1, :], 1, user]
        random_posterior_theta = np.array([random_posterior_theta_0, random_posterior_theta_1])
        random_posterior_lines = random_posterior_theta.transpose() @ line_x
        if not plot_log:
            random_posterior_lines = np.exp(random_posterior_lines)
        #sns.lineplot(x=line_x[1, :], y=random_posterior_lines, ax=ax)
        ax.plot(line_x[1, :], random_posterior_lines.transpose(), color='orange', alpha=0.01, zorder=0)
        return_thetas.append(random_posterior_theta)

    return fig, np.array(return_thetas)

def plot_hdi_lines(fig, random_posterior_theta, users, plot_log=False):
    allaxes = fig.get_axes()
    line_x = np.array([[1]*58, np.arange(-5, 24, 0.5)])
    for i, (ax, posterior_theta, user) in enumerate(zip(allaxes, random_posterior_theta, users)):
        posterior_lines = posterior_theta.T @ line_x
        if not plot_log:
            posterior_lines = np.exp(posterior_lines)
        posterior_hdi = np.array(list(map(hdi, posterior_lines.T))) 
        #ax.plot(np.array([line_x[1, :], line_x[1, :]]), posterior_hdi.T)
        ax.plot(line_x[1, :], posterior_hdi[:, 0], color='tab:blue', zorder=1)
        ax.plot(line_x[1, :], posterior_hdi[:, 1], color='tab:blue', label='Posterior predicitive 95% HDI', zorder=1)

    return fig

def chain_correlation(fit):
    posterior_mu_0 = fit['mu'][:, 0, 0]
    posterior_sigma = fit['sigma']

    num_samples = len(posterior_sigma)

    # First way to create chains
    mu_chains = np.array_split(posterior_mu_0, 4)
    sigma_chains = np.array_split(posterior_sigma, 4)

    plt.figure()
    plt.subplot(2, 1, 1)
    for chain in mu_chains:
        plt.plot(np.arange(5000, 5201), chain[:201])
    plt.title(f'ν')
    plt.subplot(2, 1, 2)
    for chain in sigma_chains:
        plt.plot(np.arange(5000, 5201), chain[:201])
    plt.title(f'σ')

def before_points(fit, data):
    print(f'θ: {np.mean(fit["theta"], axis=0)}')
    print(f'σ: {np.mean(fit["sigma"], axis=0)}')
    print(f'μ: {np.mean(fit["mu"], axis=0)}')
    print(f'τ: {np.mean(fit["tau"], axis=0)}')

    users = [0, 2, 3]
    estimated_attempts = [1, 5]
    num_posterior_lines = 1000
    fig, num_users = plot_ass_users(data, users, 'reaction_times')
    fig = plot_user_avg(fig, fit, users, estimated_attempts, plot_log=False)
    fig, rpt = plot_credible_lines(fig, fit, users, estimated_attempts, num_posterior_lines, plot_log=False)
    fig = plot_hdi_lines(fig, rpt, users, plot_log=False)
    for i, ax in enumerate(fig.get_axes()):
        if i != 0:
            ax.legend().set_visible(False)
        ax.legend(ncol=3)


if __name__ == '__main__':
    data_df = pd.read_csv('07-assignment/data.csv')
    data_df['log_reaction_times'] = np.log(data_df['reaction_times'])
    print(data_df.columns)
    fit = pickle.load(open('07-assignment/normal_model/fit.pkl', 'rb'))
    #plot_all_users(data_df, 'log_reaction_times')

    before_points(fit, data_df)
    plt.savefig('07-assignment/normal_model/task_3.png', dpi=600)
    chain_correlation(fit)
    plt.savefig('07-assignment/normal_model/corr.png', dpi=600)
