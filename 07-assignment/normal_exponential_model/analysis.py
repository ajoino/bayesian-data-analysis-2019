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
        sns.scatterplot(x='attempt', y=column, data=data[data['users'] == user + 1])
        plt.xlim([0, 21])
        plt.ylim(data_range)

    return fig, num_users

def plot_user_avg(fig, fit, users, attempts):
    allaxes = fig.get_axes()
    theta_mean = np.mean(fit['theta'], axis=0)[:, users]
    sigma_mean = np.mean(fit['sigma'])
    attempts = np.array([[1] * len(attempts), attempts])
    means = theta_mean.transpose() @ attempts
    for i, (ax, mean) in enumerate(zip(allaxes, means)):
        data = pd.DataFrame()
        data['attempts'] = attempts[1, :]
        data['mean'] = mean[:, np.newaxis]
        sns.scatterplot(x='attempts', y='mean', data=data, ax=ax, zorder=2)
        print(f'Expected log reaction time for user {users[i]}: {mean}')
        print(f'Expected reaction time for user {users[i]}: {np.exp(mean + sigma_mean**2 / 2)}')

    return fig

def plot_credible_lines(fig, fit, users, attempts, num_posterior_lines):
    allaxes = fig.get_axes()
    num_samples, *sample_size = fit['theta'].shape
    posterior_theta = fit['theta']
    line_x = np.array([[1, 1], [-1, 23]])
    for i, (ax, user) in enumerate(zip(allaxes, users)):
        random_posterior_index = np.random.randint(num_samples, size=(sample_size[0], num_posterior_lines))
        random_posterior_theta_0 = posterior_theta[random_posterior_index[0, :], 0, user]
        random_posterior_theta_1 = posterior_theta[random_posterior_index[1, :], 1, user]
        random_posterior_theta = np.array([random_posterior_theta_0, random_posterior_theta_1])
        random_posterior_lines = random_posterior_theta.transpose() @ line_x
        #sns.lineplot(x=line_x[1, :], y=random_posterior_lines, ax=ax)
        ax.plot(line_x[1, :], random_posterior_lines.transpose(), color='orange', alpha=0.05, zorder=0)

    return fig

def plot_hdi_lines(fig, fit, users):
    allaxes = fig.get_axes()
    posterior_theta = fit['theta']
    posterior_sigma = fit['sigma']
    line_x = np.array([[1, 1], [-1, 23]])
    for i, (ax, user) in enumerate(zip(allaxes, users)):
        posterior_theta_mean = np.mean(posterior_theta, axis=0)[:, user]
        posterior_sigma_mean = np.mean(posterior_sigma)
        #posterior_theta_hdi_lines = posterior_theta_hdi.transpose() @ line_x
        #print(posterior_theta_hdi_lines)
        posterior_theta_line = posterior_theta_mean @ line_x
        posterior_sigma_line_upper = posterior_theta_line + 1.96 * posterior_sigma_mean
        posterior_sigma_line_lower = posterior_theta_line - 1.96 * posterior_sigma_mean
        ax.plot(line_x[1, :], posterior_theta_line, color='red', zorder=1)
        #ax.plot(line_x[1, :], posterior_sigma_line_upper, color='red', linestyle='--', zorder=1)
        #ax.plot(line_x[1, :], posterior_sigma_line_lower, color='red', linestyle='--', zorder=1)

    return fig

def before_points(fit, data):
    print(f'θ: {np.mean(fit["theta"], axis=0)}')
    print(f'σ: {np.mean(fit["sigma"], axis=0)}')
    print(f'μ: {np.mean(fit["mu"], axis=0)}')
    print(f'τ: {np.mean(fit["tau"], axis=0)}')

    users = [0, 3, 4]
    estimated_attempts = [1, 5]
    num_posterior_lines = 1000
    fig, num_users = plot_ass_users(data, users, 'log_reaction_times')
    fig = plot_user_avg(fig, fit, users, estimated_attempts)
    fig = plot_credible_lines(fig, fit, users, estimated_attempts, num_posterior_lines)
    fig = plot_hdi_lines(fig, fit, users)

if __name__ == '__main__':
    data_df = pd.read_csv('07-assignment/data.csv')
    data_df['log_reaction_times'] = np.log(data_df['reaction_times'])
    print(data_df.columns)
    fit = pickle.load(open('07-assignment/normal_exponential_model/fit.pkl', 'rb'))
    plot_all_users(data_df, 'log_reaction_times')

    before_points(fit, data_df)
    plt.show()
