import pickle
from pathlib import Path
import pystan
import numpy as np
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

random_seed = 1337
iterations = 10000

def bernoulli(y, theta):
    return theta**y * (1 - theta)**(1 - y)

def likelihood(y, theta, axis=None):
    return np.prod(bernoulli(y, theta), axis=axis)

def recreate_figure_64(fitted_models, y):
    fit_1, fit_2, fit_3 = fitted_models
    theta = np.linspace(0, 1, 1001)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex=True)
    # First column
    # Prior
    beta_prior = stats.beta.pdf(theta, 100, 100)
    ax1.plot(theta, beta_prior)
    # Likelihood
    yv, theta_v = np.meshgrid(y, theta)
    likelihood_ = likelihood(yv, theta_v, axis=1)
    ax4.plot(theta, likelihood_)
    # Posterior
    plot_posterior(fit_1['theta'], param_name='θ', ax=ax7, hist=False)

    # Second column
    # Prior
    beta_prior = stats.beta.pdf(theta, 18.25, 6.75)
    ax2.plot(theta, beta_prior)
    # Likelihood
    yv, theta_v = np.meshgrid(y, theta)
    likelihood_ = likelihood(yv, theta_v, axis=1)
    ax5.plot(theta, likelihood_)
    # Posterior
    plot_posterior(fit_2['theta'], param_name='θ', ax=ax8, hist=False)
    # Third column
    # Prior
    beta_prior = stats.beta.pdf(theta, 1, 1)
    ax3.plot(theta, beta_prior)
    # Likelihood
    yv, theta_v = np.meshgrid(y, theta)
    likelihood_ = likelihood(yv, theta_v, axis=1)
    ax6.plot(theta, likelihood_)
    # Posterior
    plot_posterior(fit_3['theta'], param_name='θ', ax=ax9, hist=False)


def point_1():
    model_1 = load_stan_model('04-assignment/model_1.stan')
    model_2 = load_stan_model('04-assignment/model_2.stan')
    model_3 = load_stan_model('04-assignment/model_3.stan')

    y = [1] * 17 + [0] * 3
    data = {'N': len(y), 'y': y}

    fit_1, summary_1, _ = fit_stan_model(model_1, data, iter=iterations, seed=random_seed)
    fit_2, summary_2, _ = fit_stan_model(model_2, data, iter=iterations, seed=random_seed)
    fit_3, summary_3, _ = fit_stan_model(model_3, data, iter=iterations, seed=random_seed)

    recreate_figure_64([fit_1, fit_2, fit_3], y)
    plt.tight_layout()
    sns.despine()
    plt.savefig('04-assignment/figure-64.png')

def point_2():
    model = load_stan_model('04-assignment/model_3.stan')

    y = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1 ,1 ,1]
    z = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0]

    data_y = {'N': len(y), 'y': y}
    data_z = {'N': len(z), 'y': z}

    fit_y, summary_y, _ = fit_stan_model(model, data_y, iter=iterations, seed=random_seed)
    fit_z, summary_z, _ = fit_stan_model(model, data_z, iter=iterations, seed=random_seed)
    
    hdi_y = hdi(fit_y['theta'])
    hdi_z = hdi(fit_z['theta'])
    print(f'Expected probability for getting heads with data y: {summary_y.loc["theta", "mean"]:.2f}')
    print(f'HDI of y: {hdi_y[0]:.2f} - {hdi_y[1]:.2f}')
    print(f'Probability of θ > 0.5: {np.mean(fit_y["theta"] > 0.5)}')

    d_theta = fit_y['theta'] - fit_z['theta']
    hdi_d_theta = hdi(d_theta)

    probability_higher, point_of_even = compare_distributions(fit_y['theta'], fit_z['theta'])
    plt.figure()
    sns.distplot(fit_y['theta'], bins=60, hist=True, label='θ_y')
    sns.distplot(fit_z['theta'], bins=60, hist=True, label='θ_z')
    plt.plot(hdi_y, [0, 0], 'b', alpha=0.5, linewidth=5, label='y 95% HDI')
    plt.plot(hdi_z, [0, 0], 'r', alpha=0.5, linewidth=5, label='z 95% HDI')
    plt.axvline(point_of_even, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('θ')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    sns.despine()
    plt.savefig('04-assignment/comparison.png')

    print(f'Expected probability that y and z are drawn from the same coin: {1 - np.mean(d_theta > 0):.2f}')
    print(f'True positive rate of choosing coin y: {1 - probability_higher:.2f}')
    print(f'HDI of difference between y and z: {hdi_d_theta[0]:.2f} - {hdi_d_theta[1]:.2f}')

    plt.figure()
    sns.distplot(d_theta, hist=True)
    plt.plot(hdi_d_theta, [0, 0], 'k', linewidth=5, label='95% HDI')
    plt.xlabel('dθ')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    sns.despine()
    plt.savefig('04-assignment/figure-point-2.png')

    plt.show()

if __name__ == '__main__':
    point_1()
    point_2()
