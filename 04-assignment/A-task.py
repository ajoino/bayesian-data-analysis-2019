import pickle
from pathlib import Path
import pystan
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_stan_model, fit_stan_model, \
                        plot_trace, plot_posterior
from utils.hdi import hdi_from_book as hdi

random_seed = 1338
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
    plot_posterior(fit_1['theta'], ax=ax7, hist=False)
    hdi_1 = hdi(fit_1['theta'])

    # Second column
    # Prior
    beta_prior = stats.beta.pdf(theta, 18.25, 6.75)
    ax2.plot(theta, beta_prior)
    # Likelihood
    yv, theta_v = np.meshgrid(y, theta)
    likelihood_ = likelihood(yv, theta_v, axis=1)
    ax5.plot(theta, likelihood_)
    # Posterior
    plot_posterior(fit_2['theta'], ax=ax8, hist=False)
    # First column
    # Prior
    beta_prior = stats.beta.pdf(theta, 1, 1)
    ax3.plot(theta, beta_prior)
    # Likelihood
    yv, theta_v = np.meshgrid(y, theta)
    likelihood_ = likelihood(yv, theta_v, axis=1)
    ax6.plot(theta, likelihood_)
    # Posterior
    plot_posterior(fit_3['theta'], ax=ax9, hist=False)


def point_1():
    model_1 = load_stan_model('04-assignment/model_1.stan')
    model_2 = load_stan_model('04-assignment/model_2.stan')
    model_3 = load_stan_model('04-assignment/model_3.stan')

    y = [1] * 17 + [0] * 3
    data = {'N': len(y), 'y': y}
    print(data)

    fit_1, summary_1, _ = fit_stan_model(model_1, data, iter=iterations, seed=random_seed)
    fit_2, summary_2, _ = fit_stan_model(model_2, data, iter=iterations, seed=random_seed)
    fit_3, summary_3, _ = fit_stan_model(model_3, data, iter=iterations, seed=random_seed)

    recreate_figure_64([fit_1, fit_2, fit_3], y)

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

    d_theta = fit_y['theta'] - fit_z['theta']
    hdi_d_theta = hdi(d_theta)

    print(f'Expected probability that y and z are drawn from the same coin: {1 - np.mean(d_theta > 0):.2f}')

    print(f'HDI of difference between y and z: {hdi_d_theta[0]:.2f} - {hdi_d_theta[1]:.2f}')

    plt.figure()
    sns.distplot(d_theta, hist=True)
    plt.plot(hdi_d_theta, [0, 0], 'k', linewidth=5)
    plt.show()


if __name__ == '__main__':
    point_1()
    point_2()
