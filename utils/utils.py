import pickle
from pathlib import Path
import pystan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .hdi import hdi

def load_stan_model(model_path):
    model_path = Path(model_path)

    if model_path.with_suffix('.pkl').exists():
        # See if model is saved, and use the saved one if that is the case
        with open(model_path.with_suffix('.pkl'), 'rb') as model_pickle:
            model = pickle.load(model_pickle)
        return model
    else:
        # If model is not saved, load it from the code
        with open(model_path) as model_file:
            model = pystan.StanModel(model_file)
        # Then save the compiled model
        with open(model_path.with_suffix('.pkl'), 'wb') as model_pickle:
            pickle.dump(model, model_pickle)
        return model

def fit_stan_model(stan_model, data, iter=1000, seed=1):
    fit = stan_model.sampling(data=data, iter=iter, chains=4, warmup=1000, thin=1, seed=seed, verbose=True)
    summary_dict = fit.summary()
    summary_df = pd.DataFrame(
            summary_dict['summary'],
            columns=summary_dict['summary_colnames'],
            index=summary_dict['summary_rownames']
            )

    return fit, summary_df, stan_model

# Plot_trace function from Matthew West's article: 
# https://towardsdatascience.com/an-introduction-to-bayesian-inference-in-pystan-c27078e58d53
def plot_trace(param, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
    #HDI, _ = hdi(

    # Plotting
    plt.subplot(2,1,1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title('Trace and Posterior Distribution for {}'.format(param_name))

    plt.subplot(2,1,2)
    plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()

# plot_posterior function adapted from Matthew West's article:
# https://towardsdatascience.com/an-introduction-to-bayesian-inference-in-pystan-c27078e58d53
def plot_posterior(param, param_name='parameter', ax=None,
                plot_mean=False, plot_median=False, ci=False, **kwargs):
    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    if not ax:
        pass

    #plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
    sns.distplot(param, bins=30, norm_hist=True, kde_kws={'shade': False}, ax=ax, **kwargs)
    ax.set_xlabel(param_name)
    ax.set_ylabel('density')

    if plot_mean:
        ax.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
    if plot_median:
        ax.axvline(median, color='c', lw=2, linestyle='--',label='median')

    if ci == 'vertical':
        ax.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
        ax.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
    elif ci == 'horizontal':
        ax.plot([cred_min, cred_max], [0, 0], linewidth=5, color='k', label='95% CI')
    elif ci == 'both':
        ax.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
        ax.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
        ax.plot([cred_min, cred_max], [0, 0], linewidth=5, color='k', label='95% CI')

    #plt.gcf().tight_layout()
    if plot_mean or plot_median or ci:
        ax.legend()

