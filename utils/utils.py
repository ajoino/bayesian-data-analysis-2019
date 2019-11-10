import pickle
from pathlib import Path
import pystan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .hdi import hdi
import hashlib
import os, contextlib

def read_model_hash(model_path):
    if not model_path.with_suffix('.md5').exists():
        return None

    with open(model_path.with_suffix('.md5'), 'rb') as hash_file:
        saved_model_hash = hash_file.read()
    return saved_model_hash

def save_model_hash(model, model_path):
    with open(model_path.with_suffix('.md5'), 'wb') as hash_file:
        # Hash the model
        model_hash = hashlib.md5(model.model_code.encode())
        # Save the model hash
        hash_file.write(model_hash.digest())

def hash_from_model(model):
    return hashlib.md5(model.model_code.encode()).digest()

def hash_from_file(model_path):
    with open(model_path) as model_file:
        model_code = model_file.read()
    return hashlib.md5(model_code.encode()).digest()

def create_model(model_path):
    # Load stan model as plaintext
    with open(model_path) as model_file:
        # Compile the model
        model = pystan.StanModel(model_file)

    save_model_hash(model, model_path)

    with open(model_path.with_suffix('.pkl'), 'wb') as model_pickle:
        # Then save the compiled model
        pickle.dump(model, model_pickle)

    return model

def compare_hashes(model_path):
    saved_model_hash = read_model_hash(model_path)
    model_hash = hash_from_file(model_path)
    print(saved_model_hash)
    print(model_hash)
    return saved_model_hash != model_hash

def load_stan_model(model_path, force_compile=False):
    model_path = Path(model_path)

    if not model_path.with_suffix('.pkl').exists():
        return create_model(model_path)
    else:
        model_update = compare_hashes(model_path)
        if model_update:
            return create_model(model_path)
        with open(model_path.with_suffix('.pkl'), 'rb') as model_pickle:
            model = pickle.load(model_pickle)
        return model

def fit_stan_model(stan_model, data, iter=1000, warmup=None, seed=1, verbose=False):
    if warmup == None:
        print('warmup = None')
        warmup = iter // 10

    fit = stan_model.sampling(data=data, iter=iter, chains=4, warmup=warmup, thin=1, seed=seed, verbose=False)

    summary_dict = fit.summary()
    summary_df = pd.DataFrame(
            summary_dict['summary'],
            columns=summary_dict['summary_colnames'],
            index=summary_dict['summary_rownames']
            )

    return fit, summary_df, stan_model

def compare_distributions(x, y):
    x.sort()
    y.sort()

    y_len = len(y)
    for i, y_sample in reversed(list(enumerate(y))):
        # If x is always larger than y, return 100%
        if np.min(x) > np.max(y):
            return 1.0, (np.min(x) + np.max(y)) / 2

        y_proportion = i / y_len
        x_proportion = np.mean(x >= y_sample)
        if x_proportion >= y_proportion:
            return (x_proportion + y_proportion) / 2, (x[y_len - i] + y_sample) / 2


def roc_curve(x, y):
    # Assuming x is negative and y is positive
    len_x = len(x)
    sorted_x = np.sort(x)

    true_positive_rate, false_positive_rate = [None]*len(x), [None]*len(x)
    for i, x_sample in enumerate(sorted_x):
        true_positive_rate[i] = np.mean(y > x_sample)
        false_positive_rate[i] = (len_x - i)/len_x

    return true_positive_rate, false_positive_rate

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

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper
