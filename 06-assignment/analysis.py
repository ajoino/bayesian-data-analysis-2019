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

def plot_data(data_df):
    sns.distplot(data_df[data_df.is_datum_child == 1]['reaction_times'], bins=20, norm_hist=True, label='Children')
    sns.distplot(data_df[data_df.is_datum_child == 0]['reaction_times'], bins=20, norm_hist=True, label='Adults')

    adult_reaction_mean, child_reaction_mean = data_df.groupby('is_datum_child').mean()['reaction_times'].to_numpy()
    adult_reaction_median, child_reaction_median = data_df.groupby('is_datum_child').median()['reaction_times'].to_numpy()
    print(adult_reaction_mean, adult_reaction_median)
    print(child_reaction_mean, child_reaction_median)

    plt.legend()
    sns.despine(left=True)
    plt.xlabel('Reaction time (ms)')
    plt.ylabel('Probability density')
