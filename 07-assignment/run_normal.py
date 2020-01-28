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

if __name__ == '__main__':
    with open('07-assignment/data.json') as json_file:
        data = json.load(json_file)
    data_df = pd.read_csv('07-assignment/data.csv')

    model_directory = Path('07-assignment/normal_model/')

    model = load_stan_model(model_directory / 'normal_model.stan')

    fit, fit_summary, fit_model = fit_stan_model(model, data, iter=35000, warmup=5000)

    fit_summary.to_csv(model_directory / 'fit_summary.csv', float_format='%.5f')

    pickle.dump(fit.extract(), open(model_directory / 'fit.pkl', 'wb'))

