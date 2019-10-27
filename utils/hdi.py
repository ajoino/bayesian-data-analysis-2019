import numpy as np
import scipy.stats as stats
from scipy.integrate import cumtrapz
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
import seaborn as sns
from pprint import pprint
import pandas as pd

np.random.seed(1341)

def find_modes(y):
    modes = argrelextrema(y, np.greater)

    return modes[0]

def find_hdi_limits(integral_difference, hdi):
    high_idx = np.array(
            # Unravels finds the matrix indices given a flattened index
            np.unravel_index(
                # Argsort returns the flattened indices sorted from smallest
                np.argsort(
                    # Compare the integral difference with the given HDI-width
                    np.abs(hdi - integral_difference), 
                    axis=None), 
                integral_difference.shape)
            # Use only the smallest index
            )[:, 0]

    return high_idx

def hdi(x, y, hdi=0.95):
    # Integrate over the samples
    integral = cumtrapz(y, x, initial=0)
    # Trying mode stuff, it's hard :(
    num_modes = find_modes(y)
    # Find the pairwise differences between the integral values, this effectively give us all pairs
    # of density intervals
    integral_difference = integral[:, np.newaxis] - integral[np.newaxis, :]
    # Given the pairs of density intervals, find the one closest to 95%. This will of course only work for unimodal
    # distributions
    hdi_index = find_hdi_limits(integral_difference, hdi)
    # Extract the HDI limits
    HDI = np.array([x[hdi_index[0]], x[hdi_index[1]]])
    return HDI, hdi_index

def hdi_from_book(sample_vector, cred_mass=0.95):
    sample_vector.sort()
    sorted_samples = sample_vector
    credible_interval_index_inc = np.ceil(cred_mass * len(sorted_samples)).astype(int)
    ncis = len(sorted_samples) - credible_interval_index_inc
    credible_interval_width = np.zeros((ncis, ))
    for i in range(ncis):
        credible_interval_width[i] = sorted_samples[i + credible_interval_index_inc] - sorted_samples[i]
    hdi_min = sorted_samples[np.argmin(credible_interval_width)]
    hdi_max = sorted_samples[np.argmin(credible_interval_width) + credible_interval_index_inc]

    return np.array([hdi_min, hdi_max])


if __name__ == '__main__':
    def test():
        num_samples = 1000
        sampled_x = np.sort(np.random.uniform(low=0.0, high=1.0, size=(1000, )))
        sampled_y = stats.beta.pdf(sampled_x, 2, 5)

        sampled_y_1 = stats.beta.pdf(sampled_x, 2, 20)
        sampled_y_2 = stats.beta.pdf(sampled_x, 10, 2)
        sampled_y = (sampled_y_1 + sampled_y_2)/2
        print(np.trapz(sampled_y, sampled_x))

        plt.plot(sampled_x, sampled_y, '.-')
        HDI, HDI_index = hdi(sampled_x, sampled_y, hdi=0.95)
        print(sampled_y[HDI_index[1]] - sampled_y[HDI_index[0]])
        plt.plot(HDI, [0, 0], 'k')
        plt.show()

    test()
