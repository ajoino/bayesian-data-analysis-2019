import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

np.random.seed(1337)

def normal_pdf(x, mu=0.0, sigma=1.0):
    return 1/sigma/np.sqrt(2 * np.pi) * np.exp(-(x - mu)**2 / 2 / sigma**2)

def lognormal_pdf(x, mu=0.0, sigma=1.0):
    return normal_pdf(np.log(x), mu, sigma) / x

def lognormal_negative(x, mu=0.0, sigma=1.0):
    return -lognormal_pdf(x, mu, sigma)

def sample_normal(N, mu=0.0, sigma=1.0):
    return sigma * np.random.randn(N) + mu

def normal_stuff(N):
    #%% normal distribution
    mu, sigma = 3.4, np.sqrt(3)
    sampled_normal = sample_normal(N, mu, sigma)
    bin_width = 0.1

    plt.figure()
    plt.hist(sampled_normal, bins=np.arange(-3, 10, bin_width), density=True)
    x = np.arange(-11.6, 18.4, bin_width)
    dx = x[1] - x[0]
    plt.plot(x, normal_pdf(x, mu, sigma))
    plt.axis([-2, 9, -0.01, 0.26])
    plt.draw()
    input()

    sample_mean = np.mean(sampled_normal)
    expected_mean = np.sum(normal_pdf(x, mu, sigma) * x) * dx
    print('NORMAL DISTRIBUTION')
    print('EXPECTED VALUE')
    print(f'Sample mean: {sample_mean:.2f}')
    print(f'Expected mean: {expected_mean:.2f}\n')
    plt.plot([expected_mean, expected_mean], [0, 0.3], 'r--')
    input()
    print('VARIANCE')
    sample_variance = np.var(sampled_normal)
    expected_variance_1 = np.sum(normal_pdf(x, mu, sigma) * (x - mu)**2) * dx
    expected_variance_2 = np.sum(normal_pdf(x, mu, sigma) * (x - expected_mean)**2) * dx
    print(f'Sample variance: {sample_variance:.2f}')
    print(f'Expected variance a): {expected_variance_1:.2f}')
    print(f'Expected variance b): {expected_variance_2:.2f}')
    plt.plot([expected_mean + expected_variance_1, expected_mean - expected_variance_1], [0, 0], 'k',
            linewidth=10)

    input()
    plt.close()
    #%% lognormal distribution
    print('LOGNORMAL DISTRIBUTION')
    sampled_lognormal = np.random.lognormal(size=N)
    x = np.linspace(0.001, 20, 1001)

    plt.figure()
    plt.hist(sampled_lognormal, bins=np.arange(-1, 8, bin_width), density=True)
    plt.plot(x, lognormal_pdf(x))

    estimated_mode = x[np.argmax(lognormal_pdf(x))]
    calculated_mode = optimize.fmin(lognormal_negative, 1, disp=False)[0]
    print(f'Estimated mode: {estimated_mode:.2f}')
    print(f'Calculated mode: {calculated_mode:.2f}')
    print(f'Theoretical mode: {np.exp(0 - 1**2):.2f}')
def main():
    plt.ion()
    N = 10000
    normal_stuff(N)

    plt.show()
    plt.ioff()

if __name__ == '__main__':
    main()
