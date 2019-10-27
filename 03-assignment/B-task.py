import numpy as np
import scipy.stats as stats
import scipy.special as special
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')
from pprint import pprint
import pandas as pd
pd.set_option('precision', 3)

def bernoulli(y, theta):
    return theta**y * (1 - theta)**(1 - y)

def likelihood(y, theta, axis=None):
    return np.prod(bernoulli(y, theta), axis=axis)

def log_bernoulli(y, theta):
    return y * np.log10(theta + 1e-10) + (1 - y) * np.log10(1 - theta + 1e-10)

def log_likelihood(y, theta, axis=None):
    return np.sum(log_bernoulli(y, theta), axis=axis)

def coin_flip(bias=0.5, num_flips=100):
    flips = (np.random.rand(num_flips) >= 1-bias).astype(int)

    return flips

def point_2():
    probabilities = []
    for y in range(2):
        for theta in [0.25, 0.5]:
            probabilities.append([y, theta, bernoulli(y, theta)])
    probabilities = pd.DataFrame(probabilities, columns=['outcome', 'theta', 'probability'])
    print(probabilities)

    sns.barplot(x='theta', y='probability', hue='outcome',
            data=probabilities)
    sns.despine()
    return probabilities

def point_3():
    theta = np.linspace(0, 1, 101)

    plt.plot(theta, bernoulli(1, theta), label='γ = 1')
    plt.plot(theta, bernoulli(0, theta), label='γ = 0')
    plt.legend()
    plt.xlabel('θ')
    plt.ylabel('Likelihood')

def point_4():
    probabilities = []
    log_probabilities = []
    bias = 0.5
    for num_flips in np.logspace(1,5,5).astype(int):
        y = coin_flip(bias, num_flips)
        theta = np.ones(y.shape)*bias
        probabilities.append([num_flips, likelihood(y, theta), log_likelihood(y, theta), 10**log_likelihood(y, theta)])
    probabilities = pd.DataFrame(probabilities, columns=['flips', 'likelihood', 'log_likelihood', 'exp_log_likehood'])
    print('Why the log-likelhood matters')
    print(probabilities)

    dataset = [[1], [1, 1], [1, 1, 0, 1]]
    theta = np.linspace(0, 1, 101)
    likelihoods = pd.DataFrame(columns=['theta', 'input', 'likelihood'])
    #pd.DataFrame(columns=['theta', 'input', 'likelihood'])
    for data in dataset:
        xv, yv = np.meshgrid(data, theta)
        specific_likelihood = likelihood(xv, yv, axis=1)

        input_string = [f'γ = {data}']*len(theta)
        likelihood_df = pd.DataFrame(list(zip(theta, input_string, specific_likelihood)),
                columns=['theta', 'input', 'likelihood'])
        likelihoods = likelihoods.append(likelihood_df, ignore_index=True)

    sns.lineplot(x='theta', y='likelihood', hue='input', data=likelihoods)
    sns.despine()
    plt.title('Likelihoods using normal likelihood function')
    plt.xlabel('θ')
    plt.ylabel('Likelihood')

    return log_probabilities

def point_5(a=1, b=1):
    theta = np.linspace(0, 1, 101)
    beta_prior = stats.beta.pdf(theta, a, b)
    print(np.max(beta_prior))

    dataset = [[1], [1, 1], [1, 1, 0, 1]]
    theta = np.linspace(0, 1, 101)
    probabilities = pd.DataFrame(columns=['theta', 'input', 'probability'])
    #pd.DataFrame(columns=['theta', 'input', 'likelihood'])
    for data in dataset:
        xv, yv = np.meshgrid(data, theta)
        specific_probability = likelihood(xv, yv, axis=1) * beta_prior / np.trapz(likelihood(xv, yv, axis=1) * beta_prior) / (theta[1] - theta[0])

        input_string = [f'γ = {data}']*len(theta)
        probability = pd.DataFrame(list(zip(theta, input_string, specific_probability)),
                columns=['theta', 'input', 'probability'])
        probabilities = probabilities.append(probability, ignore_index=True)

    sns.lineplot(x='theta', y='probability', hue='input', data=probabilities)
    sns.despine()
    plt.title(f'Probability without logs. a = {a}, b = {b}.')
    plt.xlabel('θ')
    plt.ylabel('Probability')

def point_5_log(a=1, b=1):
    theta = np.linspace(0, 1, 101)
    beta_prior = stats.beta.pdf(theta, a, b)

    dataset = [[1], [1, 1], [1, 1, 0, 1]]
    theta = np.linspace(0, 1, 101)
    probabilities = pd.DataFrame(columns=['theta', 'input', 'probability'])
    #pd.DataFrame(columns=['theta', 'input', 'likelihood'])
    for data in dataset:
        xv, yv = np.meshgrid(data, theta)
        specific_probability = 10**(log_likelihood(xv, yv, axis=1) + np.log10(beta_prior)) / np.trapz(10**(log_likelihood(xv, yv, axis=1) + np.log10(beta_prior))) / (theta[1] - theta[0])

        input_string = [f'γ = {data}']*len(theta)
        probability = pd.DataFrame(list(zip(theta, input_string, specific_probability)),
                columns=['theta', 'input', 'probability'])
        probabilities = probabilities.append(probability, ignore_index=True)

    sns.lineplot(x='theta', y='probability', hue='input', data=probabilities)
    sns.despine()
    plt.title(f'Probability with logs. a = {a}, b = {b}.')
    plt.xlabel('θ')
    plt.ylabel('Probability')

def main():
    point_2()
    plt.show()
    plt.figure()
    point_3()
    plt.show()
    plt.figure()
    point_4()
    plt.figure()
    point_5(1,1)
    plt.figure()
    point_5_log(1,1)
    plt.figure()
    point_5(2,2)
    plt.figure()
    point_5(3, 1)
    plt.figure()
    point_5(4, 2)
    plt.figure()
    point_5(15, 5)
    plt.show()

if __name__ == '__main__':
    main()
