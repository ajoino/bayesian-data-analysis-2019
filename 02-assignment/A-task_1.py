import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

np.random.seed(1334)

def coin_flip(bias=0.5, num_flips=100):
    flips = (np.random.rand(num_flips) >= 1-bias).astype(int)

    return flips

def plot_bias_line(bias = 0.5, num_flips=100):
    ax = plt.plot([1, num_flips + 1], [bias, bias], 'k:', zorder=1)
    return ax

"""
def plot_bernoulli_proportions(bias=0.5, num_flips=100, **kwargs):
    *cum_data, flips = coin_flip(bias, num_flips)
    ax = plt.plot(*cum_data, num_flips), 
            linestyle='-', linewidth=1.2, marker='o', markersize=5, markerfacecolor='none'
            , **kwargs)
    return flips
"""

def main():
    N = 100
    flips = {50: coin_flip(bias=0.5, num_flips=N),
             25: coin_flip(bias=0.25, num_flips=N)}
    abscissa = np.arange(1, N+1)

    plt.figure()
    plt.semilogx(abscissa, np.cumsum(flips[50])/abscissa, 
            linewidth=1.2, marker='o', markersize=5, markerfacecolor='none', label=r'\[\theta = 0.5\]')
    plt.semilogx(abscissa, np.cumsum(flips[25])/abscissa,
            linewidth=1.2, marker='o', markersize=5, markerfacecolor='none', label=r'\[\theta = 0.25\]')
    plot_bias_line(0.5, N)
    plot_bias_line(0.25, N)
    plt.legend()

    plt.draw()
    input()
    plt.close()
    fig, ax = plt.subplots()
    Ns = [100, 1000, 10000]
    width = 0.3
    x = np.arange(len(Ns))
    new_flips = np.array([np.sum(coin_flip(bias=0.25, num_flips=N))/N for N in Ns])
    print(1-new_flips)
    heads = plt.bar(x - width/2, new_flips, width, label='Heads')
    tails = plt.bar(x + width/2, 1-new_flips, width, label='Tails')
    plt.plot([-0.5, 2.5], [0.25, 0.25], 'k--')
    plt.plot([-0.5, 2.5], [0.75, 0.75], 'k--')
    plt.ylabel('Probability')
    plt.xlabel('Number of flips')
    ax.set_xticks(x)
    ax.set_xticklabels(Ns)
    plt.legend()

    sns.despine()
    plt.show()
    input()

if __name__ == '__main__':
    plt.ion()
    main()

