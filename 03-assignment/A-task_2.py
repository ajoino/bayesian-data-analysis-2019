import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')
from pprint import pprint
import pandas as pd
pd.set_option('precision', 3)

def prob_having_disease(test_result='+', disease_prior=0.001, hit_rate=0.99, false_alarm_rate=0.05):
    test_positive_disease_present = hit_rate * disease_prior
    test_positive_disease_absent = false_alarm_rate * (1 - disease_prior)
    test_negative_disease_present = (1 - hit_rate) * disease_prior
    test_negative_disease_absent = (1 - false_alarm_rate) * (1 - disease_prior)

    test_positive_marginal = test_positive_disease_present + test_positive_disease_absent
    test_negative_marginal = test_negative_disease_present + test_negative_disease_absent

    if test_result == '+':
        prob_disease_present = test_positive_disease_present / test_positive_marginal
        prob_disease_absent = test_positive_disease_absent / test_positive_marginal
        return {'present': prob_disease_present, 'absent': prob_disease_absent}
    elif test_result == '-':
        prob_disease_present = test_negative_disease_present / test_negative_marginal
        prob_disease_absent = test_negative_disease_absent / test_negative_marginal
        return {'present': prob_disease_present, 'absent': prob_disease_absent}

def multiple_tests(test_results='+', disease_prior=0.001):
    probability_having_disease = []
    for test, test_result in enumerate(test_results):
        probability = prob_having_disease(test_result, disease_prior)['present']
        probability_having_disease.append(probability)
        disease_prior = probability

    return probability_having_disease


if __name__ == '__main__':
    # Quick test, should be around 0.019
    first_test_probability = prob_having_disease('+')['present']
    print(f'Probability of having disease after one test: \n\t{first_test_probability:.3f}')
    input()

    # Variables that go into the loop
    disease_prior = 0.001
    test_results_array = ['+++++', '-----', '+-+-+', '-+-+-', '++---', '--+++']

    # Dataframe for storage
    test_results_frame = pd.DataFrame(columns=['test_results', 'test', 'probability'])

    # Do the loop
    for test_results in test_results_array:
        num_tests = len(test_results) + 1

        # Do the test
        probs = [disease_prior] + multiple_tests(test_results)

        # Store the results
        test_results_ = [test_results] * num_tests
        test_index = np.arange(0, num_tests)
        specific_test_results = pd.DataFrame(
                list(zip(test_results_, test_index, np.log10(probs))), 
                columns=['test_results', 'test', 'probability']
                )
        test_results_frame = test_results_frame.append(specific_test_results, ignore_index=True)
    # Print the saved data
    print(test_results_frame)
    sns.lineplot(x='test', y='probability', hue='test_results', style='test_results', 
            data=test_results_frame)
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine()
    plt.show()

