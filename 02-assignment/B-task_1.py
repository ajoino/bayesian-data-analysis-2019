import numpy as np
import pandas as pd
pd.set_option('precision', 3)
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')

np.random.seed(1338)

def pivot_table(filename, return_full=True):
    df = pd.read_csv(filename)
    pivot = df.pivot(index='Hair', columns='Eye', values='Count')

    if return_full:
        return pivot
    else:
        return pivot.values

def with_pivot():
    data = pivot_table('HairEyeColor.csv')
    data = data.transpose()

    print('Pivot table')
    print(data, '\n')

    print('Pivot table (as asked)')
    print(data.values, '\n')

    eye_total = data.sum(axis=1)
    hair_total = data.sum(axis=0)
    total = data.to_numpy().sum()
    print('Eye color counts')
    print(eye_total, '\n')
    print('Hair color counts')
    print(hair_total, '\n')
    print('Count total')
    print(total, '\n')

    probabilities = data/total
    eye_marginal = eye_total/total
    hair_marginal = hair_total/total
    print('Probabilities')
    print(probabilities, '\n')
    print('Eye color marginals')
    print(eye_marginal, '\n')
    print('Hair color marginals')
    print(hair_marginal, '\n')
    print('Total probability')
    print(probabilities.to_numpy().sum(), '\n')

    print('P(blue eyes AND blond hair)')
    print(probabilities.loc['Blue', 'Blond'], '\n')
    print('P(brown hair)')
    print(hair_marginal['Brown'], '\n')
    print('P(brown eyes | red hair)')
    print(probabilities.loc['Brown', 'Red'] / hair_marginal['Red'], '\n')
    print('P((brown eyes OR blue eyes) AND (red hair OR blond hair))')
    print(probabilities.loc[['Brown', 'Blue'], ['Red', 'Blond']].to_numpy().sum(), '\n')
    print('P((brown eyes OR blue eyes OR red hair OR blond hair))')
    print(eye_marginal['Brown'] + eye_marginal['Blue'] + 
            hair_marginal['Red'] + hair_marginal['Blond'] - 
            probabilities.loc[['Brown', 'Blue'], ['Red', 'Blond']].to_numpy().sum(), '\n')
    print('P(NOT ((green eyes OR hazel eyes) AND (black hair OR brown hair))')
    print(1 - probabilities.loc[['Green', 'Hazel'], ['Black', 'Brown']].to_numpy().sum(), '\n')

def without_pivot():
    df = pd.read_csv('HairEyeColor.csv')

    eye_total = df.groupby('Eye').sum()
    hair_total = df.groupby('Hair').sum()
    total = df['Count'].sum()

    print(eye_total)
    print(hair_total)
    print(total)

    df['Probability'] = df['Count']/total
    eye_marginal = eye_total/total
    hair_marginal = hair_total/total

    print(df)
    print(eye_marginal)
    print(hair_marginal)

    print(df[(df.Eye == 'Blue') & (df.Hair == 'Blond')].Probability.sum())
    print(df[(df.Hair == 'Brown')].Probability.sum())
    print(df[(df.Eye == 'Brown') & (df.Hair == 'Red')].Probability.sum() / df[df.Hair == 'Red'].Probability.sum())
    print(df[((df.Eye == 'Brown') | (df.Eye == 'Blue')) & ((df.Hair == 'Red') | (df.Hair == 'Blond'))].Probability.sum())
    print(df[(df.Eye == 'Brown') | (df.Eye == 'Blue') | (df.Hair == 'Red') | (df.Hair == 'Blond')].Probability.sum())

    test_df = pd.DataFrame(columns=['Eye', 'Hair', 'Test_Probability'])
    for eye_index, eye_row in eye_marginal.iterrows():
        for hair_index, hair_row in hair_marginal.iterrows():
            test_df = test_df.append(pd.DataFrame({'Eye': eye_index, 'Hair': hair_index, 'Test_Probability': eye_row * hair_row}), ignore_index=True)
    print(test_df)
    df = df.merge(test_df)
    print(df)
    df['Is independent'] = df['Probability'] == df['Test_Probability']
    print(df)
def main():
    with_pivot()
    without_pivot()

if __name__ == '__main__':
    main()
