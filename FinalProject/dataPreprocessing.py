import pandas as pd
import numpy as np

data = pd.read_csv('adult.data', header=None, sep=', ', na_values='?')

# Total records after removing NA
data_dropna = data.dropna(axis=0, how='any')

# Drop final weights column
data_dropna = data_dropna.drop(columns=[2], axis=1)
data_dropna.columns = ['age', 'workclass', 'education', 'education-num',
                       'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'capital-gain', 'capital-loss',
                       'hours-per-week', 'native-country', '50K?']

# remove education column
data_dropna = data_dropna.drop(columns=['education'], axis=1)

# change workclass to scale
data_dropna[['workclass', '50K?']].groupby(['workclass']).count()
data_dropna['workclass_num'] = data_dropna['workclass'].replace(to_replace=['State-gov', 'Self-emp-not-inc',
                                                                             'Private', 'Federal-gov',
                                                                             'Local-gov', 'Self-emp-inc',
                                                                             'Without-pay'],
                                                                value=[3, 2, 1, 3, 3, 4, 0])

# change marital status to 2 class
data_dropna['married_num'] = 0
data_dropna.loc[data_dropna['marital-status'] == 'Married-AF-spouse', 'married_num'] = 1
data_dropna.loc[data_dropna['marital-status'] == 'Married-civ-spouse', 'married_num'] = 1

# change sex to numerical scale
data_dropna['sex_num'] = data_dropna['sex'].replace(to_replace=['Female', 'Male'], value=[0, 1])

# change native country to numerical scale (the U.S. or not)
data_dropna['country_num'] = 0
data_dropna.loc[data_dropna['native-country'] == 'United-States', 'country_num'] = 1

# change target to numerical scale
data_dropna['target'] = data_dropna['50K?'].replace(to_replace=['<=50K', '>50K'], value=[0, 1])


data_clean = data_dropna.drop(columns=['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                       'capital-gain', 'capital-loss', 'native-country', '50K?', ], axis=1)

# save into new file
data_clean.to_csv('adult_clean.csv', index=True)




