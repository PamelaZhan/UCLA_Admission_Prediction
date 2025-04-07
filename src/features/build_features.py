import pandas as pd
from ..logging.logging import logging_decorator

@logging_decorator
# transform features and create dummy features
def trans_features(df):
    
    # convert the target variable Admit_Chance into a categorical variable by using a threshold of 80%
    df['Admit_Chance'] = (df['Admit_Chance'] >=0.8).astype(int)

    # store the processed dataset in data/processed
    df.to_csv('data/processed/Processed_admissions.csv', index=None)

    # Separate the input features and target variable
    x = df.drop('Admit_Chance', axis=1)
    y = df['Admit_Chance']

    return x, y