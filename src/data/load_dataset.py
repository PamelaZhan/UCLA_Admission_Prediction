
import pandas as pd
from ..logging.logging import logging_decorator

@logging_decorator
def load_and_preprocess_data(data_path):
    
    # Import the data from 'Admission.csv'
    df = pd.read_csv(data_path)

    # drop Serial_No
    df = df.drop('Serial_No', axis=1)

    return df
