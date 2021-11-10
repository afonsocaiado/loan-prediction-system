import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split(data):
	
    train = data.copy()
    test = data.copy()
    
    train = train.loc[(train['date'] < 960400)]
    
    test = test.loc[test['date'] >= 960400]
    
    # Preparing data for classifier
    training_inputs = train[['date', 'amount',
                             'duration', 'payments']].values

    training_labels = train['status'].values


    testing_inputs = test[['date', 'amount',
                             'duration', 'payments']].values

    testing_labels = test['status'].values
	
    return [training_inputs, training_labels, testing_inputs, testing_labels]