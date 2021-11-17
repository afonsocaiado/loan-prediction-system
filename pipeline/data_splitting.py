import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def split(data):
	
    train = data.copy()
    test = data.copy()
    
    train = train.loc[(train['loan_date'] < 960400)]
    
    test = test.loc[test['loan_date'] >= 960400]
    
    return [train, test]