import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dev = pd.read_csv('data\loan_train.csv',delimiter=";")
dev_lenght = len(dev)
#split up dev data into training and test data and create copies to keep original data
data_split = round((2/3)*dev_lenght)
train = (dev.iloc[:data_split,:]).copy()
test = (dev.iloc[data_split+1:,:]).copy()
print(train.columns)
print(train.shape)
print(dev['status'].value_counts(normalize=True))
dev['status'].value_counts().plot.bar(title = 'Status of loans')
dev.groupby('status')['amount'].mean().plot.bar(title = 'average loan by status')
dev['amount_month'] = dev['amount']/dev['duration']
print(dev.head())
dev.groupby('status')['amount_month'].mean().plot.bar(title = 'average monthly loan by status')
    
