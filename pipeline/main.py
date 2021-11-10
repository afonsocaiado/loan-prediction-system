import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import data_splitting

#DATA READING

account = pd.read_csv('../data/account.csv',delimiter=";")
card_train = pd.read_csv('../data/card_train.csv',delimiter=";")
client = pd.read_csv('../data/client.csv',delimiter=";")
disp = pd.read_csv('../data/disp.csv',delimiter=";")
district = pd.read_csv('../data/district.csv',delimiter=";")
loan_train = pd.read_csv('../data/loan_train.csv',delimiter=";")
#trans_train = pd.read_csv('../data/trans_train.csv',delimiter=";")

#DATA JOINING


#DATA CLEANING


#DATA SPLITTING

split_data = data_splitting.split(loan_train)


#DATA SAMPLING


#MODEL BUILDING

from sklearn.tree import DecisionTreeClassifier

# Create the classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training set
decision_tree_classifier.fit(split_data[0], split_data[1])

# Validate the classifier on the testing set using classification accuracy
score = decision_tree_classifier.score(split_data[2], split_data[3])

print(score)

#MODEL APPLYING AND OBTAINING PREDICTION