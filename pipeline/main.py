import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import data_splitting
import data_sampling
import submission_file
import data_preparation

#DATA READING

account = pd.read_csv('../data/account.csv',delimiter=";")
card_train = pd.read_csv('../data/card_train.csv',delimiter=";")
client = pd.read_csv('../data/client.csv',delimiter=";")
disp = pd.read_csv('../data/disp.csv',delimiter=";")
district = pd.read_csv('../data/district.csv',delimiter=";")
loan_train = pd.read_csv('../data/loan_train.csv',delimiter=";")

competition = pd.read_csv('../data/loan_test.csv',delimiter=";")

#trans_train = pd.read_csv('../data/trans_train.csv',delimiter=";")

#DATA JOINING

joined_data = data_preparation.prep_data()

#DATA CLEANING


#DATA SPLITTING

split_data = data_splitting.split(joined_data)

train = split_data[0]
test = split_data[1]

#DATA SAMPLING

train = data_sampling.sampling(train)

#MODEL BUILDING

# Preparing data for classifier
training_inputs = train[['loan_date', 'amount', 'duration', 'payments',
       'frequency', 'account_date', 'region', 'inhabitants',
       'inhabitants < 499', 'inhabitants 500-1999', 'inhabitants 2000-9999',
       'inhabitants >10000', 'no. of cities ', 'ratio of urban inhabitants ',
       'average salary ', 'unemploymant 96',
       'enterpreneurs', 'crimes 96']].values

training_labels = train['status'].values


testing_inputs = test[['loan_date', 'amount', 'duration', 'payments',
       'frequency', 'account_date', 'region', 'inhabitants',
       'inhabitants < 499', 'inhabitants 500-1999', 'inhabitants 2000-9999',
       'inhabitants >10000', 'no. of cities ', 'ratio of urban inhabitants ',
       'average salary ', 'unemploymant 96',
       'enterpreneurs', 'crimes 96']].values

testing_labels = test['status'].values

#MODEL APPLYING AND OBTAINING PREDICTION

#Decision Tree

# Create the classifier
classifier = DecisionTreeClassifier()
classifier = RandomForestClassifier()
classifier = GaussianNB()
classifier = LogisticRegression()
classifier = SVC(probability=True)
#classifier = Perceptron()

# Train the classifier on the training set
classifier.fit(training_inputs, training_labels)

# Validate the classifier on the testing set using classification accuracy
score = classifier.score(testing_inputs, testing_labels)

# Obtain predictions
predicted = classifier.predict(testing_inputs)


print("Accuracy score: {}".format(score))

print("AUC score: {}".format(roc_auc_score(testing_labels, predicted)))

#OBTAINING SUBMISSION

submission_file.create(competition,classifier)







