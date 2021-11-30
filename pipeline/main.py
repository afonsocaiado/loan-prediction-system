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


#DATA JOINING

joined_data = data_preparation.prep_data('train')
#joined_data_competition = data_preparation.prep_data('competition')

#DATA CLEANING

joined_data["status"].replace({-1: 1, 1: -1}, inplace=True)

#joined_data_competition["status"].replace({-1: 1, 1: -1}, inplace=True)

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
#classifier = DecisionTreeClassifier()
#classifier = RandomForestClassifier()
#classifier = GaussianNB()
classifier = LogisticRegression(multi_class="multinomial", max_iter=1000)
#classifier = SVC(probability=True)
#classifier = Perceptron()

# Train the classifier on the training set
classifier.fit(training_inputs, training_labels)

# Validate the classifier on the testing set using classification accuracy
score = classifier.score(testing_inputs, testing_labels)

# Obtain predictions
predicted = classifier.predict(testing_inputs)
prediction_proba = classifier.predict_proba(testing_inputs)

#(predicted)
#print(testing_labels)

#print(prediction_proba)

#print("Accuracy score: {}".format(score))

#print("AUC score: {}".format(roc_auc_score(testing_labels, prediction_proba[:,1])))

#OBTAINING SUBMISSION

#submission_file.create(joined_data_competition,classifier)







