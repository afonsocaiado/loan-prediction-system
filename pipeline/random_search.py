import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


from sklearn import metrics


import data_preparation

'''
This is a script to randomly find the best hyperparameters for randomforest using RandomizedSearchCV
'''

data = data_preparation.prep_data('train')

X = data.drop('status', axis = 1)
y = data.status
#splitting
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.33, random_state = 5, stratify=y)

# Number of trees in random forest
n_estimators = [int(x) for x in range(1,600,3)]
# Number of features to consider at every split
max_features = [2, 3, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 100, num =  50)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 150, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

def evaluate(model, test_features, test_labels):
    y_pred = model.predict_proba(test_features)
    auc_score = metrics.roc_auc_score(y_test, y_pred[:,1])
    print (auc_score)
    

best_random = rf_random.best_estimator_
evaluate(best_random, X_test, y_test)

#{'n_estimators': 238, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 93, 'bootstrap': False}