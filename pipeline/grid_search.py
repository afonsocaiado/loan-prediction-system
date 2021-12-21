from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import data_preparation

'''
This is a script to narrow donw the most optimal parameters
'''

data = data_preparation.prep_data('train')

X = data.drop('status', axis = 1)
y = data.status
#splitting
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.33, random_state = 5, stratify=y)

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [4,5, 6],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [4,5,6],
    'n_estimators': [30, 35, 41, 50,60]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 2, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)

def evaluate(model, test_features, test_labels):
    y_pred = model.predict_proba(test_features)
    auc_score = metrics.roc_auc_score(y_test, y_pred[:,1])
    print (auc_score)
    
best_grid = grid_search.best_estimator_
evaluate(best_grid,X_test,y_test)