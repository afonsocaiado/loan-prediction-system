from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


import submission_file
import data_preparation

data = data_preparation.prep_data('train')
competition = data_preparation.prep_data('competition')

X = data.drop(['status','loan_id'], axis = 1)
y = data.status

#splitting
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.33, random_state = 5, stratify=y)

#Scale feature
# scaler = preprocessing.StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

model = RandomForestClassifier(max_depth=5, n_estimators=41, random_state=4)
#model = LogisticRegression(multi_class="multinomial", max_iter=1000, class_weight="balanced")

model.fit(X_train, y_train)
prediction_proba = model.predict_proba(X_test)
print("AUC score: {}".format(metrics.roc_auc_score(y_test, prediction_proba[:,1])))

submission_file.create(competition, model)