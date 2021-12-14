from imblearn.over_sampling import SMOTE

import data_preparation

data = data_preparation.prep_data('train')
competition = data_preparation.prep_data('competition')
def SMOTE_sample(X, y):
    smote = SMOTE()
    
    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(X, y)
    return x_smote, y_smote




