import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector

"""
Here, the best Hyperparameters to train the SVR models are acquired using 
Grid Search
"""


# grid search best hyperparameters
def grid_search(param, X_train, y_train):
    modelsvr = SVR()

    grids = GridSearchCV(modelsvr, param_grid=param, cv=3, n_jobs=-1, verbose=2) # n_jobs: use all cores, verbose=see something of the process
    grids.fit(X_train, y_train)
    print(grids.best_params_)




def main():
    #data = pd.read_csv("../pre-processed-data/data_strict.csv")
    data = pd.read_csv('../dat/trainings_data.csv')
    if 'index' in data.columns:
        data = data.drop(['index'],axis=1)

    # ============= Normal SVR =============

    X = data.loc[:,data.columns != 'y'].to_numpy()
    y = data.loc[:,data.columns == 'y'].values.flatten()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

    # Grid search best hyperparameters
    param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')}
    grid_search(param, X_train, y_train)

main()