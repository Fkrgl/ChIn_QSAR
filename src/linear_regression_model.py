from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np

def get_root_mean_squared_error(predictions, labels):
    rsme = 0
    for pred, label in zip(predictions, labels):
        rsme += (label - pred)**2
    return np.sqrt(1/len(predictions) * rsme)


def print_model_performance(reg, X_train, X_test, y_train, y_test):
    '''
    get perfomance of a model
    '''
    predictions = reg.predict(X_train)
    print(f'prediction: {list(predictions[:5])}')
    print(f'y: {y_train[:5]}')
    rsme_normal = np.sqrt(mean_squared_error(y_train, predictions))
    print(f'train error: {rsme_normal}')

    # cross validation
    model = Lasso()
    scorer = make_scorer(get_root_mean_squared_error)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
    print(f'mean cross validation error: {scores.mean()}')

    # get test error
    predictions_training = reg.predict(X_test)
    print(f'test error: {np.sqrt(mean_squared_error(y_test, predictions_training))}')

def drop_highly_correlated_features(X, cut_off):
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [i for i in range(len(upper_tri.columns)) if any(upper_tri.iloc[:, i] > cut_off)]
    X_no_corr = X.drop(X.columns[to_drop], axis=1)
    return X_no_corr

def main():
    data = pd.read_csv('../pre-processed-data/data_strict.csv', index_col=0)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1].values

    ########################################## normal regression #######################################################

    # split data into test and trainings set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    reg = LinearRegression().fit(X_train, y_train)
    #print_model_performance(reg, X_train, X_test, y_train, y_test)



    ######################################### normal regression + covariance/variance  #################################

    # remove features with high covariance
    # drop all features that are highly correlated
    X_no_corr = drop_highly_correlated_features(X, 0.9)
    X_train, X_test, y_train, y_test = train_test_split(X_no_corr, y, test_size=0.33, random_state=42)
    reg = LinearRegression().fit(X_train, y_train)
    print_model_performance(reg, X_train, X_test, y_train, y_test)
    print((X_no_corr.shape))

    ############################################## lasso regression ####################################################
    alpha = [0.01, 0.1, 1.0, 10, 100, 1000, 10000]
    param_grid = {'alpha' : alpha}
    scorer = make_scorer(get_root_mean_squared_error)
    print(param_grid)
    clf = Lasso()
    grid = GridSearchCV(estimator=clf, param_grid=param_grid,
                        cv=KFold(n_splits=5), verbose=5, scoring=scorer)
    grid_results = grid.fit(X_no_corr, y)

    # Summarize the results in a readable format
    print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

    means = grid_results.cv_results_['mean_test_score']
    stds = grid_results.cv_results_['std_test_score']
    params = grid_results.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print('{0} ({1}) with: {2}'.format(mean, stdev, param))

    clf = Lasso(alpha=0.01).fit(X_train, y_train)
    print_model_performance(clf, X_train, X_test, y_train, y_test)

main()