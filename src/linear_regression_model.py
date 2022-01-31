from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

import pandas as pd
import numpy as np

def get_root_mean_squared_error(predictions, labels):
    rsme = 0
    for pred, label in zip(predictions, labels):
        rsme += (label - pred)**2
    return np.sqrt(1/len(predictions) * rsme)


def print_model_performance(reg, X_train, X_test, y_train, y_test, model):
    '''
    print model perfomance on training, cross valisation and test data set using the RMSE as scoring function
    '''

    # traings error
    predictions = reg.predict(X_train)
    rsme_normal = np.sqrt(mean_squared_error(y_train, predictions))
    print(f'\ntrain error: {rsme_normal}')

    # cross validation
    scorer = make_scorer(get_root_mean_squared_error)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
    print(f'mean cross validation error: {scores.mean()}')

    # test error
    predictions_training = reg.predict(X_test)
    print(f'test error: {np.sqrt(mean_squared_error(y_test, predictions_training))}')

def get_best_alpha(X, y):
    alpha = [0.01, 0.1, 1.0, 10, 100, 1000, 10000]
    param_grid = {'alpha': alpha}
    clf = Lasso()
    grid = GridSearchCV(estimator=clf, param_grid=param_grid,
                        cv=KFold(n_splits=5), verbose=5, scoring='neg_root_mean_squared_error')
    grid_results = grid.fit(X, y)

    # Summarize results
    print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
    return grid_results.best_params_['alpha']

def main():
    # load trainings data
    data = pd.read_csv('../dat/trainings_data.csv', index_col=0)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1].values

    ########################################## normal regression #######################################################

    # split data into test and trainings set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    reg = LinearRegression().fit(X_train, y_train)
    print_model_performance(reg, X_train, X_test, y_train, y_test, LinearRegression())


    ############################################## lasso regression ####################################################
    # find best alpha
    best_alpha = get_best_alpha(X, y)
    # train Lasso regression
    clf = Lasso(alpha=best_alpha).fit(X_train, y_train)
    print_model_performance(clf, X_train, X_test, y_train, y_test, Lasso())

main()
