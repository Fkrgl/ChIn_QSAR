import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector

def get_root_mean_squared_error(predictions, labels):
    rsme = 0
    for pred, label in zip(predictions, labels):
        rsme += (label - pred)**2
    return np.sqrt(1/len(predictions) * rsme)

def cross_val(svr, X_train, y_train):
    # cross validation
    scorer = make_scorer(get_root_mean_squared_error)
    scores = cross_val_score(svr, X_train, y_train, cv=5, scoring=scorer)
    return scores.mean()

def generate_feature_subset(X, feature_set, feature):
    feature_set.add(feature)
    return X[feature_set]

def feature_selection_cross_val(X_train, X_test, y_train, y_test, features, C, coef0, degree, gamma, kernel):
    """
    performs the forward feature selection as seen in the lecture.
    Iteratively adds descriptors to set of descriptors if they improve the score
    cross-validation was chosen as score
    """
    svr = SVR(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel)
    S = set()
    F = np.inf
    D = set(features)

    while len(D) > 0:
        F_tmp = np.inf
        D_tmp = ""
        count= 0
        for feature in D:
            count +=1
            X_train_features = generate_feature_subset(X_train, S, feature).to_numpy() # optimize "normal rmse"

            S.remove(feature)
            svr_fit = svr.fit(X_train_features, y_train)

            score = cross_val(svr, X_train_features, y_train) # optimize cross_validation
            if score < F_tmp:
                F_tmp = score
                D_tmp = feature
        if F_tmp < F:
            print(F_tmp, F)
            F = F_tmp
            S.add(D_tmp)
            print(S)
            D.remove(D_tmp)
        else:
            break

    print("len(S)",len(S))
    print(S)


def main():
    data = pd.read_csv('../dat/trainings_data.csv')

    if 'index' in data.columns:
        data = data.drop(['index'],axis=1)

    # remove highly correlated features

    X = data.loc[:,data.columns != 'y']
    y = data.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    # feature selection
    features = X.loc[:,X.columns != 'y'].columns.to_list()[1:]
    #feature_selection_forward(X_train, X_test, y_train, y_test, features, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')
    feature_selection_cross_val(X_train, X_test, y_train, y_test, features, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')


main()
