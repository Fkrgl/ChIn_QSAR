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



def normal_rmse(pred_train, y_train):
    # normal root mean squared error
    rsme_normal = mean_squared_error(y_train, pred_train, squared=False)
    return rsme_normal

def generate_feature_subset(X, feature_set, feature):
    feature_set.add(feature)
    return X[feature_set]

def test_error(pred_test, y_test):
    # test error
    test_error = mean_squared_error(y_test, pred_test, squared=False)
    return test_error

def cross_val(svr, X_train, y_train):
    # cross validation
    scorer = make_scorer(get_root_mean_squared_error)
    scores = cross_val_score(svr, X_train, y_train, cv=5, scoring=scorer)
    return scores.mean()



def feature_selection_forward(X_train, X_test, y_train, y_test, features, C, coef0, degree, gamma, kernel):
    """
    performs the forward feature selection as seen in the lecture.
    Iteratively adds descriptors to set of descriptors if they improve the score
    As "Score" the "normal" RMSE was chosen
    """
    svr = SVR(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel)#.fit(X_train, y_train)
    S = set()
    F = np.inf
    D = set(features)
    f_1 = ['MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'NumValenceElectrons']
    f_2 = 'NumValenceElectrons'
    #print(X_train[f_1])
    while len(D) > 0:
        F_tmp = np.inf
        D_tmp = ""
        count= 0
        for feature in D:
            count +=1
            #X_train_features = X_train[D_tmp].to_numpy()
            X_train_features = generate_feature_subset(X_train, S, feature).to_numpy() # optimize "normal rmse"
            X_test_features = generate_feature_subset(X_test, S, feature).to_numpy() # optimize "test error"
            S.remove(feature)
            svr_fit = svr.fit(X_train_features, y_train)
            ''' The following parameters are selected based on which parameter should be optimized '''
            #pred_train = svr_fit.predict(X_train_features) # optimize "normal rmse"
            #pred_test = svr_fit.predict(X_test_features) # optimize "test error"
            ''' if (almost) all items in column are equal, prediction will not work ==> its probably better to remove them for good (fr_oxazole, fr_SH)'''
            #if not np.isnan(np.sum(pred_train)): # optimize "normal rmse"
            #score = normal_rmse(pred_train, y_train) # optimize "normal rmse"
            #if not np.isnan(np.sum(pred_test)): # optimize "test error"
                #score = test_error(pred_test, y_test) # optimize "test error"
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



'''def feature_selection(X_train, X_test, y_train, y_test, features, C, coef0, degree, gamma, kernel):
    svr = SVR(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel)#.fit(X_train, y_train)

    # feature selection
    poss_number_features = np.arange(1, len(features)+1, 10)

    param = {'n_features_to_select': poss_number_features}


    fs_backward = SequentialFeatureSelector(svr, direction="backward")#.fit(X_train, y_train)
    scorer = make_scorer(get_root_mean_squared_error)

    grid = GridSearchCV(estimator=fs_backward, param_grid=param, verbose=2, n_jobs=-1, scoring='neg_root_mean_squared_error', refit=False)
    grid.fit(X_train, y_train)
    features = list(features[grid.best_estimator_.support_])
    print(features)
    #print(fs_backward.get_support())'''


# print(features[fs_backward.get_support()])


def main():
    data = pd.read_csv("../pre-processed-data/data_pca.csv")

    if 'index' in data.columns:
        data = data.drop(['index'],axis=1)

    # remove highly correlated features

    X_corr = data.loc[:,data.columns != 'y']

    cor_matrix = X_corr.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [i for i in range(len(upper_tri.columns)) if any(upper_tri.iloc[:,i] > 0.90)]
    X_no_corr = X_corr.drop(X_corr.columns[to_drop], axis=1)
    y = data.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(X_no_corr, y, test_size=0.33, random_state=42)


    # feature selection
    features = X_no_corr.loc[:,X_no_corr.columns != 'y'].columns.to_list()[1:]
    feature_selection_forward(X_train, X_test, y_train, y_test, features, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')


main()