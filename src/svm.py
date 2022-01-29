import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SequentialFeatureSelector





# grid search best hyperparameters
def grid_search(param, X_train, y_train):
    modelsvr = SVR()

    grids = GridSearchCV(modelsvr, param_grid=param, cv=3, n_jobs=-1, verbose=2) # n_jobs: use all cores, verbose=see something of the process
    grids.fit(X_train, y_train)
    print(grids.best_params_)




def svr(X_train, X_test, y_train, y_test, C, coef0, degree, gamma, kernel):

    svr = SVR(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel)
    svr_fit = svr.fit(X_train, y_train)


    #print(y_test[:5])
    #print(y_pred[:5])

    #rmse = mean_squared_error(y_test, y_pred, squared=False)
    pred_train = svr.predict(X_train)
    normal_rmse(pred_train, y_train)

    cross_val(svr, X_train, y_train)

    pred_test = svr_fit.predict(X_test)
    test_error(pred_test, y_test)


def get_root_mean_squared_error(predictions, labels):
    rsme = 0
    for pred, label in zip(predictions, labels):
        rsme += (label - pred)**2
    return np.sqrt(1/len(predictions) * rsme)

def normal_rmse(pred_train, y_train):
    # normal root mean squared error
    rsme_normal = mean_squared_error(y_train, pred_train, squared=False)
    print(f'Train error: {rsme_normal}')


def cross_val(svr, X_train, y_train):
    # cross validation
    scorer = make_scorer(get_root_mean_squared_error)
    scores = cross_val_score(svr, X_train, y_train, cv=5, scoring=scorer)
    print(f'mean cross validation error: {scores.mean()}')

def test_error(pred_test, y_test):
    # test error
    test_error = mean_squared_error(y_test, pred_test, squared=False)
    print(f'test error: {test_error}')







def main():
    data = pd.read_csv("../pre-processed-data/data_pca.csv")
    if 'index' in data.columns:
        data = data.drop(['index'],axis=1)

    # ============= Normal SVR =============

    X = data.loc[:,data.columns != 'y'].to_numpy()
    y = data.loc[:,data.columns == 'y'].values.flatten()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

    # Grid search best hyperparameters
    param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')}
    #grid_search(param, X_train, y_train)



    # feature selection
    #features = data.loc[:,data.columns != 'y'].columns
    #feature_selection(X_train, X_test, y_train, y_test, features, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')

    # train model
    svr(X_train, X_test, y_train, y_test, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')


    # ============= SVR + Remove highly correlated features =============

    # remove highly correlated features

    X_corr = data.loc[:,data.columns != 'y']

    cor_matrix = X_corr.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [i for i in range(len(upper_tri.columns)) if any(upper_tri.iloc[:,i] > 0.90)]
    X_no_corr = X_corr.drop(X_corr.columns[to_drop], axis=1)


    X_final = X_no_corr.to_numpy()

    X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(X_final, y, test_size=0.33, random_state=69)

    # train model
    print("\n\nhighly correlated features removed:")
    svr(X_train_corr, X_test_corr, y_train_corr, y_test_corr, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')


    # ============= SVR + Remove highly correlated features + only features generated using forward feature selection =============

    feature_list_normal_rmse = ['PEOE_VSA13', 'fr_ester', 'PEOE_VSA7', 'PEOE_VSA10', 'fr_C_O', 'fr_NH0', 'BCUT2D_MRHI', 'NumAromaticCarbocycles', 'PEOE_VSA5', 'fr_imidazole', 'SlogP_VSA1', 'SMR_VSA5', 'fr_Ar_OH', 'SlogP_VSA2', 'EState_VSA3', 'EState_VSA4', 'FractionCSP3', 'FpDensityMorgan1', 'BCUT2D_CHGHI', 'VSA_EState4', 'fr_Al_OH', 'fr_ketone', 'MolLogP', 'HallKierAlpha', 'PEOE_VSA6', 'BalabanJ', 'PEOE_VSA2', 'EState_VSA6']
    feature_list_test_error = ['SMR_VSA10', 'fr_SH', 'SlogP_VSA7', 'PEOE_VSA3', 'fr_furan', 'SMR_VSA9', 'fr_Al_OH', 'fr_Imine', 'VSA_EState2', 'fr_aldehyde', 'BCUT2D_MWHI', 'MolLogP']
    #feature_list_cross_val = ['fr_NH2', 'fr_Ar_N', 'FractionCSP3', 'fr_Ndealkylation1', 'MolLogP']
    feature_list_cross_val = ['0', '2', '74', '41', '25', '1', '15', '67', '33']

    X_feature_selected = X_no_corr[feature_list_cross_val]

    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_feature_selected, y, test_size=0.33, random_state=69)

    # train model
    print("\n\nhighly correlated features removed, features selected :")
    svr(X_train_sel, X_test_sel, y_train_sel, y_test_sel, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')


    # ============= PCA + only features generated using forward feature selection =============



    # Train/test split

    # Grid search best hyperparameters
    param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')}

    feature_list_cross_val = ['0', '2', '74', '41', '25', '1', '15', '67', '33']

    X_feature_pca = data[feature_list_cross_val]

    X = data.loc[:,data.columns != 'y'].to_numpy()
    y = data.loc[:,data.columns == 'y'].values.flatten()

    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_feature_pca, y, test_size=0.33, random_state=69)

    print("\n\nPCA, features selected :")
    svr(X_train_pca, X_test_pca, y_train_pca, y_test_pca, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')

main()




