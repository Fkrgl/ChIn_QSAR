import pandas as pd
import numpy as np
from rdkit import Chem

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

import argparse


import feature_generation as f_g

def train_model(X_train, X_test, y_train, y_test, C, coef0, degree, gamma, kernel):
    """
    Trains a SVR model using the hyperparameters generated using grid search (grid_search_svr.py) and a training dataset.
    """
    svr = SVR(C=C, coef0=coef0, degree=degree, gamma=gamma, kernel=kernel)
    svr_fit = svr.fit(X_train, y_train)

    return svr_fit


# ====================================== Functions used for testing the model ======================================

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

# ==================================================================================================================

def write_prediction(comp_names, pred, file_name):
    """
    writes the required output to a csv file,
    first col: compound indices
    sec col: Predicted pLC50 values
    """
    df = pd.DataFrame({'Compound': comp_names, 'pLC50_pred': pred})
    df.to_csv(file_name, index=False)




def main():
    # some stuff for the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data',
                        metavar='--input_data',
                        type=str,
                        help=' Absolute Path to dataset, for which the prediction should be made')

    parser.add_argument('--output_path',
                        metavar='--output_path',
                        type=str,
                        help='Path, where the output .csv should be stored')

    args = parser.parse_args()
    input_data = args.input_data
    output = args.output_path

    ''' ====================== train Model using training dataset ====================== '''

    # read the training dataset.(High-correlated features removed)
    # data = pd.read_csv('/home/mp/Documents/Cheminformatics/Assignments/Project/ChIn_QSAR/dat/trainings_data.csv')
    data = pd.read_csv('../dat/trainings_data.csv')
    if 'index' in data.columns:
        data = data.drop(['index'],axis=1)

    # ============= SVR + Remove highly correlated features + only features generated using forward feature selection =============

    # split test dataset in X and y
    X = data.loc[:,data.columns != 'y']
    y = data.loc[:,data.columns == 'y'].values.flatten()



    # apply much smaller set of features generated using Forward Feature Selection (feature_selection.py)
    feature_list_cross_val = ['NumAromaticCarbocycles', 'MolLogP', 'HallKierAlpha', 'VSA_EState10', 'BCUT2D_LOGPHI', 'PEOE_VSA14', 'FractionCSP3', 'PEOE_VSA6']

    X_feature_selected = X[feature_list_cross_val]

    X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_feature_selected, y, test_size=0.33, random_state=42)


    # train SVR model
    model = train_model(X_train_sel, X_test_sel, y_train_sel, y_test_sel, C=1, coef0=10, degree=3, gamma='scale', kernel='poly')

    '''test_pred = model.predict(X_test_sel)
    test = pd.DataFrame(X_test_sel)
    test["y"] = y_test_sel
    test["y_pred"] = test_pred
    test.to_csv("test_pred.csv")'''


    ''' ====================== predict new dataset ====================== '''

    # read in mols
    suppl = Chem.SDMolSupplier(input_data)
    # get standardized feature matrix and labels
    feature_matrix, y = f_g.get_feature_matrix(suppl)
    feature_matrix = feature_matrix[feature_list_cross_val]
    # remove all columns which have NaN values
    feature_matrix = feature_matrix.dropna().reset_index(drop=True)


    print("\ntest error")
    dataset_prediction = model.predict(feature_matrix)
    test_error(dataset_prediction, y)


    # create output
    row_ids = feature_matrix.index.values
    write_prediction(row_ids, dataset_prediction, output)

main()