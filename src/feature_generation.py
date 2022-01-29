'''This script generates molecular descriptors for the molecules'''

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import pandas as pd
import numpy as np


def check_for_anorganics(mol):
    '''
    Check weather a mol is anorganic. Returns True if the mol is anorganic (has no C).
    '''
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            return False
    return False


def get_anorganics(suppl):
    '''
    returns list of all anorganic molecules in the data
    '''
    anorganics = []
    for mol in suppl:
        if check_for_anorganics(mol):
            for atom in mol.GetAtoms():
                print(atom.GetSymbol())
    return anorganics


def get_feature_matrix(suppl):
    '''
    Generates molecular 2D discriptors for a molecule
    :param suppl: iter. mol object
    :return: feature matrix and labels
    '''
    # generate descriptor caculator from Descriptors._descList
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    print('initial number of desciptors: ', len(Descriptors._descList))
    # feature matrix
    X = []
    # labels
    y = []
    for mol in suppl:
        X.append(list(calc.CalcDescriptors(mol)))
        y.append(float(mol.GetProp('pLC50')))
    # rows are samples, columns are features
    X = np.array(X)
    # standardize features
    X_std = Z_score_normalization(X)
    feature_matrix = pd.DataFrame(data=X_std, columns=calc.GetDescriptorNames())
    return feature_matrix, y

def Z_score_normalization(feature_matrix):
    '''
    perform Z-score transformation to standardize the data
    '''
    # Create a scaler object
    sc = StandardScaler()
    # Fit the scaler to the features and transform
    X_std = sc.fit_transform(feature_matrix)
    return X_std

def drop_highly_correlated_features(X, cut_off):
    '''
    remove one of two features that have a pearson correlation coefficient above a cut_off
    :param X: feature matrix
    :param cut_off: threshold for correletion coeff
    :return: feature matrix without highly correlated features
    '''
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    to_drop = [i for i in range(len(upper_tri.columns)) if any(upper_tri.iloc[:, i] > cut_off)]
    X_no_corr = X.drop(X.columns[to_drop], axis=1)
    return X_no_corr

def remove_null_features(feature_matrix, threshold):
    '''
    removes less diveres features (feature value counts less than threshold)
    '''
    idx_drop = []
    for i in range(len(feature_matrix.columns)):
        if len(Counter(feature_matrix.iloc[:,i])) <= threshold:
            idx_drop.append(i)
    feature_matrix = feature_matrix.drop(feature_matrix.columns[idx_drop], axis=1)
    return feature_matrix

def conduct_pca(X):
    n_components = 75
    pca = PCA(n_components=n_components)
    pca.fit(X)
    transformed_data = pca.transform(X)
    print(f'{n_components} components contain {sum(pca.explained_variance_ratio_)*100:.4f} % of the variance')
    return transformed_data


def main():
    # read in mols
    suppl = Chem.SDMolSupplier('../dat/qspr-dataset-02.sdf')
    # get standardized feature matrix and labels
    feature_matrix, y = get_feature_matrix(suppl)
    # remove all columns which have NaN values
    feature_matrix['y'] = y
    feature_matrix = feature_matrix.dropna().reset_index(drop=True)
    feature_matrix = remove_null_features(feature_matrix, 3)
    feature_matrix.to_csv('../dat/trainings_data.csv', index=False)
    print('check for metal organics:')
    print(get_anorganics(suppl))
    print(f'final number of desciptors: {feature_matrix.shape[1]}')

    # pca
    pca_transformed_data = conduct_pca(feature_matrix.iloc[:,:-1])
    pca_transformed_data = pd.DataFrame(pca_transformed_data)
    pca_transformed_data['y'] = feature_matrix['y']
    pca_transformed_data.to_csv('../dat/trainings_data_pca.csv', index=False)


if __name__ == '__main__':
    main()