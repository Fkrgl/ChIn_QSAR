'''This script generates molecular descriptors for the molecules'''

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np



def get_feature_matrix(suppl):
    # Descriptors._descList is a list of all available descriptors
    print(len(Descriptors._descList))
    # generate descriptor caculator from Descriptors._descList
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    # print descriptions of descriptors
    print(calc.GetDescriptorSummaries())

    # feature matrix
    X = []
    # labels
    y = []
    print(X)
    for mol in suppl:
        X.append(list(calc.CalcDescriptors(mol)))
        y.append(float(mol.GetProp('pLC50')))
    # rows are samples, columns are features
    X = np.array(X)
    print(X.shape)
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

# Frequencies of diffrent atoms in training data
{'C': 2766, 'N': 177, 'O': 446, 'Na': 1, 'Cl': 173, 'S': 28, 'Br': 23, 'P': 8, 'F': 33, 'I': 2}

def plot_p50_frequencies(y):
    """
    plots a histogram of the pLC50 values
    :param y: vector with pLC50 values
    """
    counts, bins = np.histogram(y, bins=20)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()

def remove_null_features(feature_matrix, threshold):
    '''
    removes less diveres features (feature value counts less than threshold)
    '''
    c = Counter(feature_matrix.iloc[:,0])
    print(c)
    idx_drop = []
    for i in range(len(feature_matrix.columns)):
        if len(Counter(feature_matrix.iloc[:,i])) <= threshold:
            print(feature_matrix.columns[i])
            print(Counter(feature_matrix.iloc[:,i]))
            idx_drop.append(i)
    print(f'the following columns are dropped {feature_matrix.columns[idx_drop]}')
    print(f'total number of dropped columns: {len(idx_drop)}')
    feature_matrix = feature_matrix.drop(feature_matrix.columns[idx_drop], axis=1)
    return feature_matrix

def conduct_pca(X):
    n_components = 75
    pca = PCA(n_components=n_components)
    # do pca
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(f'sum of explained variance: {sum(pca.explained_variance_ratio_)}')
    transformed_data = pca.transform(X)
    return transformed_data

def count_nan(feature_matrix):
    '''
    function counts how often NaN appears in a column to estimate the number of samples that have to be dropped
    '''
    for i in range(len(feature_matrix.columns)):
        if feature_matrix.iloc[:, i].isnull().any():
            print(f'feature {feature_matrix.columns[i]} has {feature_matrix.iloc[:, i].isnull().sum()} NaN values')

# check for atoms
def check_for_atom(atom_symbol, mol):
    '''
    Returns 1 if a atom is present
    '''
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == atom_symbol:
            return 1
    return 0

# check for aromaticity
def check_for_arom(mol):
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            return 1
    return 0

def main():
    # read in mols
    suppl = Chem.SDMolSupplier('../qspr-dataset-02.sdf/qspr-dataset-02.sdf')
    # get standardized feature matrix and labels
    feature_matrix, y = get_feature_matrix(suppl)
    feature_matrix = remove_null_features(feature_matrix, 1)
    # remove all columns which have NaN values
    '''only one sample gets dropped which schows NaN only in ca. 13 columns. Maybe consider to remove the features to
    save the one sample (look at what the features are)
    '''
    feature_matrix['y'] = y
    feature_matrix = feature_matrix.dropna().reset_index(drop=True)
    feature_matrix.to_csv('../pre-processed-data/data.csv')
    # get more filtered features (drop everything with less than three different values)
    feature_matrix_strict = remove_null_features(feature_matrix, 3)
    feature_matrix_strict.to_csv('../pre-processed-data/data_strict.csv')
    pca_transformed_data = conduct_pca(feature_matrix.iloc[:,:-1])
    pca_transformed_data = pd.DataFrame(pca_transformed_data)
    pca_transformed_data['y'] = feature_matrix['y']
    pca_transformed_data.to_csv('../pre-processed-data/data_pca.csv')
    print(pca_transformed_data.shape)
    print(feature_matrix.shape)
    print(pca_transformed_data['y'].any() == feature_matrix['y'].any())
    print(feature_matrix.head())

main()